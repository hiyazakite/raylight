"""Safetensors (lazy-tensor) model context — zero-copy mmap + pinned-cache offload."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
import os

import torch

from ._base import ModelContext, ModelState, _compute_vram_budget, _ops_for_model_options
from raylight.utils.cache import CachedState
from raylight.utils.checksum import verify_model_checksum
from raylight.expansion.comfyui_lazytensors.loader import SafetensorMmapWrapper

if TYPE_CHECKING:
    from raylight.types import LoraManagerLike, WorkerConfigLike


class LazyTensorContext(ModelContext):
    """Context for Safetensors model loading with streaming mmap support.

    Features:
    - Zero-copy mmap sharing via OS page cache (like GGUF)
    - Streaming GPU transfer (per-tensor to avoid RAM spike)
    - Fallback to full clone if streaming fails
    """

    def __init__(self, use_mmap: bool = True):
        super().__init__(use_mmap)
        self._streaming_enabled = True

    # ─── Disk loading ────────────────────────────────────────

    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfigLike") -> Any:
        import safetensors.torch
        import safetensors
        sd = safetensors.torch.load_file(state.unet_path, device="cpu")
        try:
            with safetensors.safe_open(state.unet_path, framework="pt") as f:
                metadata = f.metadata() or {}
        except Exception:
            metadata = {}
        return sd, metadata

    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfigLike") -> Any:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)

    # ─── Pinned cache factory ────────────────────────────────

    @staticmethod
    def _make_pinned_cache(config: "WorkerConfigLike", model_path: str):
        """Select shared or private pinned cache based on worker topology."""
        is_shared = config.global_world_size > 1 and not config.is_fsdp
        if is_shared:
            from raylight.distributed_modules.pinned_cache import (
                SharedPinnedParamCache, make_cache_id,
            )
            cache_id = make_cache_id(model_path)
            is_writer = config.local_rank == 0
            print(
                f"[LazyTensorContext] Shared pinned cache (rank {config.local_rank}): "
                f"{config.global_world_size} workers share 1 buffer (parallel build)."
            )
            return SharedPinnedParamCache(
                cache_id=cache_id,
                is_writer=is_writer,
                local_rank=config.local_rank,
                world_size=config.global_world_size,
            )
        else:
            from raylight.distributed_modules.pinned_cache import PinnedParamCache
            return PinnedParamCache()

    # ─── Instantiation ───────────────────────────────────────

    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfigLike",
                          metadata: Any = None, **kwargs) -> Any:
        """Instantiate model with lazy state dict (zero-copy until load)."""
        import comfy.sd
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

        print("[LazyTensorContext] Wrapping state dict with LazySafetensors...")
        lazy_sd = wrap_state_dict_lazy(sd)

        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        load_options["custom_operations"] = _ops_for_model_options(load_options)

        model = comfy.sd.load_diffusion_model_state_dict(
            lazy_sd, model_options=load_options, metadata=metadata,
        )

        if model is None:
            raise RuntimeError(f"Could not load model: {state.unet_path}")

        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype

        model.mmap_cache = sd
        model.__class__ = RaylightModelPatcher
        model.pinned_param_cache = self._make_pinned_cache(config, state.unet_path)
        print("[LazyTensorContext] Pinned param cache attached (will build on first offload).")
        return model

    def instantiate_model_with_fallback(self, sd: Dict, state: ModelState,
                                        config: "WorkerConfigLike") -> Any:
        """Try streaming approach with fallback to full clone."""
        import comfy.sd

        try:
            if self._streaming_enabled:
                model = self.instantiate_model(sd, state, config)
                print("[LazyTensorContext] Streaming instantiation successful")
                return model
        except Exception as e:
            print(f"[LazyTensorContext] Streaming failed: {e}")
            print("[LazyTensorContext] Falling back to full clone...")
            self._streaming_enabled = False

        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

        isolated = {k: v.clone() for k, v in sd.items()}
        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        load_options["custom_operations"] = _ops_for_model_options(load_options)

        model = comfy.sd.load_diffusion_model_state_dict(isolated, model_options=load_options)
        if model is None:
            raise RuntimeError(f"Could not load model: {state.unet_path}")

        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype

        model.__class__ = RaylightModelPatcher
        model.pinned_param_cache = self._make_pinned_cache(config, state.unet_path)
        return model

    # ─── Streaming ───────────────────────────────────────────

    def stream_to_device(self, model: Any, device: torch.device) -> bool:
        """Stream model weights to GPU per-tensor (avoids RAM spike)."""
        mmap_cache = getattr(model, "mmap_cache", None)

        if not mmap_cache or not self._streaming_enabled:
            print(f"[LazyTensorContext] Using standard .to({device})...")
            if model is not None and hasattr(model, "model"):
                model.model.to(device)
            return False

        try:
            print(f"[LazyTensorContext] Streaming to {device} per-tensor...")
            wrapper = SafetensorMmapWrapper(mmap_cache)
            if model is not None and hasattr(model, "model"):
                transferred = wrapper.stream_to_model(model.model, device)
                print(f"[LazyTensorContext] Streamed {transferred} parameters to {device}")
            return True
        except Exception as e:
            print(f"[LazyTensorContext] Streaming transfer failed: {e}, falling back...")
            if model is not None and hasattr(model, "model"):
                model.model.to(device)
            return False

    # ─── Fast-reload (delegates to _fast_reload_common) ──────

    @staticmethod
    def _fast_reload_full(model: Any, inner: Any, device: torch.device) -> None:
        """Bypass model.load() after reload_to_cuda — params already on CUDA."""

        def _apply_safetensor_patches(
            *,
            model: Any,
            inner: Any,
            device: torch.device,
            patches: Dict,
            wrapper_patches: Dict,
            force_cast: bool,
        ) -> None:
            # (1) Re-apply LoRA / weight patches to CUDA params.
            if patches:
                for key in patches:
                    model.patch_weight_to_device(key, device_to=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # (2) Mark all modules as patched
            for n, m in inner.named_modules():
                if hasattr(m, "comfy_cast_weights"):
                    if not hasattr(m, "prev_comfy_cast_weights"):
                        m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_force_cast_weights = force_cast
                    wk = f"{n}.weight"
                    bk = f"{n}.bias"
                    m.weight_function = list(wrapper_patches.get(wk, []))
                    m.bias_function = list(wrapper_patches.get(bk, []))
                m.comfy_patched_weights = True

        ModelContext._fast_reload_common(
            model, inner, device,
            apply_patches_fn=_apply_safetensor_patches,
            lowvram=False,
        )

    # ─── Offload (Template Method hook) ──────────────────────

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "WorkerConfigLike",
    ) -> bool:
        """Fast pinned-RAM offload: cache CUDA params, free VRAM via storage resize."""
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import swap_model_to_mmap

        if model is None:
            return False

        pinned_cache = getattr(model, "pinned_param_cache", None)
        diffusion_model = getattr(model, "model", None)

        # --- Pinned-cache fast path ---
        if pinned_cache is not None and diffusion_model is not None:
            print(f"[LazyTensorContext {config.local_rank}] Pinned-RAM Offload: CUDA → pinned CPU...")

            is_partial = getattr(diffusion_model, "model_lowvram", False)

            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)

            if hasattr(model, "unpatch_model"):
                model.unpatch_model(device_to=None, unpatch_weights=True)

            try:
                if is_partial:
                    pinned_cache.offload_cuda_only(diffusion_model)
                else:
                    pinned_cache.offload_to_cpu(diffusion_model)
                if hasattr(model, "mmap_cache"):
                    del model.mmap_cache
            except Exception as e:
                print(f"[LazyTensorContext {config.local_rank}] Pinned offload error: {e}")

            model.current_device = torch.device("cpu")

            for target in (diffusion_model, getattr(diffusion_model, "diffusion_model", None)):
                if target is None:
                    continue
                for attr in ("_cached_pe", "cached_pe", "pe_cache", "_pe"):
                    if hasattr(target, attr):
                        delattr(target, attr)

            if lora_manager:
                lora_manager.clear_tracking()

            # Heavy cleanup (gc, cache flush, VRAM logging) delegated to
            # the MemoryPolicy in the template-method's after_offload().
            return False

        # --- Legacy fallback: mmap pointer swap ---
        print(f"[LazyTensorContext {config.local_rank}] Soft-Offload: Performing Pointer Swap (legacy)...")

        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)

        if hasattr(model, "unpatch_model"):
            model.unpatch_model(device_to=None, unpatch_weights=True)

        mmap_cache = getattr(model, "mmap_cache", None)
        if diffusion_model is not None and mmap_cache:
            swap_model_to_mmap(diffusion_model, mmap_cache)

        model.current_device = torch.device("cpu")

        print(f"[LazyTensorContext {config.local_rank}] Soft-Offload Complete (legacy).")
        return False

    # ─── Hot load ────────────────────────────────────────────

    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any], state_cache: Any) -> None:
        """Hot reload: restore from pinned RAM (fast) or mmap (legacy)."""
        pinned_cache = getattr(model, "pinned_param_cache", None)
        diffusion_model = getattr(model, "model", None)

        # --- Eagerly build cache if not built yet (first run) ---
        if (
            pinned_cache is not None
            and not pinned_cache.built
            and diffusion_model is not None
        ):
            print("[LazyTensorContext] Building pinned cache from CPU params...")
            try:
                pinned_cache.build(diffusion_model)
                if hasattr(model, "mmap_cache"):
                    del model.mmap_cache
                if state_cache is not None:
                    unet_path = getattr(model, "unet_path", None)
                    if unet_path:
                        cached = state_cache.get(unet_path)
                        if cached is not None and hasattr(cached, "state_dict"):
                            cached.state_dict = None
            except Exception as e:
                print(f"[LazyTensorContext] Eager cache build failed: {e}")

        # --- Pinned-cache fast path ---
        if pinned_cache is not None and pinned_cache.built and diffusion_model is not None:
            vram_limit = getattr(model, "vram_limit_bytes", 0)
            model_bytes = model.model_size() if hasattr(model, "model_size") else 0
            budget = (
                _compute_vram_budget(device, model_bytes, vram_limit_bytes=vram_limit)
                if model_bytes > 0
                else 0
            )

            if budget == 0:
                # ── FULL CUDA ──
                import time as _time
                t0 = _time.perf_counter()
                print("[LazyTensorContext] Pinned-RAM Hot-Reload (full): pinned CPU → CUDA...")
                try:
                    pinned_cache.reload_to_cuda(diffusion_model)
                    diffusion_model.device = device
                    self._fast_reload_full(model, diffusion_model, device)
                    model.current_device = device
                    dt = (_time.perf_counter() - t0) * 1000
                    print(f"[LazyTensorContext] Pinned-RAM Hot-Reload (full) complete ({dt:.0f} ms).")
                    return
                except Exception as e:
                    print(f"[LazyTensorContext] Pinned-RAM full reload failed: {e}, falling back to model.load()...")
                    try:
                        if hasattr(model, "load"):
                            model.load(device)
                        model.current_device = device
                        return
                    except Exception as e2:
                        print(f"[LazyTensorContext] model.load() fallback also failed: {e2}, trying mmap...")
            else:
                # ── PARTIAL CUDA ──
                budget_mb = budget / (1024 ** 2)
                model_mb = model_bytes / (1024 ** 2)
                print(
                    f"[LazyTensorContext] Pinned-RAM Hot-Reload (partial): "
                    f"model {model_mb:.0f} MB, VRAM budget {budget_mb:.0f} MB — "
                    f"streaming overflow from pinned RAM..."
                )
                try:
                    pinned_cache.reload_to_cpu(diffusion_model)
                    if diffusion_model is not None:
                        model.model.device = torch.device("cpu")
                    if hasattr(model, "load"):
                        model._skip_pinned_auto_restore = True
                        try:
                            model.load(device, lowvram_model_memory=budget)
                        finally:
                            model._skip_pinned_auto_restore = False
                    model.current_device = device
                    pinned_cache._on_cuda = True
                    print("[LazyTensorContext] Pinned-RAM Hot-Reload (partial) complete.")
                    return
                except Exception as e:
                    print(f"[LazyTensorContext] Pinned-RAM partial reload failed: {e}, falling back to mmap...")

        # --- Legacy fallback: mmap streaming ---
        print(f"[LazyTensorContext] Hot reload to {device} (mmap)...")

        unet_path = getattr(model, "unet_path", None)
        if unet_path and unet_path in state_cache:
            cached = state_cache.get(unet_path)
            if isinstance(cached, CachedState) and cached.checksum:
                if isinstance(getattr(model, "mmap_cache", None), dict):
                    verify_model_checksum(
                        model.mmap_cache, cached.checksum, cached.metadata,
                        context_tag="LazyTensorContext",
                    )

        if model is not None and hasattr(model, "load"):
            model.load(device)
            model.current_device = device
