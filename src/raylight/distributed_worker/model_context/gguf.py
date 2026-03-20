"""GGUF model context — mmap + pinned-RAM caching for quantised models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import os

import torch

from ._base import ModelContext, ModelState, _compute_vram_budget
from raylight.utils.cache import CachedState

if TYPE_CHECKING:
    from raylight.types import LoraManagerLike, WorkerConfigLike


class GGUFContext(ModelContext):
    """Context for GGUF model loading with mmap support."""

    # ─── Pinned mmap_cache swap ──────────────────────────────

    def load(self, state: ModelState, config: "WorkerConfigLike", state_cache: Any) -> Any:
        """Override base load to pin mmap_cache for DMA-speed reloads.

        After the base class sets ``model.mmap_cache = sd`` (mmap-backed),
        we swap every tensor with a pinned-RAM copy.  All subsequent
        ``model.load()`` / ``unpatch_model()`` read from pinned tensors,
        so CUDA transfers use DMA instead of page-faulting through mmap.

        For multi-GPU DP/xDiT the pinned copy lives in ``/dev/shm`` so
        all workers share ONE host-RAM copy instead of N.
        """
        model = super().load(state, config, state_cache)

        if model is not None and getattr(model, "mmap_cache", None):
            pinned, shm_handle = self._pin_mmap_cache(
                model.mmap_cache, config, state.unet_path
            )
            if pinned is not None:
                model.mmap_cache = pinned
                # Keep SharedMemory alive as long as model lives
                if shm_handle is not None:
                    model._pinned_shm = shm_handle
                # Replace mmap-backed sd in state_cache with the pinned
                # dict so the OS can unmap the GGUF file.
                cached = state_cache.get(state.cache_key)
                if isinstance(cached, CachedState):
                    cached.state_dict = pinned
                # Replace model's own nn.Parameters with pinned copies.
                self._swap_params_to_pinned(model, pinned)
                print("[GGUFContext] mmap_cache swapped to pinned RAM.")

        return model

    # ─── Param swap ──────────────────────────────────────────

    @staticmethod
    def _swap_params_to_pinned(model: Any, pinned: Dict[str, torch.Tensor]) -> None:
        """Replace mmap-backed model parameters with their pinned copies."""
        import gc
        import comfy.utils

        diff_model = getattr(model, "model", None)
        if diff_model is None:
            return

        param_dict = dict(diff_model.named_parameters())

        swapped = 0
        for key, pinned_tensor in pinned.items():
            for full_key in (key, f"diffusion_model.{key}"):
                if full_key in param_dict:
                    comfy.utils.set_attr_param(diff_model, full_key, pinned_tensor)
                    swapped += 1
                    break

        if swapped > 0:
            gc.collect()
            print(f"[GGUFContext] Replaced {swapped}/{len(pinned)} model params with pinned tensors.")

    # ─── Pinning ─────────────────────────────────────────────

    @staticmethod
    def _pin_mmap_cache(
        mmap_cache: Dict[str, torch.Tensor],
        config: "WorkerConfigLike",
        model_path: str,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Any]:
        """Replace mmap-backed tensors with pinned-RAM copies.

        Single-GPU  → private ``pin_memory()`` per tensor.
        Multi-GPU   → one shared ``/dev/shm`` buffer + ``cudaHostRegister``.
        FSDP        → skip (shards differ across ranks).
        """
        is_shared = config.global_world_size > 1 and not config.is_fsdp
        if is_shared:
            return GGUFContext._pin_mmap_shared(mmap_cache, config, model_path)
        else:
            return GGUFContext._pin_mmap_private(mmap_cache), None

    @staticmethod
    def _pin_mmap_private(mmap_cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single-GPU: pre-allocate pinned destination + parallel copy."""
        from raylight.expansion.comfyui_gguf.ops import GGMLTensor
        from concurrent.futures import ThreadPoolExecutor
        import os as _os

        # ── Phase 1: allocate pinned buffers (no data movement) ───────
        allocs: list = []
        total_bytes = 0
        for key, tensor in mmap_cache.items():
            is_ggml = isinstance(tensor, GGMLTensor)
            raw = tensor.as_subclass(torch.Tensor) if is_ggml else tensor
            p = torch.empty_like(raw).pin_memory()
            total_bytes += p.nbytes
            allocs.append((key, p, raw, is_ggml, tensor))

        # ── Phase 2: parallel memcpy (CPU-only, safe to thread) ───────
        def _copy(item):
            _key, _p, _raw, *_ = item
            _p.copy_(_raw)
            return item

        num_workers = min(4, max(1, _os.cpu_count() or 1))
        pinned: Dict[str, torch.Tensor] = {}

        def _store(item):
            key, p, _raw, is_ggml, tensor = item
            if is_ggml:
                pinned[key] = GGMLTensor(
                    p, tensor_type=tensor.tensor_type, tensor_shape=tensor.tensor_shape,
                )
            else:
                pinned[key] = p

        if len(allocs) > 32 and num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for item in pool.map(_copy, allocs):
                    _store(item)
        else:
            for item in allocs:
                _copy(item)
                _store(item)

        print(
            f"[GGUFContext] Pinned mmap→RAM (private, {num_workers} threads): "
            f"{len(pinned)} tensors, {total_bytes / 1e9:.2f} GB"
        )
        return pinned

    @staticmethod
    def _pin_mmap_shared(
        mmap_cache: Dict[str, torch.Tensor],
        config: "WorkerConfigLike",
        model_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Multi-GPU shared: one ``/dev/shm`` buffer + ``cudaHostRegister``."""
        import ctypes
        import hashlib
        import time
        from multiprocessing.shared_memory import SharedMemory
        import torch.distributed as dist
        from raylight.expansion.comfyui_gguf.ops import GGMLTensor
        from raylight.distributed_modules.pinned_cache import (
            _align_up, _cuda_host_register, _ALIGNMENT,
        )

        is_writer = config.local_rank == 0
        local_rank = config.local_rank
        world_size = config.global_world_size
        cache_id = hashlib.md5(model_path.encode()).hexdigest()[:16]
        shm_name = f"raylight_gguf_{cache_id}"

        # --- deterministic layout from sorted keys -----------------------
        layout = []
        offset = 0
        for key in sorted(mmap_cache.keys()):
            tensor = mmap_cache[key]
            is_ggml = isinstance(tensor, GGMLTensor)
            raw = tensor.as_subclass(torch.Tensor) if is_ggml else tensor
            aligned = _align_up(offset)
            nb = raw.nbytes
            layout.append((
                key, aligned, nb,
                tuple(raw.shape), raw.dtype,
                getattr(tensor, "tensor_type", None),
                getattr(tensor, "tensor_shape", raw.shape),
            ))
            offset = aligned + nb
        total_bytes = _align_up(offset)

        if total_bytes == 0:
            return mmap_cache, None

        # --- Step 1: Rank 0 creates /dev/shm segment ---
        if is_writer:
            try:
                old = SharedMemory(name=shm_name, create=False)
                old.close(); old.unlink()
            except FileNotFoundError:
                pass
            shm = SharedMemory(name=shm_name, create=True, size=total_bytes)

        # --- Step 2: Barrier — all ranks attach + register ---
        if dist.is_initialized():
            dist.barrier()

        if not is_writer:
            retries = 50
            for i in range(retries):
                try:
                    shm = SharedMemory(name=shm_name, create=False)
                    break
                except FileNotFoundError:
                    if i < retries - 1:
                        time.sleep(0.2)
            else:
                raise RuntimeError(
                    f"[GGUFContext] Rank {local_rank} timed out waiting for "
                    f"shm '{shm_name}'"
                )
            if shm.size < total_bytes:
                raise RuntimeError(
                    f"[GGUFContext] shm size mismatch: need {total_bytes}, got {shm.size}"
                )

        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        ok = _cuda_host_register(ptr, total_bytes)
        if not ok and is_writer:
            print("[GGUFContext] WARNING: cudaHostRegister failed (DMA will use staging buffer).")

        # --- Step 3: Each rank copies its slice (round-robin) ---
        my_count = 0
        for i, (key, off, nb, s_shape, dtype, tt, ts) in enumerate(layout):
            if i % world_size == local_rank:
                src = mmap_cache[key]
                raw = src.as_subclass(torch.Tensor) if isinstance(src, GGMLTensor) else src
                dst = torch.frombuffer(
                    memoryview(shm.buf)[off:off + nb], dtype=dtype,
                ).reshape(s_shape)
                dst.copy_(raw)
                my_count += 1

        # --- Step 4: Barrier — all copies complete ---
        if dist.is_initialized():
            dist.barrier()

        # --- Step 5: All ranks create full view dict ---
        pinned: Dict[str, torch.Tensor] = {}
        for key, off, nb, s_shape, dtype, tt, ts in layout:
            view = torch.frombuffer(
                memoryview(shm.buf)[off:off + nb], dtype=dtype,
            ).reshape(s_shape)
            if tt is not None:
                pinned[key] = GGMLTensor(view, tensor_type=tt, tensor_shape=ts)
            else:
                pinned[key] = view

        print(
            f"[GGUFContext] Pinned mmap→shm (rank {local_rank}): "
            f"{len(pinned)} tensors, {total_bytes / 1e9:.2f} GB — "
            f"copied {my_count} (shm={shm_name})"
        )
        return pinned, shm

    # ─── Disk loading ────────────────────────────────────────

    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfigLike") -> Any:
        from raylight.expansion.comfyui_gguf.loader import gguf_sd_loader
        sd, extra = gguf_sd_loader(state.unet_path)
        metadata = extra.get("metadata", {})
        return sd, metadata

    def prepare_state_dict(self, sd: Dict[str, torch.Tensor], config: "WorkerConfigLike") -> Dict[str, torch.Tensor]:
        """GGUF: pass-through (ComfyUI handles prefix stripping)."""
        return sd

    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfigLike") -> Any:
        return self.load_state_dict_mmap(state, config)

    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfigLike",
                          metadata: Any = None, **kwargs) -> Any:
        from raylight.expansion.comfyui_gguf.ops import GGMLOps
        from raylight.expansion.comfyui_gguf.nodes import GGUFModelPatcher
        import comfy.sd
        import inspect

        ops = GGMLOps()

        if state.dequant_dtype and state.dequant_dtype not in ("default", None):
            if state.dequant_dtype == "target":
                setattr(ops.Linear, "dequant_dtype", state.dequant_dtype)
            else:
                setattr(ops.Linear, "dequant_dtype", getattr(torch, state.dequant_dtype))

        if state.patch_dtype and state.patch_dtype not in ("default", None):
            if state.patch_dtype == "target":
                setattr(ops.Linear, "patch_dtype", state.patch_dtype)
            else:
                setattr(ops.Linear, "patch_dtype", getattr(torch, state.patch_dtype))

        print(f"[GGUFContext DEBUG] Instantiate Config | Dequant: {state.dequant_dtype} | Patch: {state.patch_dtype}")

        isolated = sd.copy()
        gguf_meta = metadata if metadata is not None else getattr(self, "_temp_gguf_metadata", {})

        build_kwargs: Dict[str, Any] = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            build_kwargs["metadata"] = gguf_meta

        model_options = state.model_options.copy()
        model_options["custom_operations"] = ops

        if model_options.get("cache_patched_weights"):
            setattr(ops.Linear, "cache_patched_weights", True)
            print(f"[GGUFContext] Enabling cache_patched_weights for {state.unet_path}")

        model = comfy.sd.load_diffusion_model_state_dict(
            isolated, model_options=model_options, **build_kwargs,
        )

        if model is None:
            raise RuntimeError(f"Could not load GGUF model: {state.unet_path}")

        model = GGUFModelPatcher.clone(model)
        model.gguf_metadata = gguf_meta
        return model

    def _apply_ops_config(self, ops: Any, dequant_dtype: Any, patch_dtype: Any) -> None:
        """Helper to apply GGUF ops configuration."""
        print(f"[GGUFContext DEBUG] Applying Ops Config | Dequant: {dequant_dtype} | Patch: {patch_dtype}")
        if dequant_dtype and dequant_dtype not in ("default", None):
            if dequant_dtype == "target":
                setattr(ops.Linear, "dequant_dtype", dequant_dtype)
            else:
                setattr(ops.Linear, "dequant_dtype", getattr(torch, dequant_dtype, None))

        if patch_dtype and patch_dtype not in ("default", None):
            if patch_dtype == "target":
                setattr(ops.Linear, "patch_dtype", patch_dtype)
            else:
                setattr(ops.Linear, "patch_dtype", getattr(torch, patch_dtype, None))

        print(
            f"[GGUFContext DEBUG] Ops Config Result | "
            f"Linear.dequant_dtype: {getattr(ops.Linear, 'dequant_dtype', 'None')} | "
            f"Linear.patch_dtype: {getattr(ops.Linear, 'patch_dtype', 'None')}"
        )

        if dequant_dtype == "cache_patched_weights" or (
            isinstance(dequant_dtype, dict) and dequant_dtype.get("cache_patched_weights")
        ):
            setattr(ops.Linear, "cache_patched_weights", True)

    # ─── Offload (Template Method hook) ──────────────────────

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "WorkerConfigLike",
    ) -> bool:
        """GGUF Soft-Offload via pointer swap.

        ``GGUFModelPatcher.unpatch_model()`` replaces CUDA params with
        pinned-backed ``mmap_cache`` entries.
        """
        if model is None:
            return False

        inner = getattr(model, "model", None)
        is_partial = inner is not None and getattr(inner, "model_lowvram", False)
        tag = "partial" if is_partial else "full"
        print(f"[GGUFContext {config.local_rank}] GGUF Soft-Offload ({tag}): pointer swap to pinned mmap_cache...")

        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)

        model.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=True)
        model.current_device = torch.device("cpu")

        print(f"[GGUFContext {config.local_rank}] GGUF Soft-Offload Complete.")
        return False

    # ─── Hot load ────────────────────────────────────────────

    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any], state_cache: Any) -> None:
        """GGUF Hot-Reload: budget-aware fast path."""
        if model is None:
            return

        import time
        t0 = time.perf_counter()

        if hasattr(model, "model_options"):
            ops = model.model_options.get("custom_operations")
            if ops:
                self._apply_ops_config(
                    ops,
                    reload_params.get("dequant_dtype"),
                    reload_params.get("patch_dtype"),
                )

        # Re-hydrate mmap_cache if lost
        if not getattr(model, "mmap_cache", None):
            unet_path = getattr(model, "unet_path", None)
            if unet_path and unet_path in state_cache:
                cached = state_cache.get(unet_path)
                if isinstance(cached, CachedState):
                    print("[GGUFContext] Re-hydrating mmap_cache from cache")
                    model.mmap_cache = cached.state_dict

        inner = getattr(model, "model", None)
        if inner is None:
            if hasattr(model, "load"):
                model.load(device, force_patch_weights=True)
                model.current_device = device
            print("[GGUFContext] Hot-load complete (fallback).")
            return

        # ── Budget calculation ────────────────────────────────
        vram_limit = getattr(model, "vram_limit_bytes", 0)
        model_bytes = model.model_size() if hasattr(model, "model_size") else 0
        budget = (
            _compute_vram_budget(device, model_bytes, vram_limit_bytes=vram_limit)
            if vram_limit > 0 and model_bytes > 0
            else 0
        )

        if budget > 0:
            budget_mb = budget / (1024 ** 2)
            model_mb = model_bytes / (1024 ** 2)
            print(
                f"[GGUFContext] Hot-Reload (partial): "
                f"model {model_mb:.0f} MB, VRAM budget {budget_mb:.0f} MB..."
            )
            try:
                model.load(device, force_patch_weights=True, lowvram_model_memory=budget)
                model.current_device = device
                dt = (time.perf_counter() - t0) * 1000
                print(f"[GGUFContext] Hot-Reload (partial) complete ({dt:.0f} ms).")
                return
            except Exception as e:
                print(f"[GGUFContext] Partial hot-reload failed: {e}, falling back to full lowvram...")

        # ── FULL LOWVRAM: fast path ──────────────────────────
        print(f"[GGUFContext] Hot-loading to {device} (fast path)...")
        try:
            self._fast_reload_gguf(model, inner, device)
        except Exception as e:
            print(f"[GGUFContext] Fast reload failed: {e}, falling back to model.load()...")
            if hasattr(model, "load"):
                model.load(device, force_patch_weights=True)

        model.current_device = device
        dt = (time.perf_counter() - t0) * 1000
        print(f"[GGUFContext] Hot-load complete ({dt:.0f} ms).")

    # ─── Fast reload (delegates to _fast_reload_common) ──────

    @staticmethod
    def _fast_reload_gguf(model: Any, inner: Any, device: torch.device) -> None:
        """GGUF-specific fast path: re-enable lowvram hooks + LoRA patches."""

        def _apply_gguf_patches(
            *,
            model: Any,
            inner: Any,
            device: torch.device,
            patches: Dict,
            wrapper_patches: Dict,
            force_cast: bool,
        ) -> None:
            # (1) Re-enable lowvram hooks on every castable module
            for n, m in inner.named_modules():
                if not hasattr(m, "comfy_cast_weights"):
                    continue
                m.weight_function = []
                m.bias_function = []
                m.comfy_force_cast_weights = force_cast
                if not hasattr(m, "prev_comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
                m.comfy_patched_weights = True
                wk = f"{n}.weight"
                bk = f"{n}.bias"
                if wk in wrapper_patches:
                    m.weight_function.extend(wrapper_patches[wk])
                if bk in wrapper_patches:
                    m.bias_function.extend(wrapper_patches[bk])

            # (2) Re-attach LoRA .patches to quantized GGMLTensors
            if patches:
                from raylight.expansion.comfyui_gguf.ops import move_patch_to_device
                from raylight.expansion.comfyui_gguf.dequant import is_quantized
                mmap_cache = getattr(model, "mmap_cache", {})
                patch_dev = (
                    model.load_device
                    if getattr(model, "patch_on_device", False)
                    else model.offload_device
                )
                for key, patch_list in patches.items():
                    w = mmap_cache.get(key)
                    if w is not None and is_quantized(w):
                        w.patches = [(move_patch_to_device(patch_list, patch_dev), key)]

        ModelContext._fast_reload_common(
            model, inner, device,
            apply_patches_fn=_apply_gguf_patches,
            lowvram=True,
        )
