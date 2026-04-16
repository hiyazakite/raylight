"""TP model context — streaming shard loading with pinned-cache offload/reload.

Follows the same ModelContext lifecycle as FSDPContext / LazyTensorContext but
owns the full TP pipeline: load → TP-patch → stream weights → offload → hot reload.

For .safetensors checkpoints the loader uses a streaming approach inspired by
sglang: weights are read one-at-a-time via ``safetensors.safe_open()`` and fed
directly into each ``TPLinear.weight_loader()``.  The full checkpoint is never
materialised in RAM, reducing peak memory from ``full_model + shards`` to
``single_weight + shards``.

Non-safetensors (.pt / .ckpt) files fall back to the legacy full-load path.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from ._base import ModelContext, ModelState, _compute_vram_budget, _ops_for_model_options

if TYPE_CHECKING:
    from raylight.raylight_types import LoraManagerLike, ActorConfigLike


def _drop_page_cache(path: str) -> None:
    """Advise the kernel to drop page cache for *path* (Linux only).

    After a streaming model load, the safetensors pages are no longer
    needed — weights are on CUDA.  Dropping the page cache prevents Ray's
    memory monitor from counting those pages as "used" RAM, which can
    push past the OOM threshold during VAE decode.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"[TPContext] Dropped page cache for checkpoint ({size_gb:.1f} GB)")
    except (OSError, AttributeError):
        pass  # Not Linux, or file removed


class TPContext(ModelContext):
    """Context for Tensor Parallel model loading with streaming weight transfer."""

    def __init__(self, use_mmap: bool = True, zero_ram: bool = False):
        super().__init__(use_mmap)
        self._zero_ram = zero_ram

    # ─── Activate (initial load → CUDA) ──────────────────────

    def activate(self, model: Any, device: torch.device,
                 memory: Any = None) -> None:
        """Move a freshly-loaded TP model onto *device*.

        Both zero_ram and deferred-pinned modes stream directly to CUDA
        during ``_load_streaming``, so activate only needs to stamp
        metadata and apply ComfyUI patch hooks.

        Legacy loads (non-safetensors fallback) still go through the
        pinned-cache → cold-start path.
        """
        if model is None:
            return

        current = getattr(model, "current_device", None)

        if current is not None and str(current) == str(device):
            # Weights were streamed directly to CUDA, OR the interceptor
            # may have pressure-evicted some/all weights without updating
            # current_device.  Restore any evicted params first.
            pinned_cache = getattr(model, "pinned_param_cache", None)
            if pinned_cache is not None and pinned_cache._partial_freed:
                diffusion_model = getattr(model, "model", None)
                if diffusion_model is not None:
                    print("[TPContext] activate: restoring pressure-evicted weights...")
                    pinned_cache.reload_evicted(diffusion_model)

            # Apply ComfyUI patch metadata then return.
            diffusion_model = getattr(model, "model", None)
            if diffusion_model is not None:
                diffusion_model.device = device
                self._fast_reload_tp(model, diffusion_model, device)
            return

        # Legacy / pinned-cache path: weights on CPU, move to CUDA.
        pinned_cache = getattr(model, "pinned_param_cache", None)
        diffusion_model = getattr(model, "model", None)

        if pinned_cache is not None and not pinned_cache.built and diffusion_model is not None:
            print("[TPContext] activate: building pinned cache from CPU params...")
            pinned_cache.build(diffusion_model)

        if pinned_cache is not None and pinned_cache.built and diffusion_model is not None:
            print("[TPContext] activate: pinned cache → CUDA...")
            pinned_cache.reload_to_cuda(diffusion_model)
            if hasattr(pinned_cache, 'release_host_copy'):
                pinned_cache.release_host_copy()
            diffusion_model.device = device
            self._fast_reload_tp(model, diffusion_model, device)
            model.current_device = device
            return

        # Fallback: model.load() for legacy / non-pinned path
        if hasattr(model, "load"):
            model.load(device)
        model.current_device = device

    # ─── Disk loading (used by legacy fallback) ──────────────

    def load_state_dict_mmap(self, state: ModelState, config: "ActorConfigLike") -> Any:
        import safetensors.torch
        import safetensors
        sd = safetensors.torch.load_file(state.unet_path, device="cpu")
        try:
            with safetensors.safe_open(state.unet_path, framework="pt") as f:
                metadata = f.metadata() or {}
        except Exception:
            metadata = {}
        return sd, metadata

    def load_state_dict_standard(self, state: ModelState, config: "ActorConfigLike") -> Any:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)

    # ─── Pinned cache factory ────────────────────────────────

    @staticmethod
    def _make_pinned_cache(config: "ActorConfigLike", model_path: str):
        """TP ranks hold different shards — always use private contiguous cache."""
        from raylight.distributed_modules.pinned_cache import ContiguousPinnedCache
        return ContiguousPinnedCache()

    # ─── Streaming load (main path) ──────────────────────────

    def load(self, state: ModelState, config: "ActorConfigLike", state_cache: Any) -> Any:
        """Load model with streaming TP weight transfer for .safetensors.

        Pipeline:
          1. Read safetensors header → meta state dict (zero RAM)
          2. Instantiate model architecture with meta weights
          3. TP-patch structure only (TPLinear with empty shards)
          4. Stream weights from disk → weight_loader → discard
          5. Attach pinned cache (or skip in zero_ram mode)
        """
        is_safetensors = state.unet_path.lower().endswith(".safetensors")

        if not is_safetensors and self._zero_ram:
            raise RuntimeError(
                "[TPContext] zero_ram mode requires .safetensors checkpoints. "
                f"Got: {state.unet_path}"
            )

        if is_safetensors:
            if self._zero_ram:
                # zero_ram: do NOT catch exceptions — no legacy fallback.
                return self._load_streaming(state, config)
            try:
                return self._load_streaming(state, config)
            except Exception as e:
                print(f"[TPContext] Streaming load failed: {e}. Falling back to legacy path...")

        return self._load_legacy(state, config, state_cache)

    def _load_streaming(self, state: ModelState, config: "ActorConfigLike") -> Any:
        """Streaming TP load — peak RAM ≈ single weight tensor + shards.

        Both zero_ram and standard modes stream directly to CUDA.  An
        unbuilt ``ContiguousPinnedCache`` is attached for standard mode
        so the alloc interceptor can do incremental per-layer eviction
        under VRAM pressure; zero_ram gets no cache at all.
        """
        from safetensors import safe_open
        from raylight.comfy_dist.tp_registry import TPRegistry
        from raylight.distributed_modules.tensor_parallel import TensorParallelState
        from raylight.distributed_modules.tp_compress import TPCompressConfig
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

        t0 = time.perf_counter()
        path = state.unet_path
        basename = os.path.basename(path)
        zero_ram = self._zero_ram

        mode_label = "zero-RAM" if zero_ram else "streaming"

        # ── Phase 1: Header-only meta state dict (zero RAM) ──
        print(f"[TPContext] {mode_label} TP load: {basename}")
        meta_sd: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, Any] = {}

        _SF_DTYPE_MAP = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
            "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
            "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
            "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
        }

        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
            for k in f.keys():
                t_slice = f.get_slice(k)
                shape = t_slice.get_shape()
                sf_dtype = t_slice.get_dtype()
                dtype = _SF_DTYPE_MAP.get(sf_dtype, torch.float16)
                meta_sd[k] = torch.empty(shape, dtype=dtype, device="meta")

        # ── Phase 2: Instantiate model with meta weights ──
        print("[TPContext] Phase 2: Instantiating model architecture (meta)...")
        model = self._instantiate_model_meta(meta_sd, state, config, metadata)
        del meta_sd

        # ── Phase 3: TP patching (structure only, no weight copies) ──
        print("[TPContext] Phase 3: Applying TP structure...")
        _strategy = config.raylight_config.strategy
        _compress_config = TPCompressConfig(
            mode=_strategy.tp_allreduce_compress,
            bits=_strategy.tp_compress_bits,
            group_size=_strategy.tp_compress_group_size,
            use_residual=_strategy.tp_compress_residual,
            rotation=_strategy.tp_compress_rotation,
            residual_bits=_strategy.tp_compress_residual_bits,
            residual_skip_threshold=_strategy.tp_compress_skip_threshold,
        )
        TPRegistry.apply(
            model,
            tp_group=TensorParallelState.get_group(),
            compress_config=_compress_config,
            structure_only=True,
        )

        # ── Phase 4: Prepare target storage ──
        #
        # Streaming load never creates a full state dict — weights flow
        # one-at-a-time from safe_open → model params → discard.  There is
        # no mmap_cache to retain (unlike the legacy path).
        model.mmap_cache = None

        if zero_ram:
            # zero_ram: no cache at all — offload frees CUDA to meta,
            # reload re-streams from disk.
            model.pinned_param_cache = None
        else:
            # Create cache object (unbuilt) so it can be registered with
            # MemoryPolicy + alloc interceptor.  No RAM allocated until
            # first pressure eviction or full offload.  Weights stream
            # directly to CUDA — inference runs with zero pinned overhead.
            model.pinned_param_cache = self._make_pinned_cache(config, path)
        target_device = config.device

        # ── Phase 5: Stream weights from disk into TP shards ──
        target_label = str(target_device) if target_device else "pinned"
        print(f"[TPContext] Phase 5: Streaming weights into TP shards (target={target_label})...")
        loaded_count = self._stream_weights_into_model(model, path, target_device=target_device)

        # ── Phase 5b: Drop file page cache ──
        # After streaming, all weights live on CUDA.  The safetensors file
        # pages (~22 GB) linger in the OS page cache which Ray counts as
        # "used" via `(total - available) / total`.  Advise the kernel to
        # drop them so VAE decode doesn't push past the OOM threshold.
        _drop_page_cache(path)

        model.unet_path = state.unet_path
        model.load_config = state
        model._zero_ram = zero_ram
        model.current_device = config.device  # already on CUDA after streaming

        dt = (time.perf_counter() - t0) * 1000
        print(
            f"[TPContext] {mode_label} TP load complete: {loaded_count} params "
            f"in {dt:.0f} ms"
        )
        return model

    def _instantiate_model_meta(
        self, meta_sd: Dict[str, torch.Tensor], state: ModelState,
        config: "ActorConfigLike", metadata: Any,
    ) -> Any:
        """Build ComfyUI model from meta state dict (zero-weight instantiation).

        Weight loading uses ``assign=True`` so that meta tensors are
        pointer-swapped into modules (no copy from meta).  This gives every
        module a proper weight parameter with the correct shape/dtype that
        Phase 3 TP patching can inspect.  Phase 4 then streams real weights
        from disk, replacing the meta placeholders.
        """
        import comfy.sd
        import comfy.model_base
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

        lazy_sd = wrap_state_dict_lazy(meta_sd)

        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        load_options["custom_operations"] = _ops_for_model_options(load_options)

        # Force assign=True so meta tensors are pointer-swapped (not copied).
        # This avoids "Cannot copy out of meta tensor" for RMSNorm etc.
        _orig_load = comfy.model_base.BaseModel.load_model_weights
        def _assign_load_weights(self_model, sd, unet_prefix="", assign=False):
            return _orig_load(self_model, sd, unet_prefix, assign=True)

        comfy.model_base.BaseModel.load_model_weights = _assign_load_weights
        try:
            model = comfy.sd.load_diffusion_model_state_dict(
                lazy_sd, model_options=load_options, metadata=metadata,
            )
        finally:
            comfy.model_base.BaseModel.load_model_weights = _orig_load

        if model is None:
            raise RuntimeError(f"[TPContext] Could not detect model from: {state.unet_path}")

        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype

        model.__class__ = RaylightModelPatcher
        return model

    @staticmethod
    def _remap_key(model_key: str, model_config: Any) -> str:
        """Apply model-specific key remapping (e.g. Flux ``_norm.scale`` → ``_norm.weight``)."""
        if model_config is not None and hasattr(model_config, "process_unet_state_dict"):
            remapped = model_config.process_unet_state_dict({model_key: None})
            if remapped:
                return next(iter(remapped))
        return model_key

    def _stream_weights_into_model(
        self, model: Any, path: str, target_device: Optional[torch.device] = None,
    ) -> int:
        """Stream weights from safetensors file into the TP-patched model.

        Each weight is read individually via safe_open, dispatched to the
        corresponding TPLinear/TPRMSNorm weight_loader (which narrows + copies
        the shard), and immediately discarded.

        Args:
            target_device: When set (e.g. zero_ram mode), final parameters are
                placed on this device.  ``None`` keeps them on CPU (default).

        Non-TP parameters (norms, embeddings, etc.) are copied or assigned
        directly.  Handles:
        - Registered ``nn.Parameter`` → ``data.copy_()``
        - ``weight = None`` (SafetensorOps) → assign new ``nn.Parameter``
        - Meta-device params → allocate real storage first, then copy
        """
        from safetensors import safe_open
        from raylight.distributed_modules.tensor_parallel import TPLinear, TPRMSNormAcrossHeads
        from comfy import model_detection

        diffusion_model = getattr(model, "model", None)
        if diffusion_model is None:
            raise RuntimeError("[TPContext] Model has no .model attribute")

        inner = getattr(diffusion_model, "diffusion_model", None)
        if inner is None:
            raise RuntimeError("[TPContext] Model has no .model.diffusion_model attribute")

        # Model config for key remapping (e.g. Flux _norm.scale → _norm.weight)
        model_config = getattr(diffusion_model, "model_config", None)

        # Build module path → module mapping for dispatch
        module_map: Dict[str, torch.nn.Module] = dict(inner.named_modules())

        # Build parameter name → parameter mapping for registered params
        param_map: Dict[str, torch.nn.Parameter] = dict(inner.named_parameters())

        # Identify TP modules for weight_loader dispatch
        tp_modules: Dict[str, Any] = {}
        for name, module in module_map.items():
            if isinstance(module, (TPLinear, TPRMSNormAcrossHeads)):
                tp_modules[name] = module

        # Determine checkpoint key prefix
        with safe_open(path, framework="pt") as f:
            all_keys = list(f.keys())

        prefix = ""
        if all_keys:
            sample_sd = {k: torch.empty(0) for k in all_keys}
            detected_prefix = model_detection.unet_prefix_from_state_dict(sample_sd)
            if detected_prefix:
                prefix = detected_prefix
            del sample_sd

        # Pre-compute which checkpoint keys map to valid model params so we
        # can skip loading tensors that won't be used (1500+ keys for LTXAV).
        valid_keys = set()
        for ckpt_key in all_keys:
            if prefix and ckpt_key.startswith(prefix):
                model_key = ckpt_key[len(prefix):]
            else:
                model_key = ckpt_key
            model_key = self._remap_key(model_key, model_config)
            parts = model_key.rsplit(".", 1)
            mod_path = parts[0] if len(parts) == 2 else ""
            p_attr = parts[-1]
            if mod_path in tp_modules:
                valid_keys.add(ckpt_key)
            elif mod_path in module_map:
                valid_keys.add(ckpt_key)
            elif model_key in param_map:
                valid_keys.add(ckpt_key)

        loaded_count = 0
        skipped_count = 0

        # Always load tensors to CPU first, then surgically place shards on
        # target_device.  For TP modules the full unsharded tensor is narrowed
        # on CPU so only the 1/tp_size shard is copied to CUDA — peak VRAM
        # equals exactly the final model size with zero transient overhead.
        with safe_open(path, framework="pt") as f:
            for ckpt_key in f.keys():
                # Skip keys that don't map to any model parameter
                if ckpt_key not in valid_keys:
                    skipped_count += 1
                    continue

                # Strip prefix to get model-relative parameter name
                if prefix and ckpt_key.startswith(prefix):
                    model_key = ckpt_key[len(prefix):]
                else:
                    model_key = ckpt_key

                # Apply model-specific key remapping
                model_key = self._remap_key(model_key, model_config)

                # Load single tensor from disk to CPU
                tensor = f.get_tensor(ckpt_key)

                # Determine the module path and parameter attr
                parts = model_key.rsplit(".", 1)
                if len(parts) == 2:
                    module_path, param_attr = parts
                else:
                    module_path, param_attr = "", parts[0]

                # ── TP module dispatch ──
                tp_module = tp_modules.get(module_path)
                if tp_module is not None and hasattr(tp_module, "weight_loader"):
                    param = getattr(tp_module, param_attr, None)
                    if param is not None:
                        # Reallocate param if on wrong device (meta from
                        # structure_only, or CPU when target is CUDA).
                        desired = target_device if target_device is not None else tensor.device
                        if param.device != desired:
                            new_p = torch.nn.Parameter(
                                torch.empty_like(param, device=desired),
                                requires_grad=False,
                            )
                            setattr(tp_module, param_attr, new_p)
                            param = new_p
                        if (
                            param_attr == "bias"
                            and isinstance(tp_module, TPLinear)
                            and tp_module.parallelism == "row"
                        ):
                            param.data.copy_(tensor)
                        else:
                            tp_module.weight_loader(param, tensor)
                        loaded_count += 1
                        del tensor
                        continue

                    # Handle INT8 scale attributes (not nn.Parameters but
                    # needed by TPLinear.forward for W8A8 inference).
                    if param_attr in ("weight_scale", "input_scale") and isinstance(tp_module, TPLinear):
                        if param_attr == "input_scale":
                            # input_scale not used by TPLinear — skip silently
                            loaded_count += 1
                            del tensor
                            continue
                        # weight_scale: shard per-row scales for column-parallel
                        scale = tensor.float()
                        if scale.dim() >= 1 and scale.numel() > 1:
                            # Per-row scale — shard along output dim for column-parallel
                            if tp_module.parallelism == "column":
                                # Use QKV-aware scale loader if available (fused QKV layers)
                                if hasattr(tp_module, "_qkv_scale_loader"):
                                    scale = tp_module._qkv_scale_loader(scale)
                                else:
                                    tp_rank = TensorParallelState.get_rank()
                                    shard_size = scale.shape[0] // TensorParallelState.get_size()
                                    scale = scale.narrow(0, tp_rank * shard_size, shard_size).contiguous()
                            # Row-parallel: output dim is not split — keep full
                        else:
                            # Scalar per-tensor scale
                            scale = scale.item()
                        if target_device is not None and isinstance(scale, torch.Tensor):
                            scale = scale.to(target_device)
                        setattr(tp_module, param_attr, scale)
                        loaded_count += 1
                        del tensor
                        continue

                # ── Non-TP parameter dispatch ──
                target_module = module_map.get(module_path)
                if target_module is not None:
                    existing = getattr(target_module, param_attr, None)
                    if existing is None:
                        # SafetensorOps sets weight/bias = None; assign directly
                        t = tensor.to(target_device) if target_device is not None else tensor
                        setattr(
                            target_module, param_attr,
                            torch.nn.Parameter(t, requires_grad=False),
                        )
                        loaded_count += 1
                        del tensor
                        continue
                    elif isinstance(existing, torch.nn.Parameter):
                        if existing.device == torch.device("meta"):
                            # Meta param — replace with real storage
                            dev = target_device if target_device is not None else None
                            new_p = torch.nn.Parameter(
                                tensor.to(dtype=existing.dtype, device=dev),
                                requires_grad=False,
                            )
                            setattr(target_module, param_attr, new_p)
                        else:
                            if target_device is not None and existing.device != target_device:
                                new_p = torch.nn.Parameter(
                                    tensor.to(dtype=existing.dtype, device=target_device),
                                    requires_grad=False,
                                )
                                setattr(target_module, param_attr, new_p)
                            else:
                                existing.data.copy_(tensor)
                        loaded_count += 1
                        del tensor
                        continue

                # Fallback: registered param by full dotted name
                if model_key in param_map:
                    param_map[model_key].data.copy_(tensor)
                    loaded_count += 1
                else:
                    skipped_count += 1

                del tensor

        if skipped_count > 0:
            print(f"[TPContext] Skipped {skipped_count} checkpoint keys (not in model)")

        return loaded_count

    # ─── Legacy fallback (non-safetensors or streaming failure) ─

    def _load_legacy(self, state: ModelState, config: "ActorConfigLike", state_cache: Any) -> Any:
        """Full-load path: loads entire checkpoint then TP-patches.

        Used for .pt/.ckpt files and as a fallback when streaming fails.
        """
        from raylight.comfy_dist.tp_registry import TPRegistry
        from raylight.distributed_modules.tensor_parallel import TensorParallelState
        from raylight.distributed_modules.tp_compress import TPCompressConfig

        print(f"[TPContext] Legacy load: {os.path.basename(state.unet_path)}")

        # Use parent's full load pipeline
        model = super().load(state, config, state_cache)

        # Apply TP patching with weight copies (current behavior)
        if model is not None and TPRegistry.has_handler(model):
            _strategy = config.raylight_config.strategy
            _compress_config = TPCompressConfig(
                mode=_strategy.tp_allreduce_compress,
                bits=_strategy.tp_compress_bits,
                group_size=_strategy.tp_compress_group_size,
                use_residual=_strategy.tp_compress_residual,
                rotation=_strategy.tp_compress_rotation,
                residual_bits=_strategy.tp_compress_residual_bits,
                residual_skip_threshold=_strategy.tp_compress_skip_threshold,
            )
            TPRegistry.apply(
                model,
                tp_group=TensorParallelState.get_group(),
                compress_config=_compress_config,
            )

        # Stamp lifecycle attributes (same as _load_streaming Phase 5)
        if model is not None:
            # The base class load() retains the full state dict as mmap_cache.
            # For TP safetensors, the pinned cache reads from live params (not
            # from the source dict), so drop mmap_cache immediately to free
            # ~1× model worth of RAM.
            model.mmap_cache = None
            model.unet_path = state.unet_path
            model.load_config = state
            model._zero_ram = False
            model.current_device = torch.device("cpu")

        return model

    # ─── Instantiation (for legacy path) ─────────────────────

    def instantiate_model(self, sd: Dict, state: ModelState, config: "ActorConfigLike",
                          metadata: Any = None, **kwargs) -> Any:
        """Instantiate model with lazy state dict (used by legacy path)."""
        import comfy.sd
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

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

        # The state dict has been consumed by load_diffusion_model_state_dict;
        # weights now live in model parameters.  Do NOT retain the full dict
        # as mmap_cache — the pinned cache reads from live params, not from
        # the source dict.  Dropping it immediately saves ~1× model in RAM
        # (sglang pattern: stream → consume → release source).
        model.mmap_cache = None
        model.__class__ = RaylightModelPatcher
        # Build pinned cache eagerly — params are CPU-resident right now,
        # so build() does pin_memory + pointer-swap with zero transient
        # overhead (each param swapped in-place, old freed by refcount).
        cache = self._make_pinned_cache(config, state.unet_path)
        diffusion_model = getattr(model, "model", None)
        if diffusion_model is not None:
            cache.build(diffusion_model)
        model.pinned_param_cache = cache
        return model

    # ─── Offload (Template Method hook) ──────────────────────

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "ActorConfigLike",
    ) -> bool:
        """Offload TP model: pinned-RAM or zero_ram (free CUDA storages)."""
        if model is None:
            return False

        zero_ram = getattr(model, "_zero_ram", False)

        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)

        if hasattr(model, "unpatch_model"):
            model.unpatch_model(device_to=None, unpatch_weights=True)

        if zero_ram:
            # zero_ram: free all CUDA parameter storages to reclaim VRAM.
            # Reload will re-stream from disk.
            print(f"[TPContext {config.local_rank}] zero-RAM Offload: freeing CUDA storages...")
            diffusion_model = getattr(model, "model", None)
            if diffusion_model is not None:
                inner = getattr(diffusion_model, "diffusion_model", None)
                if inner is not None:
                    for p in inner.parameters():
                        if p.device.type == "cuda":
                            # Preserve shape so reload can reallocate correctly.
                            p.data = torch.empty(p.data.shape, dtype=p.dtype, device="meta")
            model.current_device = torch.device("cpu")
            if lora_manager:
                lora_manager.clear_tracking()
            return False

        diffusion_model = getattr(model, "model", None)
        pinned_cache = getattr(model, "pinned_param_cache", None)

        if pinned_cache is not None and diffusion_model is not None:
            # Restore any pressure-evicted params whose CUDA storages were
            # freed.  build() (called by offload_to_cpu when !built) needs
            # live CUDA data to construct the contiguous slab.
            if pinned_cache._partial_freed:
                pinned_cache.reload_evicted(diffusion_model)

            print(f"[TPContext {config.local_rank}] Pinned-RAM Offload: CUDA → pinned CPU...")

            try:
                pinned_cache.offload_to_cpu(diffusion_model)
            except Exception as e:
                print(f"[TPContext {config.local_rank}] Pinned offload error: {e}")

            model.current_device = torch.device("cpu")

            if lora_manager:
                lora_manager.clear_tracking()

            return False

        # No pinned cache — hard offload
        print(f"[TPContext {config.local_rank}] Hard-Offload: no pinned cache.")
        return True

    # ─── Hot load ────────────────────────────────────────────

    @staticmethod
    def _has_meta_params(model: Any) -> bool:
        """Check if the inner diffusion model has meta-device parameters.

        Meta tensors are the hallmark of a zero-RAM offload: real CUDA
        storages were freed and replaced with shape-preserving meta
        placeholders.  Detecting this lets ``hot_load`` recover even when
        the ``_zero_ram`` flag was lost (e.g. through a clone).
        """
        inner = getattr(getattr(model, "model", None), "diffusion_model", None)
        if inner is None:
            return False
        for p in inner.parameters():
            if p.device.type == "meta":
                return True
        return False

    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any]) -> None:
        """Restore TP model from pinned RAM (or re-stream in zero_ram mode)."""
        zero_ram = getattr(model, "_zero_ram", False)

        # Fallback detection: if the flag was lost (e.g. via clone) but
        # parameters are on ``meta`` device, treat as zero_ram.
        if not zero_ram and self._has_meta_params(model):
            unet_path = getattr(model, "unet_path", None)
            if unet_path and unet_path.lower().endswith(".safetensors"):
                print("[TPContext] Detected meta params without _zero_ram flag — treating as zero-RAM.")
                model._zero_ram = True  # restore for future cycles
                zero_ram = True

        if zero_ram:
            return self._hot_load_zero_ram(model, device)

        pinned_cache = getattr(model, "pinned_param_cache", None)
        diffusion_model = getattr(model, "model", None)

        # Cache is now built eagerly during load() — no deferred build
        # needed here.  Guard retained for safety (e.g. external callers).
        if (
            pinned_cache is not None
            and not pinned_cache.built
            and diffusion_model is not None
        ):
            print("[TPContext] Late cache build (should not happen — check load path)...")
            try:
                pinned_cache.build(diffusion_model)
            except Exception as e:
                print(f"[TPContext] Late cache build failed: {e}")

        # Pinned-cache fast path
        if pinned_cache is not None and pinned_cache.built and diffusion_model is not None:
            vram_limit = getattr(model, "vram_limit_bytes", 0)
            model_bytes = model.model_size() if hasattr(model, "model_size") else 0
            budget = (
                _compute_vram_budget(device, model_bytes, vram_limit_bytes=vram_limit)
                if model_bytes > 0
                else 0
            )

            if budget == 0:
                t0 = time.perf_counter()
                print("[TPContext] Pinned-RAM Hot-Reload (full): pinned CPU → CUDA...")
                try:
                    pinned_cache.reload_to_cuda(diffusion_model)
                    if hasattr(pinned_cache, 'release_host_copy'):
                        pinned_cache.release_host_copy()
                    diffusion_model.device = device
                    self._fast_reload_tp(model, diffusion_model, device)
                    model.current_device = device
                    dt = (time.perf_counter() - t0) * 1000
                    print(f"[TPContext] Hot-Reload complete ({dt:.0f} ms).")
                    return
                except Exception as e:
                    print(f"[TPContext] Hot-Reload failed: {e}, falling back...")

        # Fallback: full model.load()
        print("[TPContext] Hot-Reload: falling back to model.load()...")
        if model is not None and hasattr(model, "load"):
            model.load(device)
            model.current_device = device

    def _hot_load_zero_ram(self, model: Any, device: torch.device) -> None:
        """Re-stream weights from disk directly to CUDA (zero_ram reload)."""
        unet_path = getattr(model, "unet_path", None)
        if not unet_path or not unet_path.lower().endswith(".safetensors"):
            print("[TPContext] zero-RAM reload: no safetensors path, falling back...")
            if hasattr(model, "load"):
                model.load(device)
                model.current_device = device
            return

        t0 = time.perf_counter()
        print(f"[TPContext] zero-RAM Hot-Reload: re-streaming from {os.path.basename(unet_path)}...")

        loaded_count = self._stream_weights_into_model(model, unet_path, target_device=device)

        diffusion_model = getattr(model, "model", None)
        if diffusion_model is not None:
            diffusion_model.device = device
            inner = getattr(diffusion_model, "diffusion_model", None)
            if inner is not None:
                self._fast_reload_tp(model, inner, device)

        model.current_device = device
        dt = (time.perf_counter() - t0) * 1000
        print(f"[TPContext] zero-RAM Hot-Reload complete: {loaded_count} params in {dt:.0f} ms")

    @staticmethod
    def _fast_reload_tp(model: Any, inner: Any, device: torch.device) -> None:
        """Bypass model.load() after reload_to_cuda — params already on CUDA."""

        def _apply_patches(
            *,
            model: Any,
            inner: Any,
            device: torch.device,
            patches: Dict,
            wrapper_patches: Dict,
            force_cast: bool,
        ) -> None:
            if patches:
                for key in patches:
                    model.patch_weight_to_device(key, device_to=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

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
            apply_patches_fn=_apply_patches,
            lowvram=False,
        )
