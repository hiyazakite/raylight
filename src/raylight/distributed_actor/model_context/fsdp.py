"""FSDP model context — sharded training/inference with pinned shard caching."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, Optional
import os

import torch

from ._base import ModelContext, ModelState, _ops_for_model_options
from raylight.distributed_modules.fsdp_utils import prefetch_state_dict

if TYPE_CHECKING:
    from raylight.raylight_types import LoraManagerLike, ActorConfigLike


class FSDPContext(ModelContext):
    """Context for FSDP model loading with optional mmap for safetensors."""

    # ─── Disk loading ────────────────────────────────────────

    def load_state_dict_mmap(self, state: ModelState, config: "ActorConfigLike") -> Any:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)

    def load_state_dict_standard(self, state: ModelState, config: "ActorConfigLike") -> Any:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)

    # ─── Load (with prefetch optimisation) ───────────────────

    def load(self, state: ModelState, config: "ActorConfigLike", state_cache: Any) -> Any:
        """Override load to implement prefetch optimization for Safetensors."""
        is_safetensors = state.unet_path.lower().endswith(".safetensors")

        if is_safetensors and not self.use_mmap:
            print(f"[FSDPContext] Prefetch Optimization Active for {os.path.basename(state.unet_path)}")

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(prefetch_state_dict, state.unet_path)

            from safetensors import safe_open
            dummy_sd: Dict[str, torch.Tensor] = {}
            metadata = None
            try:
                with safe_open(state.unet_path, framework="pt") as f: # type: ignore
                    metadata = f.metadata()
                    for k in f.keys():
                        t_slice = f.get_slice(k)
                        shape = t_slice.get_shape()
                        dummy_sd[k] = torch.empty(shape, dtype=torch.float16, device="meta")
            except Exception as e:
                print(f"[FSDPContext] Header read failed: {e}. Fallback to sync load.")
                executor.shutdown(wait=False)
                return super().load(state, config, state_cache)

            try:
                model = self.instantiate_model(
                    dummy_sd, state, config, metadata=metadata, load_weights=False,
                )

                print("[FSDPContext] FSDP Wrapping done. Waiting for weights...")
                real_sd = future.result()

                print("[FSDPContext] Weights loaded. Loading into FSDP model...")

                from comfy import model_detection
                import comfy.utils
                diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(real_sd)
                temp_sd = comfy.utils.state_dict_prefix_replace(
                    real_sd, {diffusion_model_prefix: ""}, filter_keys=True,
                )
                if len(temp_sd) > 0:
                    real_sd = temp_sd

                model.load_model_weights(real_sd, "")
                executor.shutdown(wait=False)

                if model is not None:
                    model.unet_path = state.unet_path
                    model.load_config = state

                return model

            except Exception as e:
                print(f"[FSDPContext] Prefetch/Wrap failed: {e}. Fallback to sync load.")
                executor.shutdown(wait=False)
                return super().load(state, config, state_cache)

        return super().load(state, config, state_cache)

    # ─── Instantiation ───────────────────────────────────────

    def instantiate_model(self, sd: Dict, state: ModelState, config: "ActorConfigLike",
                          metadata: Any = None, load_weights: bool = True, **kwargs) -> Any:
        from raylight.comfy_dist.sd import fsdp_load_diffusion_model_stat_dict
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy

        import comfy.model_patcher as model_patcher
        import comfy.model_management as model_management
        from raylight.comfy_dist.model_management import cleanup_models_gc
        from raylight.comfy_dist.model_patcher import LowVramPatch
        model_patcher.LowVramPatch = LowVramPatch
        model_management.cleanup_models_gc = cleanup_models_gc

        if self.use_mmap:
            print("[FSDPContext] Wrapping state dict with LazySafetensors for JIT loading...")
            sd = wrap_state_dict_lazy(sd)

        model_options = state.model_options.copy()
        model_options["custom_operations"] = _ops_for_model_options(model_options)

        model, state_dict = fsdp_load_diffusion_model_stat_dict(
            sd,
            config.local_rank,
            getattr(config, "device_mesh", None),
            config.fsdp_cpu_offload,
            model_options=model_options,
            metadata=metadata,
            load_weights=load_weights,
        )

        if model is None:
            raise RuntimeError(f"Could not load FSDP model: {state.unet_path}")

        model.is_fsdp_baked = True

        fsdp_cpu_offload = config.fsdp_cpu_offload
        if not fsdp_cpu_offload:
            from raylight.distributed_modules.pinned_cache import FSDPShardPinnedCache
            model.fsdp_shard_cache = FSDPShardPinnedCache()
            print(f"[FSDPContext {config.local_rank}] Shard cache attached (will build on first offload).")

        return model

    # ─── Offload (Template Method hook) ──────────────────────

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "ActorConfigLike",
    ) -> bool:
        """FSDP soft offload: shards → pinned CPU RAM, or hard offload if no cache."""
        shard_cache = getattr(model, "fsdp_shard_cache", None) if model is not None else None

        if shard_cache is not None:
            print(f"[FSDPContext {config.local_rank}] FSDP Soft-Offload: shards → pinned CPU RAM...")

            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)

            try:
                shard_cache.offload_to_cpu(model.model.diffusion_model)
            except Exception as e:
                print(f"[FSDPContext {config.local_rank}] Shard cache offload error: {e}")

            if model.model is not None:
                model.model.device = torch.device("cpu")

            if lora_manager:
                lora_manager.clear_tracking()

            # Heavy cleanup (gc, cache flush, VRAM logging) delegated to
            # the MemoryPolicy in the template-method's after_offload().
            return False

        else:
            print(
                f"[FSDPContext {config.local_rank}] FSDP Hard-Offload: "
                f"no shard cache, releasing all resources..."
            )

            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)

            if model is not None and hasattr(model, "release_memory"):
                model.release_memory()

            return True

    # ─── Hot load ────────────────────────────────────────────

    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any]) -> None:
        """Restore FSDP local shards from pinned CPU RAM back to CUDA."""
        shard_cache = getattr(model, "fsdp_shard_cache", None)
        if shard_cache is not None and shard_cache.built:
            print("[FSDPContext] FSDP Hot-Reload: pinned CPU RAM → CUDA...")
            try:
                shard_cache.reload_to_cuda(model.model.diffusion_model)
                self._activate_fp8_buffers(model, device)
                if model.model is not None:
                    model.model.device = device
                model.current_device = device
                print("[FSDPContext] Hot-Reload complete.")
                return
            except Exception as e:
                print(f"[FSDPContext] Hot-Reload failed: {e}, falling back to full reload...")

        print("[FSDPContext] Hot-Reload: no shard cache or pinned reload failed — full reload.")
        if model is not None and hasattr(model, "load"):
            model.load(device)
            self._activate_fp8_buffers(model, device)
            model.current_device = device
