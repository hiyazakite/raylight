"""VAE model context — streaming mmap + zero-copy offload."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from .lazy_tensor import LazyTensorContext
from ._base import ModelState

if TYPE_CHECKING:
    from raylight.types import LoraManagerLike, ActorConfigLike


class VAEContext(LazyTensorContext):
    """Context for VAE model loading with streaming mmap support."""

    def prepare_state_dict(
        self, sd: Dict[str, torch.Tensor], config: "ActorConfigLike"
    ) -> Dict[str, torch.Tensor]:
        """VAE: No prefix stripping needed."""
        return sd

    # ─── Instantiation ───────────────────────────────────────

    def instantiate_model(self, sd: Dict, state: ModelState, config: "ActorConfigLike",
                          metadata: Any = None, **kwargs) -> Any:
        """Instantiate VAE model using comfy.sd.VAE."""
        import comfy.sd
        import safetensors

        print(f"[VAEContext] Standard VAE init with {len(sd)} keys")

        metadata = None
        try:
            with safetensors.safe_open(state.unet_path, framework="pt") as f: # type: ignore
                metadata = f.metadata()
            if metadata:
                print(f"[VAEContext] Loaded metadata: contains 'config': {'config' in metadata}")
        except Exception:
            pass

        # DEBUG: Check VAE version detection
        if "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd:
            tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
            print(f"[VAEContext DEBUG] conv1 shape: {tensor_conv1.shape} (chan={tensor_conv1.shape[0]})")
            if tensor_conv1.shape[0] == 512:
                print("[VAEContext DEBUG] Detected LTX VAE version 0 (LTX1)")
            elif tensor_conv1.shape[0] == 1024:
                if "encoder.down_blocks.1.conv.conv.bias" in sd:
                    print("[VAEContext DEBUG] Detected LTX VAE version 2 (LTX2 full)")
                else:
                    print("[VAEContext DEBUG] Detected LTX VAE version 1 (LTX2 basic)")

        try:
            vae_model = comfy.sd.VAE(sd=sd, metadata=metadata)
            vae_model.throw_exception_if_invalid()
        except Exception as e:
            raise RuntimeError(f"Could not load VAE model: {state.unet_path}. Error: {e}")

        vae_model.mmap_cache = sd
        return vae_model

    # ─── Hot load ────────────────────────────────────────────

    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any]) -> None:
        """Hot reload VAE using stream_vae_to_device logic."""
        print(f"[VAEContext] Hot-loading VAE to {device} via mmap stream...")

        self.stream_vae_to_device(model, device)

    def stream_vae_to_device(self, vae_model: Any, device: torch.device) -> bool:
        """Stream VAE weights to GPU per-tensor (avoids RAM spike)."""
        from raylight.expansion.comfyui_lazytensors.loader import SafetensorMmapWrapper
        from raylight.distributed_modules.utils import align_model_to_cuda

        mmap_cache = getattr(vae_model, "mmap_cache", None)

        if not mmap_cache:
            print(f"[VAEContext] Using standard .to({device})...")
            if hasattr(vae_model, "first_stage_model"):
                vae_model.first_stage_model.to(device)
            return False

        try:
            print(f"[VAEContext] Streaming VAE to {device} per-tensor...")
            wrapper = SafetensorMmapWrapper(mmap_cache)
            if hasattr(vae_model, "first_stage_model"):
                transferred = wrapper.stream_to_model(vae_model.first_stage_model, device)
                print(f"[VAEContext] Streamed {transferred} parameters to {device}")
                align_model_to_cuda(vae_model.first_stage_model)
                print("[VAEContext] Aligned VAE stragglers to CUDA")
            return True
        except Exception as e:
            print(f"[VAEContext] Streaming transfer failed: {e}, falling back...")
            if hasattr(vae_model, "first_stage_model"):
                vae_model.first_stage_model.to(device)
            return False

    # ─── Offload (Template Method hook) ──────────────────────

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "ActorConfigLike",
    ) -> bool:
        """Zero-copy soft-offload for VAE."""
        if model is None:
            return False

        mmap_cache = getattr(model, "mmap_cache", None)
        first_stage = getattr(model, "first_stage_model", None)

        if mmap_cache and first_stage:
            print(f"[VAEContext {config.local_rank}] Zero-Copy Offload: Restoring mmap pointers...")

            param_map = dict(first_stage.named_parameters())
            buffer_map = dict(first_stage.named_buffers())

            mmap_keys_original = set(mmap_cache.keys())
            mmap_keys_simple = {k.replace("first_stage_model.", "") for k in mmap_keys_original}

            restored = 0
            for name, mmap_tensor in mmap_cache.items():
                target_param = param_map.get(name)
                if target_param is None:
                    target_param = buffer_map.get(name)
                if target_param is None:
                    simple_name = name.replace("first_stage_model.", "")
                    target_param = param_map.get(simple_name)
                    if target_param is None:
                        target_param = buffer_map.get(simple_name)
                if target_param is not None:
                    target_param.data = mmap_tensor
                    restored += 1

            stragglers_moved = 0
            for name, param in param_map.items():
                if name not in mmap_keys_original and name not in mmap_keys_simple:
                    if param.device.type == "cuda":
                        param.data = param.data.to("cpu")
                        stragglers_moved += 1
            for name, buf in buffer_map.items():
                if name not in mmap_keys_original and name not in mmap_keys_simple:
                    if buf.device.type == "cuda":
                        buf.data = buf.data.to("cpu")
                        stragglers_moved += 1

            print(
                f"[VAEContext {config.local_rank}] Zero-Copy: "
                f"Restored {restored}/{len(mmap_cache)} tensors to mmap."
            )
            if stragglers_moved > 0:
                print(f"[VAEContext {config.local_rank}] Moved {stragglers_moved} stragglers to CPU.")

        else:
            print(f"[VAEContext {config.local_rank}] VAE Offload: No mmap cache, using standard offload...")
            if first_stage:
                first_stage.to("cpu")

        return False
