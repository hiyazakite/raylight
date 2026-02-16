
import torch

def prefetch_state_dict(unet_path: str) -> dict:
    """Load state dict in background thread"""
    print(f"[FSDP Prefetch] Starting background load for {unet_path}")
    sd = torch.load(unet_path, map_location="cpu", weights_only=True)
    print(f"[FSDP Prefetch] Finished background load for {unet_path}")
    return sd


def bake_lora_block(block, block_prefix, patcher):
    """
    Applies LoRA patches to a model block in-place.
    Moves standard CPU parameters to GPU before patching to optimize speed and RAM usage.
    """
    if not patcher or not patcher.patches:
        return

    # We need to reconstruct the full parameter name for the patcher
    for name, param in block.named_parameters(recurse=True):
        full_key = f"diffusion_model.{block_prefix}.{name}"
        if full_key in patcher.patches:
            # OPTIMIZATION: Move to GPU to avoid RAM OOM during baking
            if param.device.type == "cpu":
                param.data = param.data.to("cuda")

            patcher.patch_weight_to_device(full_key, device_to=None, inplace_update=True)
            # Clean up partials to save RAM
            if hasattr(patcher, "backup"):
                patcher.backup.clear()
