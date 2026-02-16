from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import align_model_to_cuda



def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload, patcher=None):
    diffusion_model = model.diffusion_model
    # Check for Quantization
    import os
    import torch
    import torch.distributed as dist
    from raylight.distributed_modules.quantize import quantize_model
    use_kitchen = os.environ.get("RAYLIGHT_ENABLE_KITCHEN", "0") == "1"
    use_parallel_disk = os.environ.get("RAYLIGHT_FSDP_PARALLEL_LOAD", "1") == "1"

    if use_kitchen:
        from raylight.distributed_modules.kitchen_patch import apply_patches
        apply_patches()

    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("layers."):
            ignored_params.add(param)

    # Helper to bake LoRAs into a block
    from raylight.distributed_modules.fsdp_utils import bake_lora_block
    def bake_block(block, block_prefix):
         bake_lora_block(block, block_prefix, patcher)

    # Pre-bake the first block if patcher exists
    if len(diffusion_model.layers) > 0:
        # Move first layer to CUDA for Pre-bake / First Loop Quantization
        diffusion_model.layers[0].to("cuda")
        if patcher:
            bake_block(diffusion_model.layers[0], "layers.0")

    for i, layer in enumerate(diffusion_model.layers):
        # Ensure layer is on CUDA (idempotent if already moved)
        layer.to("cuda")

        if use_kitchen:
             # Streaming Quantization: Quantize block before sharding
             from raylight.distributed_modules.quantize import quantize_model
             quantize_model(layer, layout="fp8")
        
        # Bake the NEXT block (i+1) while current is sharding (prefetch style)
        if (i + 1) < len(diffusion_model.layers):
             # Move next layer to CUDA for baking
             diffusion_model.layers[i+1].to("cuda")
             if patcher:
                 bake_block(diffusion_model.layers[i+1], f"layers.{i+1}")

        # Idempotency check: Skip if already wrapped
        if isinstance(layer, FSDPModule):
             continue

        # Double safety: try-except to catch manual re-wrapping attempts
        try:
            diffusion_model.layers[i] = fully_shard(
                module=layer,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
                offload_policy=CPUOffload(offload_params=enable_cpu_offload)
            )
        except AssertionError as e:
            if "fully_shard has already been applied" in str(e):
                print(f"Layer {i} already sharded, skipping...")
                continue
            raise e
        
        # Cleanup VRAM after each block shard
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # Root wrap - Shards everything else
    if not isinstance(diffusion_model, FSDPModule):
        try:
            fully_shard(diffusion_model, ignored_params=ignored_params, reshard_after_forward=True, offload_policy=CPUOffload(offload_params=enable_cpu_offload)) 
            model.diffusion_model = diffusion_model
        except AssertionError as e:
            if "fully_shard has already been applied" in str(e):
                print("Root model already sharded, skipping...")
            else:
                raise e

    # If broadcast_from_rank0 is True, only rank 0 needs to load the state dict.
    # Other ranks can clear their local state dict to save massive amounts of RAM/VRAM.
    
    broadcast_from_rank0 = not use_parallel_disk

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        align_model_to_cuda(model)

    if model_state_dict is not None:
        if dist.is_initialized() and dist.get_rank() > 0 and broadcast_from_rank0:
            model_state_dict.clear()

        set_model_state_dict(
            model=model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=broadcast_from_rank0, 
                cpu_offload=enable_cpu_offload
            ),
        )

    return model
