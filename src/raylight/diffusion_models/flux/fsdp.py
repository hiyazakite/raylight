from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import align_model_to_cuda
from raylight.distributed_modules.quantize import quantize_model
import os

def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    
    # Check for Quantization
    use_kitchen = os.environ.get("RAYLIGHT_ENABLE_KITCHEN", "0") == "1"
    
    # FSDP Preparation (Quantization) is memory intensive.
    # To avoid OOM, we must stream the quantization:
    # 1. Quantize Block N (Load -> FP8).
    # 2. Shard Block N (FP8 -> Shards).
    # 3. Repeat.
    # This keeps peak RAM low (~1 Block size extra) rather than Full Model size extra.
    
    # We initialize kitchen patches once globally if needed
    if use_kitchen:
        from raylight.distributed_modules.kitchen_patch import apply_patches
        apply_patches()
        
    # use_orig_params is NOT supported in FSDP2 (fully_shard).
    # FSDP2 natively shards parameters (per-parameter sharding).
    # If we are using FSDP1 (FullyShardedDataParallel class), we might need it.
    # But this function is `shard_model_fsdp2`, implying FSDP2.
    # Therefore, we should NOT pass use_orig_params.
    
    # We REMOVE the argument completely because fully_shard() in FSDP2 rejects it 
    # even if it is False.
    
    # Shard single_blocks (Streaming Quantization)
    from torch.distributed.fsdp import CPUOffload

    for i, block in enumerate(diffusion_model.single_blocks):
        if use_kitchen:
             # Quantize JUST this block before sharding
             # This spikes RAM by block_size, then sharding drops it.
             quantize_model(block, layout="fp8")
             
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    # Shard double_blocks (Streaming Quantization)
    for i, block in enumerate(diffusion_model.double_blocks):
        if use_kitchen:
             quantize_model(block, layout="fp8")
             
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    # Root wrap - Shards everything else (no skipped params)
    # We must quantize the remaining parts of diffusion_model (final_layer, embeddings etc)
    if use_kitchen:
         # Strategy: Quantize the *root* components explicitly.
         components_to_quantize = [
             "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer"
         ]
         for name in components_to_quantize:
             if hasattr(diffusion_model, name):
                 sub = getattr(diffusion_model, name)
                 if sub is not None:
                      quantize_model(sub, layout="fp8")
                      
    fully_shard(diffusion_model,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
                offload_policy=CPUOffload(offload_params=enable_cpu_offload))

    model.diffusion_model = diffusion_model

    # Sync before loading state dict
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        align_model_to_cuda(model)

    # If broadcast_from_rank0 is True, only rank 0 needs to load the state dict.
    # Other ranks can clear their local state dict to save massive amounts of RAM/VRAM.
    # If broadcast_from_rank0 is True, only rank 0 needs to load the state dict.
    # Other ranks can clear their local state dict to save massive amounts of RAM/VRAM.
    if model_state_dict is not None:
        if dist.is_initialized() and dist.get_rank() > 0:
            model_state_dict.clear()

        set_model_state_dict(
            model=model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                cpu_offload=enable_cpu_offload
            ),
        )
    else:
        # If state_dict is None, we assume weights are already in place (e.g. baked)
        # We still need to sync if distributed?
        # Initialization of FSDP handled parameters automatically if we don't load state dict.
        pass

    return model
