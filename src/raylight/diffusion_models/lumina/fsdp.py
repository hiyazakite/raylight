from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from raylight.distributed_modules.utils import detect_dtype_mismatch
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("layers."):
            ignored_params.add(param)

    ref_dtype = diffusion_model.layers[0].attention.qkv.weight.dtype
    
    from torch.distributed.fsdp import CPUOffload
    
    for i, layer in enumerate(diffusion_model.layers):
        # This is for scaled model
        ignored_block_params = detect_dtype_mismatch(layer, ref_dtype)
        diffusion_model.layers[i] = fully_shard(
            module=layer,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    fully_shard(diffusion_model, ignored_params=ignored_params, reshard_after_forward=True, offload_policy=CPUOffload(offload_params=enable_cpu_offload)) 
    model.diffusion_model = diffusion_model
    
    # CRITICAL FIX: If CPU offloading is disabled, FSDP (via fully_shard) moves wrapped layers to CUDA.
    # However, unwrapped parts (embeddings, final layers) stay on CPU (enforced by SamplerManager).
    # This creates a mixed-device model that crashes set_model_state_dict with "Multiple devices found".
    # We must rigorously align the entire model to CUDA if offload is False.
    if not enable_cpu_offload:
        import torch
        if torch.cuda.is_available():
            # This is safe; FSDP modules handle .to() by moving their local shard.
            # Unwrapped modules move fully.
            # We attempt to move the ROOT model (if possible) or at least the diffusion_model 
            # and all its submodules.
            
            # 1. Move the ENTIRE model (handles unwrapped params and parent buffers)
            # This fixes "Multiple devices found" if parent has buffers on CPU
            model.to("cuda")
            
            # 2. Iterate and force any stragglers (buffers, etc not in children? unlikely but safe)
            for p in model.parameters():
                if p.device.type != 'cuda':
                    p.data = p.to("cuda")
            for b in model.buffers():
                 if b.device.type != 'cuda':
                    b.data = b.to("cuda")

    
    # Sync before loading state dict
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True, 
            cpu_offload=enable_cpu_offload
        ),
    )

    return model
