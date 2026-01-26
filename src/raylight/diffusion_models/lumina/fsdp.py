from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import align_model_to_cuda


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("layers."):
            ignored_params.add(param)

    for i, layer in enumerate(diffusion_model.layers):
        diffusion_model.layers[i] = fully_shard(
            module=layer,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    fully_shard(diffusion_model, ignored_params=ignored_params, reshard_after_forward=True, offload_policy=CPUOffload(offload_params=enable_cpu_offload)) 
    model.diffusion_model = diffusion_model
    
    # Sync before loading state dict
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        align_model_to_cuda(model)

    if model_state_dict is not None:
        # If broadcast_from_rank0 is True, only rank 0 needs to load the state dict.
        # Other ranks can clear their local state dict to save massive amounts of RAM/VRAM.
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

    return model
