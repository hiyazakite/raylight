from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import align_model_to_cuda


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Shard single_blocks
    from torch.distributed.fsdp import CPUOffload

    for i, block in enumerate(diffusion_model.single_blocks):
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    # Shard double_blocks
    for i, block in enumerate(diffusion_model.double_blocks):
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    # Root wrap - Shards everything else (no skipped params)
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
