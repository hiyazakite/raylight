from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from raylight.distributed_modules.utils import align_model_to_cuda


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if (
            not name.startswith("single_blocks.")
            and not name.startswith("double_blocks.")
            and not name.startswith("txt_in.")
            and not name.startswith("guidance_in.")
            and not name.startswith("byt5_in.")
        ):
            ignored_params.add(param)

    if diffusion_model.byt5_in:
        diffusion_model.byt5_in = fully_shard(
            module=diffusion_model.byt5_in,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
        )

    diffusion_model.txt_in = fully_shard(
        module=diffusion_model.txt_in,
        mp_policy=MixedPrecisionPolicy(),
        reshard_after_forward=True,
    )

    diffusion_model.guidance_in = fully_shard(
        module=diffusion_model.guidance_in,
        mp_policy=MixedPrecisionPolicy(),
        reshard_after_forward=True,
    )

    for i, block in enumerate(diffusion_model.single_blocks):
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload),
        )

    for i, block in enumerate(diffusion_model.double_blocks):
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload),
        )

    # Root wrap with ignored params
    fully_shard(
        diffusion_model,
        ignored_params=ignored_params,
        mp_policy=MixedPrecisionPolicy(),
        reshard_after_forward=True,
    )

    model.diffusion_model = diffusion_model

    # Sync before loading
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        align_model_to_cuda(model)

    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=False,
            cpu_offload=enable_cpu_offload,
        ),
    )

    return model
