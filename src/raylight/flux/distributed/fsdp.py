from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
import torch
from raylight.distributed_modules.utils import ensure_no_scalar


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if (not name.startswith("single_blocks.")) and (not name.startswith("double_blocks.")):
            ignored_params.add(param)

    # Shard single_blocks
    for i, block in enumerate(diffusion_model.single_blocks):
        block = ensure_no_scalar(block)
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.bfloat16),
            reshard_after_forward=True,
        )

    # Shard double_blocks
    for i, block in enumerate(diffusion_model.double_blocks):
        block = ensure_no_scalar(block)
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.bfloat16),
            reshard_after_forward=True,
        )

    # Root wrap with ignored params
    fully_shard(diffusion_model,
                ignored_params=ignored_params,
                mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.bfloat16),
                reshard_after_forward=True)

    model.diffusion_model = diffusion_model

    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=enable_cpu_offload
        ),
    )

    print("SHARD COMPLETE")
    return model
