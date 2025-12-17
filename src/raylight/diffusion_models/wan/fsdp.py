from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from raylight.distributed_modules.utils import ensure_no_scalar
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    for param in diffusion_model.parameters():
        param.requires_grad_(False)

    diffusion_model = ensure_no_scalar(diffusion_model)

    # Collect params we want to ignore (everything except blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("blocks."):
            ignored_params.add(param)

    for i, block in enumerate(diffusion_model.blocks):
        diffusion_model.blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
        )

    fully_shard(diffusion_model, ignored_params=ignored_params, reshard_after_forward=True)
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

    return model
