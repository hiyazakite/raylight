# For FSDP2
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def shard_model_fsdp2(model, device_to):
    diffusion_model = model.diffusion_model

    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if (not name.startswith("single_blocks.")) and (not name.startswith("double_blocks.")):
            ignored_params.add(param)

    # And also blocks is the most compute heavy part
    diffusion_model.single_blocks = diffusion_model.single_blocks.to("cpu")
    diffusion_model.double_blocks = diffusion_model.double_blocks.to("cpu")
    for i, block in enumerate(diffusion_model.single_blocks):
        if "FSDP" not in block.__class__.__name__:
            diffusion_model.single_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )
    diffusion_model.single_blocks = diffusion_model.single_blocks.to(device_to)

    for i, block in enumerate(diffusion_model.double_blocks):
        if "FSDP" not in block.__class__.__name__:
            diffusion_model.double_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )
    diffusion_model.double_blocks = diffusion_model.double_blocks.to(device_to)

    # Model root wrap with ignored params
    fully_shard(diffusion_model, ignored_params=ignored_params)
    model.diffusion_model = diffusion_model

    print("SHARD COMPLETE")
    return model
