from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FSDPModule


def shard_model_fsdp2(model, device_to):
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    if not isinstance(diffusion_model, FSDPModule):
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if (not name.startswith("single_blocks.")) and (not name.startswith("double_blocks.")):
                ignored_params.add(param)

        # Move compute-heavy blocks to CPU before sharding
        diffusion_model.single_blocks = diffusion_model.single_blocks.to("cpu")
        diffusion_model.double_blocks = diffusion_model.double_blocks.to("cpu")

        # Shard single_blocks
        for i, block in enumerate(diffusion_model.single_blocks):
            diffusion_model.single_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )
        diffusion_model.single_blocks = diffusion_model.single_blocks.to(device_to)

        # Shard double_blocks
        for i, block in enumerate(diffusion_model.double_blocks):
            diffusion_model.double_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )
        diffusion_model.double_blocks = diffusion_model.double_blocks.to(device_to)

        # Root wrap with ignored params
        fully_shard(diffusion_model, ignored_params=ignored_params)
        model.diffusion_model = diffusion_model

        print("SHARD COMPLETE")
        return model
    else:
        print("FSDP Already applied, skipping")
        return model
