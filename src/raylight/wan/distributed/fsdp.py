from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FSDPModule


def shard_model_fsdp2(model, device_to):
    diffusion_model = model.diffusion_model

    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    if not isinstance(diffusion_model, FSDPModule):
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if not name.startswith("blocks."):
                ignored_params.add(param)

        # And also blocks is the most compute heavy part
        diffusion_model.blocks = diffusion_model.blocks.to("cpu")
        for i, block in enumerate(diffusion_model.blocks):
            diffusion_model.blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )
        diffusion_model.blocks = diffusion_model.blocks.to(device_to)

        # Model root wrap with ignored params
        fully_shard(diffusion_model, ignored_params=ignored_params)
        model.diffusion_model = diffusion_model

        print("SHARD COMPLETE")
        return model
    else:
        print("FSDP Already applied, skipping")
        return model
