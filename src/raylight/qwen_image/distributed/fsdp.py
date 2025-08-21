from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def shard_model_fsdp2(model, device_to):
    diffusion_model = model.diffusion_model

    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("transformer_blocks."):
            ignored_params.add(param)

    # And also blocks is the most compute heavy part
    diffusion_model.transformer_blocks = diffusion_model.transformer_blocks.to("cpu")
    for i, block in enumerate(diffusion_model.transformer_blocks):
        if "FSDP" not in block.__class__.__name__:
            diffusion_model.transformer_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True,
            )
    diffusion_model.transformer_blocks = diffusion_model.transformer_blocks.to(device_to)

    # Model root wrap with ignored params
    fully_shard(diffusion_model, ignored_params=ignored_params)
    model.diffusion_model = diffusion_model

    print("SHARD COMPLETE")
    return model
