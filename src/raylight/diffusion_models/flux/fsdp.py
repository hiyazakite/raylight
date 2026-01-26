from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import detect_dtype_mismatch


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if (not name.startswith("single_blocks.")) and (not name.startswith("double_blocks.")):
            ignored_params.add(param)

    # Check dtype missmatch from scaled model
    ref_dtype = diffusion_model.double_blocks[0].img_attn.qkv.weight.dtype

    # Shard single_blocks
    for i, block in enumerate(diffusion_model.single_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
        )

    # Shard double_blocks
    for i, block in enumerate(diffusion_model.double_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
        )

    # Root wrap with ignored params
    fully_shard(diffusion_model,
                ignored_params=ignored_params,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True)

    model.diffusion_model = diffusion_model

    # CRITICAL: Ensure entire model is on CUDA if offloading is disabled
    # This prevents "Multiple devices found" errors for unwrapped parameters/buffers
    if not enable_cpu_offload:
        import torch
        if torch.cuda.is_available():
            model.to("cuda")
            # Force stragglers
            for p in model.parameters():
                if p.device.type != 'cuda':
                    p.data = p.to("cuda")
            for b in model.buffers():
                 if b.device.type != 'cuda':
                    b.data = b.to("cuda")

    set_model_state_dict(
        model=model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=False,
            cpu_offload=enable_cpu_offload
        ),
    )

    return model
