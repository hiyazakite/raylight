from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import detect_dtype_mismatch


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Debug: Check for large buffers (which FSDP ignores)
    large_buffers = []
    for name, buf in diffusion_model.named_buffers():
        size_mb = buf.numel() * buf.element_size() / (1024 * 1024)
        if size_mb > 100:
            large_buffers.append((name, size_mb))
            
    if large_buffers:
        print(f"[FSDP Flux] Found {len(large_buffers)} Large Buffers > 100MB:")
        for name, size in large_buffers:
            print(f"  - {name}: {size:.2f} MB")
    else:
        print("[FSDP Flux] No large buffers found.")

    # Check dtype missmatch from scaled model
    ref_dtype = diffusion_model.double_blocks[0].img_attn.qkv.weight.dtype
    print(f"[FSDP Flux] Model Dtype: {ref_dtype}")

    # Shard single_blocks
    from torch.distributed.fsdp import CPUOffload

    for i, block in enumerate(diffusion_model.single_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    # Shard double_blocks
    for i, block in enumerate(diffusion_model.double_blocks):
        ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            ignored_params=ignored_block_params,
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
