# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
# For FSDP1
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# For FSDP2
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule


# Not being used!
def shard_model(
    model,
    device_id,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    model.blocks = model.blocks.to("cpu")

    # fp8 is not currently supported in FSDP1, if param_dtype = model.dtype
    # It would resulted into "ufunc_add_CUDA" not implemented for 'Float8_e4m3fn'
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32,
    buffer_dtype = torch.float32,

    for i, block in enumerate(model.blocks):
        model.blocks[i] = FSDP(
            module=block,
            process_group=process_group,
            sharding_strategy=sharding_strategy,
            mixed_precision=MixedPrecision(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                buffer_dtype=buffer_dtype
            ),
            device_id=device_id,
            sync_module_states=False,
            use_orig_params=True
        )

    return model


def shard_model_fsdp2(model):
    diffusion_model = model.diffusion_model

    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("blocks."):
            ignored_params.add(param)

    # And also blocks is the most compute heavy part
    with torch.no_grad():
        for i, block in enumerate(diffusion_model.blocks):
            if not isinstance(block, FSDPModule):
                diffusion_model.blocks[i] = fully_shard(
                    module=block,
                    mp_policy=MixedPrecisionPolicy(),
                    reshard_after_forward=True,
                )

        # Model root wrap with ignored params
        fully_shard(diffusion_model, ignored_params=ignored_params)

    print("SHARD COMPLETE")
    return model
