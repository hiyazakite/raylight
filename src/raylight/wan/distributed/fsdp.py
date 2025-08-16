# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy


def shard_model(
    model,
    device_id,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    model.blocks = model.blocks.to("cpu")

    # fp8 is not currently supported in FSDP, if param_dtype = model.dtype
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
