import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, clip_fea, time_dim_concat, transformer_options = args

    try:
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    except Exception as e:
        raise ValueError(
            "You might disable CFG where CFG=1.0. Please enable another method of parallelism or set CFG >1", e
        )
    timestep = torch.chunk(timestep, cfg_world_size, dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size, dim=0)[cfg_rank]

    if clip_fea is not None:
        clip_fea = torch.chunk(clip_fea, cfg_world_size, dim=0)[cfg_rank]

    if time_dim_concat is not None:
        time_dim_concat = torch.chunk(time_dim_concat, cfg_world_size, dim=0)[cfg_rank]

    result = executor(x, timestep, context, clip_fea, time_dim_concat, transformer_options, **kwargs)
    result = get_cfg_group().all_gather(result, dim=0)
    return result
