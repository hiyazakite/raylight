import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, y, time_dim_replace, transformer_options = args

    if x.shape[0] == cfg_world_size:
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    else:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    timestep = torch.chunk(timestep, cfg_world_size, dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size, dim=0)[cfg_rank]

    if y is not None:
        y = torch.chunk(y, cfg_world_size, dim=0)[cfg_rank]

    if time_dim_replace is not None:
        time_dim_replace = torch.chunk(time_dim_replace, cfg_world_size, dim=0)[cfg_rank]

    result = executor(x, timestep, context, y, time_dim_replace, transformer_options, **kwargs)
    result = get_cfg_group().all_gather(result, dim=0)
    return result
