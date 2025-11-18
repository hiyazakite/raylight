import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_cfg_group,
)

# def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
#     cfg_rank = get_classifier_free_guidance_rank()
#     cfg_world_size = get_classifier_free_guidance_world_size()
#
#     x = torch.chunk(x, cfg_world_size,dim=0)[cfg_rank]
#     timestep = torch.chunk(timestep, cfg_world_size,dim=0)[cfg_rank]
#     context = torch.chunk(context, cfg_world_size,dim=0)[cfg_rank]
#     y = torch.chunk(y, cfg_world_size,dim=0)[cfg_rank]
#
#     output = self.unet(x, timestep, context, y, control, transformer_options, **kwargs).contiguous()
#     model_forward = executor(x, timestep, context, y, *args, **kwargs)
#     model_forward = get_cfg_group().all_gather(model_forward, dim=0)
#
#     return model_forward


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    x: torch.tensor = args[0]
    timestep = args[1]
    context = args[2]
    clip_fea = args[3]
    time_dim_concat = args[4]

    if clip_fea is not None:
        print(f"shape of clip_fea ===================== {clip_fea.shape=}")

    if time_dim_concat is not None:
        print(f"shape of time_dim_concat ===================== {time_dim_concat.shape=}")

    print(f"shape of x ===================== {x.shape=}")
    print(f"shape of timestep ===================== {timestep.shape=}")
    print(f"shape of context ===================== {context.shape=}")
    return executor(*args, **kwargs)

def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()

    model, conds, x_in, timestep, model_options = args
    conds = [conds[cfg_rank]]

    new_args = (model, conds, x_in, timestep, model_options)
    return executor(*new_args, **kwargs)
