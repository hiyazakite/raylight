import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
)

def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size

    x = torch.chunk(x, cfg_world_size,dim=0)[cfg_rank]
    timestep = torch.chunk(timestep, cfg_world_size,dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size,dim=0)[cfg_rank]
    y = torch.chunk(y, cfg_world_size,dim=0)[cfg_rank]

    output = self.unet(x, timestep, context, y, control, transformer_options, **kwargs).contiguous()
    model_forward = executor(x, timestep, context, y, *args, **kwargs)
    model_forward = get_cfg_group().all_gather(model_forward, dim=0)

    return model_forward
