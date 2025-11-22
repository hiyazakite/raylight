import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, timestep, context, attention_mask, guidance, ref_latents, transformer_options = args

    if x.shape[0] == cfg_world_size:
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    else:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    timestep = torch.chunk(timestep, cfg_world_size, dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size, dim=0)[cfg_rank]

    if attention_mask is not None:
        attention_mask = torch.chunk(attention_mask, cfg_world_size, dim=0)[cfg_rank]

    if guidance is not None:
        guidance = torch.chunk(guidance, cfg_world_size, dim=0)[cfg_rank]

    if ref_latents is not None:
        ref_latents = torch.chunk(ref_latents, cfg_world_size, dim=0)[cfg_rank]

    result = executor(x, timestep, context, attention_mask, guidance, ref_latents, transformer_options, **kwargs)
    result = get_cfg_group().all_gather(result, dim=0)
    return result
