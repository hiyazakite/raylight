import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    (
        x,
        timestep,
        context,
        y,
        txt_byt5,
        guidance,
        attention_mask,
        guiding_frame_index,
        ref_latent,
        disable_time_r,
        control,
        transformer_options,
    ) = args

    if x.shape[0] == cfg_world_size:
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    else:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    timestep = torch.chunk(timestep, cfg_world_size, dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size, dim=0)[cfg_rank]

    if y is not None:
        y = torch.chunk(y, cfg_world_size, dim=0)[cfg_rank]

    if txt_byt5 is not None:
        txt_byt5 = torch.chunk(txt_byt5, cfg_world_size, dim=0)[cfg_rank]

    if guidance is not None:
        guidance = torch.chunk(guidance, cfg_world_size, dim=0)[cfg_rank]

    if attention_mask is not None:
        attention_mask = torch.chunk(attention_mask, cfg_world_size, dim=0)[cfg_rank]

    if guiding_frame_index is not None:
        guiding_frame_index = torch.chunk(guiding_frame_index, cfg_world_size, dim=0)[
            cfg_rank
        ]

    if ref_latent is not None:
        ref_latent = torch.chunk(ref_latent, cfg_world_size, dim=0)[cfg_rank]

    if disable_time_r is not None:
        disable_time_r = torch.chunk(disable_time_r, cfg_world_size, dim=0)[cfg_rank]

    if control is not None:
        control = torch.chunk(control, cfg_world_size, dim=0)[cfg_rank]

    result = executor(
        x,
        timestep,
        context,
        y,
        txt_byt5,
        guidance,
        attention_mask,
        guiding_frame_index,
        ref_latent,
        disable_time_r,
        control,
        transformer_options,
        **kwargs
    )
    result = get_cfg_group().all_gather(result, dim=0)
