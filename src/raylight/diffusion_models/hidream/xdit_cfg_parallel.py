import torch
from xfuser.core.distributed import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_cfg_group,
)


def cfg_parallel_forward_wrapper(executor, *args, **kwargs):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    x, t, y, context, encoder_hidden_states_llama3, image_cond, control, transformer_options = args

    if x.shape[0] == cfg_world_size:
        x = torch.chunk(x, cfg_world_size, dim=0)[cfg_rank]
    else:
        raise ValueError("CFG = 1.0, disables guidance. Increase CFG > 1.0 or switch to another parallelism mode")
    t = torch.chunk(t, cfg_world_size, dim=0)[cfg_rank]
    context = torch.chunk(context, cfg_world_size, dim=0)[cfg_rank]

    if y is not None:
        y = torch.chunk(y, cfg_world_size, dim=0)[cfg_rank]

    if encoder_hidden_states_llama3 is not None:
        encoder_hidden_states_llama3 = torch.chunk(encoder_hidden_states_llama3, cfg_world_size, dim=0)[cfg_rank]

    if image_cond is not None:
        image_cond = torch.chunk(image_cond, cfg_world_size, dim=0)[cfg_rank]

    if control is not None:
        control = torch.chunk(control, cfg_world_size, dim=0)[cfg_rank]

    result = executor(x, t, y, context, encoder_hidden_states_llama3, image_cond, control, transformer_options, **kwargs)
    result = get_cfg_group().all_gather(result, dim=0)
    return result
