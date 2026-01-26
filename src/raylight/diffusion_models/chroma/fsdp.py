from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions


def build_ignored_params(module, ref_dtype):
    ignored = set()

    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored.add(param)
            continue
        if not (name.endswith("weight") or name.endswith("bias")):
            ignored.add(param)
    return ignored


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model

    # Collect params we want to ignore (everything except single_blocks + double_blocks)
    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        # noqa: W503
        if (
            not name.startswith("single_blocks.")
            and not name.startswith("double_blocks.")
            and not name.startswith("distilled_guidance_layer.")
        ):
            ignored_params.add(param)
    # Shard distilled_guidance_layer
    ref_dtype = diffusion_model.distilled_guidance_layer.layers[0].in_layer.weight.dtype
    distil_ignored_params = build_ignored_params(
        diffusion_model.distilled_guidance_layer,
        ref_dtype
    )

    diffusion_model.distilled_guidance_layer = fully_shard(
        module=diffusion_model.distilled_guidance_layer,
        mp_policy=MixedPrecisionPolicy(),
        reshard_after_forward=True,
        ignored_params=distil_ignored_params,
    )

    # Shard single_blocks
    for i, block in enumerate(diffusion_model.single_blocks):
        diffusion_model.single_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload),
        )

    # Shard double_blocks
    for i, block in enumerate(diffusion_model.double_blocks):
        diffusion_model.double_blocks[i] = fully_shard(
            module=block,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload),
        )

    # Root wrap with ignored params
    fully_shard(diffusion_model,
                ignored_params=ignored_params,
                mp_policy=MixedPrecisionPolicy(),
                reshard_after_forward=True)

    model.diffusion_model = diffusion_model

    # Sync before loading
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        import torch
        if torch.cuda.is_available():
            for p in model.parameters():
                if not isinstance(p, torch.distributed.fsdp.FlatParameter) and p.device.type != 'cuda':
                    p.data = p.to("cuda")
            for b in model.buffers():
                 if b.device.type != 'cuda':
                    b.data = b.to("cuda")

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
