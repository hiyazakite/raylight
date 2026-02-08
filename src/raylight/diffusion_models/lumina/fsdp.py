from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import align_model_to_cuda


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload):
    diffusion_model = model.diffusion_model
    # Check for Quantization
    import os
    from raylight.distributed_modules.quantize import quantize_model
    use_kitchen = os.environ.get("RAYLIGHT_ENABLE_KITCHEN", "0") == "1"

    if use_kitchen:
        from raylight.distributed_modules.kitchen_patch import apply_patches
        apply_patches()

    ignored_params = set()
    for name, param in diffusion_model.named_parameters():
        if not name.startswith("layers."):
            ignored_params.add(param)

    for i, layer in enumerate(diffusion_model.layers):
        if use_kitchen:
             # Streaming Quantization: Quantize block before sharding
             quantize_model(layer, layout="fp8")

        diffusion_model.layers[i] = fully_shard(
            module=layer,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            offload_policy=CPUOffload(offload_params=enable_cpu_offload)
        )

    # Root wrap - Shards everything else
    if use_kitchen:
         # Quantize remaining components (embeddings, final layers, etc.)
         # Since we rely on ignore_modules or just specific targeting in flux, 
         # we can try quantize_model on the root BUT exclude 'layers' which are already sharded.
         # 'quantize_model' modifies Linear layers.
         # FSDP wrapped layers essentially hide the original Linears or replace them?
         # Safer: explicitly quantize specific remaining submodules if known.
         # Lumina usually has: 'cap_embedder', 'time_caption_embed', 'final_layer'
         # Let's target them if they exist.
         
         extras = ["cap_embedder", "time_caption_embed", "final_layer", "norm_final", "t_block", "y_embedder"]
         for name in extras:
             if hasattr(diffusion_model, name):
                 sub = getattr(diffusion_model, name)
                 if sub is not None:
                     quantize_model(sub, layout="fp8")

    fully_shard(diffusion_model, ignored_params=ignored_params, reshard_after_forward=True, offload_policy=CPUOffload(offload_params=enable_cpu_offload)) 
    model.diffusion_model = diffusion_model
    
    # Sync before loading state dict
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        align_model_to_cuda(model)

    if model_state_dict is not None:
        # If broadcast_from_rank0 is True, only rank 0 needs to load the state dict.
        # Other ranks can clear their local state dict to save massive amounts of RAM/VRAM.
        if dist.is_initialized() and dist.get_rank() > 0:
            model_state_dict.clear()

        set_model_state_dict(
            model=model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True, 
                cpu_offload=enable_cpu_offload
            ),
        )

    return model
