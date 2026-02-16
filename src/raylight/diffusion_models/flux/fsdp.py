from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from raylight.distributed_modules.utils import align_model_to_cuda
from raylight.distributed_modules.quantize import quantize_model
import os
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload, patcher=None):
    diffusion_model = model.diffusion_model
    
    # Check for Quantization
    use_kitchen = os.environ.get("RAYLIGHT_ENABLE_KITCHEN", "0") == "1"
    use_checkpointing = os.environ.get("RAYLIGHT_FSDP_GRADIENT_CHECKPOINTING", "0") == "1"
    use_parallel_disk = os.environ.get("RAYLIGHT_FSDP_PARALLEL_LOAD", "1") == "1"
    
    # FSDP Preparation (Quantization) is memory intensive.
    # To avoid OOM, we must stream the quantization:
    # 1. Quantize Block N (Load -> FP8).
    # 2. Shard Block N (FP8 -> Shards).
    # 3. Repeat.
    # This keeps peak RAM low (~1 Block size extra) rather than Full Model size extra.
    
    # We initialize kitchen patches once globally if needed
    if use_kitchen:
        from raylight.distributed_modules.kitchen_patch import apply_patches
        apply_patches()
        
    # use_orig_params is NOT supported in FSDP2 (fully_shard).
    # FSDP2 natively shards parameters (per-parameter sharding).
    


    # Streams for overlapping quantization and sharding
    stream_quant = torch.cuda.Stream()
    stream_shard = torch.cuda.Stream()

    def process_blocks(blocks):
    # Helper to bake block
        from raylight.distributed_modules.fsdp_utils import bake_lora_block
        def bake_block(block, block_prefix):
             bake_lora_block(block, block_prefix, patcher)

        # Prefetch first block quantization
        if len(blocks) > 0:
            with torch.cuda.stream(stream_quant):
                # Ensure block is on GPU for fast baking/quantization
                blocks[0].to("cuda")
                
                if patcher:
                     # Bake LoRAs for Next Block (0)
                     is_double = (blocks == diffusion_model.double_blocks)
                     prefix = "double_blocks.0" if is_double else "single_blocks.0"
                     bake_block(blocks[0], prefix)
                
                if use_kitchen:
                     quantize_model(blocks[0], layout="fp8")

        for i in range(len(blocks)):
            # Wait for quantization of current block
            torch.cuda.current_stream().wait_stream(stream_quant)
            
            # Prepare next block (Quantize + Bake)
            if i + 1 < len(blocks):
                with torch.cuda.stream(stream_quant):
                    # Ensure next block is on GPU
                    blocks[i+1].to("cuda")

                    # Determine prefix for next block
                    # blocks is either single_blocks or double_blocks
                    is_double = (blocks == diffusion_model.double_blocks)
                    next_idx = i + 1
                    prefix = f"double_blocks.{next_idx}" if is_double else f"single_blocks.{next_idx}"
                    
                    if patcher:
                        bake_block(blocks[next_idx], prefix)

                    if use_kitchen:
                        quantize_model(blocks[next_idx], layout="fp8")
            
            # Shard current block
            with torch.cuda.stream(stream_shard):
                current_block = blocks[i]
                
                # Apply Gradient Checkpointing if enabled
                if use_checkpointing:
                    current_block = checkpoint_wrapper(
                        current_block,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT
                    )
                
                # Idempotency check
                if isinstance(current_block, FSDPModule):
                    continue

                try:
                    blocks[i] = fully_shard(
                        module=current_block,
                        mp_policy=MixedPrecisionPolicy(),
                        reshard_after_forward=True,
                        offload_policy=CPUOffload(offload_params=enable_cpu_offload)
                    )
                except AssertionError as e:
                    if "fully_shard has already been applied" in str(e):
                        print(f"Block {i} already sharded, skipping...")
                        continue
                    raise e
        
    # Sync shard stream after processing all blocks
        torch.cuda.current_stream().wait_stream(stream_shard)
        
        # Explicit cleanup to release intermediate buffers from streams
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Process single blocks
    process_blocks(diffusion_model.single_blocks)

    # Process double blocks
    process_blocks(diffusion_model.double_blocks)

    # Root wrap - Shards everything else (no skipped params)
    # We must quantize the remaining parts of diffusion_model (final_layer, embeddings etc)
    if use_kitchen:
         # Strategy: Quantize the *root* components explicitly.
         components_to_quantize = [
             "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer"
         ]
         for name in components_to_quantize:
             if hasattr(diffusion_model, name):
                 sub = getattr(diffusion_model, name)
                 if sub is not None:
                      quantize_model(sub, layout="fp8")
                      
    if not isinstance(diffusion_model, FSDPModule):
        try:
            fully_shard(diffusion_model,
                        mp_policy=MixedPrecisionPolicy(),
                        reshard_after_forward=True,
                        offload_policy=CPUOffload(offload_params=enable_cpu_offload))
        except AssertionError as e:
            if "fully_shard has already been applied" in str(e):
                print("Root model already sharded, skipping...")
            else:
                raise e

    model.diffusion_model = diffusion_model

    # Sync before loading state dict
    if dist.is_initialized():
        dist.barrier()

    # Align stragglers (params/buffers not wrapped by FSDP) to CUDA
    if not enable_cpu_offload:
        # Before aligning, let's clear cache again just in case
        torch.cuda.empty_cache()
        align_model_to_cuda(model)

    # If broadcast_from_rank0 is True, only rank 0 needs to load the state dict.
    # Other ranks can clear their local state dict to save massive amounts of RAM/VRAM.
    
    broadcast_from_rank0 = not use_parallel_disk
    
    if model_state_dict is not None:
        if dist.is_initialized() and dist.get_rank() > 0 and broadcast_from_rank0:
            model_state_dict.clear()

        set_model_state_dict(
            model=model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=broadcast_from_rank0,
                cpu_offload=enable_cpu_offload
            ),
        )
    else:
        # If state_dict is None, we assume weights are already in place (e.g. baked)
        pass

    return model
