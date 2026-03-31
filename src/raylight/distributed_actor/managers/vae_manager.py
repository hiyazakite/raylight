import os
import torch
from raylight.utils.memory import monitor_memory, MemoryPolicy, NULL_POLICY
from raylight.utils.common import cleanup_memory, force_malloc_trim

from typing import Optional, Any
from raylight.distributed_actor.actor_config import ActorConfig

# VAE debugging stats (formerly RAYLIGHT_DEBUG_VAE) enabled via config.debug.debug_vae

class VaeManager:
    def __init__(self):
        self.vae_model = None
        self._cached_dtype = None    # (preference, dtype) — avoids redundant logging per chunk
        self._mmap_cache = None      # (path, shape_tuple, buffer) — reuse across work-stealing chunks

    def load_vae(
        self, 
        vae_path: str, 
        config: ActorConfig, 
        state_cache: Any,
        memory: MemoryPolicy = NULL_POLICY,
    ) -> Optional[Any]:
        from raylight.distributed_actor.model_context import get_context, ModelState
        from raylight.comfy_dist.sd import decode_tiled_1d, decode_tiled_, decode_tiled_3d
        import types

        print(f"[RayActor {config.local_rank}] VAE Load via ModelContext: {vae_path}")
        
        # 1. Get Unified Context (VAEContext for VAE models)
        ctx = get_context(vae_path, config, model_type="vae")
        
        # 2. Create Load State
        state = ModelState(
            unet_path=vae_path,
            model_options={},
        )
        
        # 3. Unified Load (Handles Caching & Instantiation)
        vae_model = ctx.load(state, config, state_cache)
        
        # 5. Monkey patch decode optimizations
        vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
        vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
        vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)

        # 6. Stream weights to GPU (handled by context, same pattern as UNET)
        from raylight.distributed_actor.model_context import VAEContext
        assert isinstance(ctx, VAEContext)
        ctx.stream_vae_to_device(vae_model, config.device)

        self.vae_model = vae_model
        
        memory.after_reload()
        
        return vae_model


    def offload_vae_from_device(self, config: ActorConfig, lora_manager: Optional[Any] = None, memory: MemoryPolicy = NULL_POLICY):
        """Explicitly offload VAE from VRAM back to CPU/mmap after work-stealing completes."""
        # Release per-job caches (mmap buffer, dtype decision)
        self._mmap_cache = None
        self._cached_dtype = None
        if self.vae_model is None:
            return
        from raylight.distributed_actor.model_context import VAEContext
        vae_ctx = VAEContext(use_mmap=config.raylight_config.device.use_mmap)
        vae_ctx.offload(self.vae_model, lora_manager, {}, config, memory=memory)
        print(f"[RayActor {config.local_rank}] VAE offloaded from VRAM.")

    def release_vae(self, local_rank: int, memory: MemoryPolicy = NULL_POLICY):
        """Explicitly release VAE model from memory to free RAM for other operations."""
        self._mmap_cache = None
        self._cached_dtype = None
        if self.vae_model is not None:
            print(f"[RayActor {local_rank}] Releasing VAE model from RAM...")
            del self.vae_model
            self.vae_model = None
            
            memory.teardown()
            
            print(f"[RayActor {local_rank}] VAE released.")
        return True

    def get_temporal_compression(self):
        if self.vae_model is None:
            return None
        return self.vae_model.temporal_compression_decode()

    def get_spatial_compression(self):
        if self.vae_model is None:
            return None
        return self.vae_model.spacial_compression_decode()

    def check_health(self, samples, shard_index, config: ActorConfig, overwrite_cast_dtype=None):
        # Diagnostic: Check input latent statistics
        l_min, l_max, l_mean = samples["samples"].min().item(), samples["samples"].max().item(), samples["samples"].mean().item()
        print(f"[RayActor {config.local_rank}] Input Latent Stats (shard {shard_index}): min={l_min:.4f}, max={l_max:.4f}, mean={l_mean:.4f}")
        if torch.isnan(samples["samples"]).any():
            print(f"[RayActor {config.local_rank}] CRITICAL: Input latents for shard {shard_index} contain NaNs!")
        
        # Check if overwrite_cast_dtype is set (from patch_temp_fix_ck_ops)
        if overwrite_cast_dtype is not None:
             print(f"[RayActor {config.local_rank}] Debug: overwrite_cast_dtype is set to {overwrite_cast_dtype}")
        
        # Check VAE weights for NaNs BEFORE decoding
        if self.vae_model is not None:
            nan_params = []
            for name, p in self.vae_model.first_stage_model.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(name)
            if nan_params:
                print(f"[RayActor {config.local_rank}] CRITICAL: VAE parameters contain NaNs: {nan_params[:5]}...")
            else:
                print(f"[RayActor {config.local_rank}] VAE parameters verified healthy (no NaNs).")

    def decode(
        self,
        shard_index,
        samples,
        tile_size,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        discard_latent_frames=0,
        vae_dtype="auto",
        mmap_path=None,
        mmap_shape=None,
        output_offset=0,
        config: Optional[ActorConfig] = None,
        lora_manager: Optional[Any] = None,
        state_cache: Optional[Any] = None,
        latent_ref=None,
        latent_slice_start=None,
        latent_slice_end=None,
        skip_offload=False,
        memory: MemoryPolicy = NULL_POLICY,
    ):
        if config is None:
             raise ValueError("ActorConfig must be passed to decode")

        # NOTE: Removed eager gc.collect() + torch.cuda.empty_cache() that blocked CPU/GPU
        # for hundreds of ms at the start of every decode call. Cleanup happens in the
        # finally block after decode completes.
        print(f"[RayActor {config.local_rank}] Entering ray_vae_decode for shard {shard_index}")
        
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        
        try:
            assert self.vae_model is not None, "VAE model must be loaded before decode"
            temporal_compression = self.vae_model.temporal_compression_decode()
            if temporal_compression is not None:
                temporal_size = max(2, temporal_size // temporal_compression)
                temporal_overlap = max(
                    1, min(temporal_size // 2, temporal_overlap // temporal_compression)
                )
            else:
                temporal_size = None
                temporal_overlap = None
    
            compression = self.vae_model.spacial_compression_decode()
    
            # 1. Select and Configure Dtype
            dtype = self._select_dtype(vae_dtype, config.local_rank)
            # Only cast when dtype actually changes (skips iterating all params on repeat chunks)
            if getattr(self.vae_model, 'vae_dtype', None) != dtype:
                self.vae_model.first_stage_model.to(dtype)
                self.vae_model.vae_dtype = dtype
            
            # 2. Transfer Latents (Optimized)
            # If latent_ref is provided, it's already resolved by Ray's actor call
            # mechanism (ObjectRefs passed to .remote() are auto-deserialized).
            # We just slice the full tensor directly — zero-copy from shared memory.
            if latent_ref is not None and latent_slice_start is not None:
                shard_latents = latent_ref[:, :, latent_slice_start:latent_slice_end]
            else:
                shard_latents = samples["samples"]
            
            # Use non_blocking=True to overlap PCIe transfer with CPU setup work
            latents_to_decode = shard_latents.to(config.device, dtype=dtype, non_blocking=True)
            del shard_latents
            print(f"[RayActor {config.local_rank}] latents shape: {latents_to_decode.shape}, tiling: tile_t={temporal_size}, overlap_t={temporal_overlap}")
    
            # Sync the non_blocking transfer before decode begins
            torch.cuda.current_stream(config.device).synchronize()
    
            # ── Reactive pressure check before decode activations allocate ──
            # We can't predict decoder activation size (depends on tile/temporal
            # config and model internals), so we just check if we're already
            # tight and proactively free more UNET tail if needed.
            freed = memory.relieve_pressure(needed_bytes=0)
            if freed > 0:
                print(f"[RayActor {config.local_rank}] Pre-decode pressure relief: "
                      f"freed {freed/1e9:.2f} GB for VAE activations")

            # 3. Decode
            # ComfyUI's decode_tiled returns [B, T, H, W, C] for 3D or [B, H, W, C] for 2D
            images = self.vae_model.decode_tiled(
                latents_to_decode,
                tile_x=tile_size // compression,
                tile_y=tile_size // compression,
                overlap=overlap // compression,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )
            print(f"[RayActor {config.local_rank}] decode_tiled complete. Shape: {images.shape}")
            
            # Post-processing: discard warmup
            # 4. Post-Process (Shape Fix + Warmup Discard)
            images = self._post_process_images(images, discard_latent_frames, config.local_rank, config, shard_index)
            
            # 5. Handle Output (Mmap or CPU Transfer)
            # RETURNS: (shard_index, result_payload)
            result = self._handle_output(images, shard_index, mmap_path, mmap_shape, output_offset, config.local_rank, config)
    
            # Proactive cleanup (images tensor is referenced in result if not mmap, but original is gone)
            del latents_to_decode
            
            return result
        
        finally:
            # Offload VAE from GPU back to CPU/mmap unless skip_offload is set.
            # With dynamic work-stealing, the same worker may receive more chunks,
            # so we keep the VAE on GPU to avoid the costly re-stream.
            # The master calls release_vae (or explicit offload) after all chunks complete.
            if not skip_offload:
                from raylight.distributed_actor.model_context import VAEContext
                vae_ctx = VAEContext(use_mmap=config.raylight_config.device.use_mmap)
                vae_ctx.offload(self.vae_model, lora_manager, {}, config)
                print(f"[RayActor {config.local_rank}] VAE Offload in finally block complete.")
                force_malloc_trim()

    def _select_dtype(self, preference, local_rank):
        """Selects appropriate VAE dtype based on hardware and preference."""
        # Fast path: return cached dtype for repeat work-stealing chunks
        if self._cached_dtype is not None and self._cached_dtype[0] == preference:
            return self._cached_dtype[1]

        if preference == "auto":
            if torch.cuda.is_bf16_supported():
                print(f"[RayActor {local_rank}] Using bfloat16 VAE (auto: bf16 supported)")
                dtype = torch.bfloat16
            else:
                print(f"[RayActor {local_rank}] Using float32 VAE (auto: bf16 not supported)")
                dtype = torch.float32
        elif preference == "bf16":
            print(f"[RayActor {local_rank}] Using bfloat16 VAE (user selected)")
            dtype = torch.bfloat16
        elif preference == "fp16":
            print(f"[RayActor {local_rank}] Using float16 VAE (user selected - may cause NaN on some models)")
            dtype = torch.float16
        else:  # fp32
            print(f"[RayActor {local_rank}] Using float32 VAE (user selected - stable but 2x memory)")
            dtype = torch.float32

        self._cached_dtype = (preference, dtype)
        return dtype

    def _post_process_images(self, images, discard_latent_frames, local_rank, config: ActorConfig, shard_index=None):
        """Normalizes image shape and handles warmup frame discarding."""
        # Debug stats (disabled by default — each .item() forces a GPU→CPU sync stall)
        if config.raylight_config.debug.debug_vae:
            raw_min, raw_max, raw_mean = images.min().item(), images.max().item(), images.mean().item()
            print(f"[RayActor {local_rank}] RAW decode stats: min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}")
            if torch.isnan(images).any():
                print(f"[RayActor {local_rank}] CRITICAL: VAE output STILL contains NaNs even in float32!")
        
        # Shape normalization: ensure [Batch, Time, Height, Width, Channels]
        # input is typically [B, H, W, C] or [B, T, H, W, C] (after movedim)
        if len(images.shape) == 4:
            # [B, H, W, C] -> Insert Time dimension: [B, 1, H, W, C]
            images = images.unsqueeze(1)
            
        # Discard warmup frames (temporal sharding logic)
        if discard_latent_frames > 0 and self.vae_model is not None:
            # Fallback to 1 for 2D VAEs to avoid massive over-discarding
            temporal_compression = self.vae_model.temporal_compression_decode() or 1
            discard_video_frames = discard_latent_frames * temporal_compression
            
            if images.shape[1] > discard_video_frames:
                print(f"[RayActor {local_rank}] Discarding {discard_video_frames} redundant warmup frames per batch element")
                images = images[:, discard_video_frames:]
            else:
                # This can happen if we are at the very beginning of a 2D batch shard and context was requested?
                # Or for LTX if shard_frames was 1 and discard was 1.
                print(f"[RayActor {local_rank}] WARNING: discard_video_frames ({discard_video_frames}) >= T ({images.shape[1]}).")
            
        return images

    def _handle_output(self, images, shard_index, mmap_path, mmap_shape, output_offset, local_rank, config: ActorConfig):
        """Handles output transport: direct mmap write or CPU serialization."""
        
        # Debug stats (disabled by default — each .item() forces a GPU→CPU sync stall)
        if config.raylight_config.debug.debug_vae:
            stats_min, stats_max, stats_mean = images.min().item(), images.max().item(), images.mean().item()
            print(f"[RayActor {local_rank}] Shard {shard_index} statistics: min={stats_min:.4f}, max={stats_max:.4f}, mean={stats_mean:.4f}")
        
        if mmap_path and mmap_shape:
            # Fast Path: Direct write to shared memory
            print(f"[RayActor {local_rank}] Writing directly to shared mmap: {mmap_path} (offset={output_offset}, shape={images.shape})...")
            
            # Reuse mmap buffer across work-stealing chunks (avoids re-mapping syscall per chunk)
            mmap_shape_t = tuple(mmap_shape) if not isinstance(mmap_shape, tuple) else mmap_shape
            if (self._mmap_cache is not None
                    and self._mmap_cache[0] == mmap_path
                    and self._mmap_cache[1] == mmap_shape_t):
                out_buffer = self._mmap_cache[2]
            else:
                num_elements = 1
                for dim in mmap_shape: num_elements *= dim
                out_buffer = torch.from_file(mmap_path, shared=True, size=num_elements, dtype=torch.float32).reshape(mmap_shape)
                self._mmap_cache = (mmap_path, mmap_shape_t, out_buffer)
            
            # Calculate write length with safety clipping
            shard_frames = images.shape[1]
            
            if len(mmap_shape) == 5:
                # [Batch, T, H, W, C]
                if output_offset + shard_frames > out_buffer.shape[1]:
                    shard_frames = max(0, out_buffer.shape[1] - output_offset)
                    images = images[:, :shard_frames]
                
                if shard_frames > 0:
                    out_buffer[:, output_offset : output_offset + shard_frames].copy_(images.to(torch.float32), non_blocking=True)
            else:
                # [TotalFrames, H, W, C] - Flattened batch (traditional ComfyUI)
                flat_images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                write_len = flat_images.shape[0]
                
                if output_offset + write_len > out_buffer.shape[0]:
                    write_len = max(0, out_buffer.shape[0] - output_offset)
                    flat_images = flat_images[:write_len]
                
                if write_len > 0:
                    out_buffer[output_offset : output_offset + write_len].copy_(flat_images.to(torch.float32), non_blocking=True)
            
            # Sync only the current stream, not the entire device.
            # This unblocks the CPU to return results while other CUDA streams remain live.
            torch.cuda.current_stream(config.device).synchronize()
            
            # Return metadata only
            result_payload = {
                "mmap": True,
                "shape": list(images.shape),
            }
            # Note: out_buffer kept alive in _mmap_cache for chunk reuse
            
        else:
            # Fast Path 2: Ray Object Store Transfer
            # Fallback path if mmap not available
            print(f"[RayActor {local_rank}] Moving to CPU and converting to float16 for transport...")
            
            # Optimization: Cast on GPU first to reduce PCIe bandwidth (FP32 -> FP16 = 50% data)
            images_cpu = images.to(torch.float16).cpu()
            result_payload = images_cpu

        # Clean up original GPU tensor
        del images
        
        print(f"[RayActor {local_rank}] Shard {shard_index} output handling complete.")
        return (shard_index, result_payload)
