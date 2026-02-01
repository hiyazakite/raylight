import torch
from raylight.utils.memory import monitor_memory
from raylight.utils.common import cleanup_memory, force_malloc_trim

from typing import Optional, Any
from raylight.distributed_worker.worker_config import WorkerConfig

class VaeManager:
    def __init__(self):
        self.vae_model = None

    def load_vae(
        self, 
        vae_path: str, 
        config: WorkerConfig, 
        state_cache: Any
    ) -> Optional[Any]:
        from raylight.distributed_worker.model_context import get_context, ModelState
        from raylight.comfy_dist.sd import decode_tiled_1d, decode_tiled_, decode_tiled_3d
        import types

        print(f"[RayWorker {config.local_rank}] VAE Load via ModelContext: {vae_path}")
        
        # 1. Get Unified Context (VAEContext for VAE models)
        ctx = get_context(vae_path, config, model_type="vae")
        
        # 2. Create Load State
        state = ModelState(
            unet_path=vae_path,
            model_options={},
        )
        
        # 3. Unified Load (Handles Caching & Instantiation)
        vae_model, _ = ctx.load(state, config, state_cache)
        
        # 5. Monkey patch decode optimizations
        vae_model.decode_tiled_1d = types.MethodType(decode_tiled_1d, vae_model)
        vae_model.decode_tiled_ = types.MethodType(decode_tiled_, vae_model)
        vae_model.decode_tiled_3d = types.MethodType(decode_tiled_3d, vae_model)

        # 6. Stream weights to GPU (handled by context, same pattern as UNET)
        from raylight.distributed_worker.model_context import VAEContext
        assert isinstance(ctx, VAEContext)
        ctx.stream_vae_to_device(vae_model, config.device)

        self.vae_model = vae_model
        
        # Cleanup
        cleanup_memory()
        
        return vae_model


    def release_vae(self, local_rank: int):
        """Explicitly release VAE model from memory to free RAM for other operations."""
        if self.vae_model is not None:
            print(f"[RayWorker {local_rank}] Releasing VAE model from RAM...")
            del self.vae_model
            self.vae_model = None
            
            # Aggressive cleanup
            cleanup_memory()
            
            # Force OS to reclaim freed memory
            force_malloc_trim()
            
            print(f"[RayWorker {local_rank}] VAE released.")
        return True

    def get_temporal_compression(self):
        if self.vae_model is None:
            return None
        return self.vae_model.temporal_compression_decode()

    def get_spatial_compression(self):
        if self.vae_model is None:
            return None
        return self.vae_model.spacial_compression_decode()

    def check_health(self, samples, shard_index, config: WorkerConfig, overwrite_cast_dtype=None):
        # Diagnostic: Check input latent statistics
        l_min, l_max, l_mean = samples["samples"].min().item(), samples["samples"].max().item(), samples["samples"].mean().item()
        print(f"[RayWorker {config.local_rank}] Input Latent Stats (shard {shard_index}): min={l_min:.4f}, max={l_max:.4f}, mean={l_mean:.4f}")
        if torch.isnan(samples["samples"]).any():
            print(f"[RayWorker {config.local_rank}] CRITICAL: Input latents for shard {shard_index} contain NaNs!")
        
        # Check if overwrite_cast_dtype is set (from patch_temp_fix_ck_ops)
        if overwrite_cast_dtype is not None:
             print(f"[RayWorker {config.local_rank}] Debug: overwrite_cast_dtype is set to {overwrite_cast_dtype}")
        
        # Check VAE weights for NaNs BEFORE decoding
        if self.vae_model is not None:
            nan_params = []
            for name, p in self.vae_model.first_stage_model.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(name)
            if nan_params:
                print(f"[RayWorker {config.local_rank}] CRITICAL: VAE parameters contain NaNs: {nan_params[:5]}...")
            else:
                print(f"[RayWorker {config.local_rank}] VAE parameters verified healthy (no NaNs).")

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
        config: Optional[WorkerConfig] = None,
        lora_manager: Optional[Any] = None,
        state_cache: Optional[Any] = None
    ):
        if config is None:
             raise ValueError("WorkerConfig must be passed to decode")

        with monitor_memory(f"RayWorker {config.local_rank} - ray_vae_decode", device=config.device):
            print(f"[RayWorker {config.local_rank}] Entering ray_vae_decode (direct method call) for shard {shard_index}...")
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
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
            self.vae_model.first_stage_model.to(dtype)
            self.vae_model.vae_dtype = dtype
            
            # 2. Transfer Latents (Optimized)
            latents_to_decode = samples["samples"].to(config.device, dtype=dtype)
            print(f"[RayWorker {config.local_rank}] latents_to_decode shape: {latents_to_decode.shape}")
            print(f"[RayWorker {config.local_rank}] tiling: tile_t={temporal_size}, overlap_t={temporal_overlap}")
            
            # DEBUG: Verify decoder is on correct device before decode
            for name, param in self.vae_model.first_stage_model.named_parameters():
                if 'decoder' in name:
                    print(f"[RayWorker {config.local_rank}] DECODER PARAM CHECK: {name} device={param.device}, mean={param.float().mean().item():.6f}")
                    break
            print(f"[RayWorker {config.local_rank}] VAE self.device={self.vae_model.device}, latents device={latents_to_decode.device}")
    
            # 3. Decode
            images = self.vae_model.decode_tiled(
                latents_to_decode,
                tile_x=tile_size // compression,
                tile_y=tile_size // compression,
                overlap=overlap // compression,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )
            print(f"[RayWorker {config.local_rank}] decode_tiled complete. Shape: {images.shape}")
            
            # 4. Post-Process (Shape Fix + Warmup Discard)
            images = self._post_process_images(images, discard_latent_frames, config.local_rank, shard_index)
            
            # 5. Handle Output (Mmap or CPU Transfer)
            # RETURNS: (shard_index, result_payload)
            result = self._handle_output(images, shard_index, mmap_path, mmap_shape, output_offset, config.local_rank)
    
            # Proactive cleanup (images tensor is referenced in result if not mmap, but original is gone)
            del latents_to_decode
            
            return result
        
        finally:
            # Release VAE from GPU VRAM using context's offload method
            # (Consistent with diffusion model pattern)
            # Release VAE from GPU VRAM using context's offload method
            # (Consistent with diffusion model pattern)
            
            # Use factory to get correct context type (VAE defaults to standard VAEContext)
            # We don't have the original path easily here if not stored, 
            # BUT VAEs are almost always standard unless we support GGUF VAEs (rare).
            # For now, we reconstruct standard VAEContext or rely on what load_vae used.
            # Strategy: Default to VAEContext since we don't persist 'ctx' in manager.
            
            from raylight.distributed_worker.model_context import VAEContext
            vae_ctx = VAEContext(use_mmap=config.parallel_dict.get("use_mmap", True))
            vae_ctx.offload(self.vae_model, lora_manager, {}, config)
            
            print(f"[RayWorker {config.local_rank}] VAE Offload in finally block complete.")
            
            # Force OS to reclaim freed memory (fixes RSS creep on Worker 0)
            force_malloc_trim()

    def _select_dtype(self, preference, local_rank):
        """Selects appropriate VAE dtype based on hardware and preference."""
        if preference == "auto":
            if torch.cuda.is_bf16_supported():
                print(f"[RayWorker {local_rank}] Using bfloat16 VAE (auto: bf16 supported)")
                return torch.bfloat16
            else:
                print(f"[RayWorker {local_rank}] Using float32 VAE (auto: bf16 not supported)")
                return torch.float32
        elif preference == "bf16":
            print(f"[RayWorker {local_rank}] Using bfloat16 VAE (user selected)")
            return torch.bfloat16
        elif preference == "fp16":
            print(f"[RayWorker {local_rank}] Using float16 VAE (user selected - may cause NaN on some models)")
            return torch.float16
        else:  # fp32
            print(f"[RayWorker {local_rank}] Using float32 VAE (user selected - stable but 2x memory)")
            return torch.float32

    def _post_process_images(self, images, discard_latent_frames, local_rank, shard_index=None):
        """Normalizes image shape and handles warmup frame discarding."""
        # Check raw output stats
        raw_min, raw_max, raw_mean = images.min().item(), images.max().item(), images.mean().item()
        print(f"[RayWorker {local_rank}] RAW decode stats: min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}")
        
        if torch.isnan(images).any():
            print(f"[RayWorker {local_rank}] CRITICAL: VAE output STILL contains NaNs even in float32!")
        
        # Shape normalization
        if len(images.shape) == 5:
            # [B, T, H, W, C] -> [B*T, H, W, C]
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        elif len(images.shape) == 4:
            # Already [T, H, W, C]
            pass
        else:
            # Fallback for unexpected shapes
            while len(images.shape) > 4 and images.shape[0] == 1:
                images = images.squeeze(0)
            if len(images.shape) == 3: # H, W, C -> 1, H, W, C
                images = images.unsqueeze(0)
            
        # Discard warmup frames
        if discard_latent_frames > 0 and self.vae_model is not None:
            temporal_compression = self.vae_model.temporal_compression_decode() or 8
            discard_video_frames = discard_latent_frames * temporal_compression
            print(f"[RayWorker {local_rank}] Discarding {discard_video_frames} redundant warmup frames")
            images = images[discard_video_frames:]
            
        return images

    def _handle_output(self, images, shard_index, mmap_path, mmap_shape, output_offset, local_rank):
        """Handles output transport: direct mmap write or CPU serialization."""
        
        # Stats for troubleshooting "Black Video" issues
        stats_min, stats_max, stats_mean = images.min().item(), images.max().item(), images.mean().item()
        print(f"[RayWorker {local_rank}] Shard {shard_index} statistics: min={stats_min:.4f}, max={stats_max:.4f}, mean={stats_mean:.4f}")
        
        if mmap_path and mmap_shape:
            # Fast Path: Direct write to shared memory
            print(f"[RayWorker {local_rank}] Writing directly to shared mmap: {mmap_path} (offset={output_offset})...")
            
            # Calculate total elements
            num_elements = 1
            for dim in mmap_shape: num_elements *= dim
            
            # Map the shared file (assumed float32 output)
            out_buffer = torch.from_file(mmap_path, shared=True, size=num_elements, dtype=torch.float32).reshape(mmap_shape)
            
            # Calculate write length with safety clipping
            write_len = images.shape[0]
            if output_offset + write_len > out_buffer.shape[0]:
                 write_len = out_buffer.shape[0] - output_offset
                 images = images[:write_len]

            # Write data
            # Optimization: implicit cast via copy_ avoids VRAM spike
            out_buffer[output_offset : output_offset + write_len].copy_(images, non_blocking=True)
            torch.cuda.synchronize() # Ensure copy completes before we return/cleanup
            
            # Return metadata only
            result_payload = {
                "mmap": True,
                "shape": images.shape,
                "stats": (stats_min, stats_max, stats_mean)
            }
            # Clean up mmap handle
            del out_buffer
            
        else:
            # Fast Path 2: Ray Object Store Transfer
            # Fallback path if mmap not available
            print(f"[RayWorker {local_rank}] Moving to CPU and converting to float16 for transport...")
            
            # Optimization: Cast on GPU first to reduce PCIe bandwidth (FP32 -> FP16 = 50% data)
            images_cpu = images.to(torch.float16).cpu()
            result_payload = images_cpu

        # Clean up original GPU tensor
        del images
        
        print(f"[RayWorker {local_rank}] Shard {shard_index} output handling complete.")
        return (shard_index, result_payload)
