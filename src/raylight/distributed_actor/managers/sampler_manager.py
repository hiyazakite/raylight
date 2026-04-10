from __future__ import annotations
import torch
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_patcher
from contextlib import contextmanager
from typing import Optional, Tuple, Any, TYPE_CHECKING
from raylight.utils.memory import monitor_memory, MemoryPolicy, NULL_POLICY
from raylight.utils.common import Noise_RandomNoise, patch_ray_tqdm
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
import time
import os
from raylight.utils.profiler import CProfileContext


if TYPE_CHECKING:
    from raylight.distributed_actor.actor_config import ActorConfig


class SamplerManager:
    """
    Stateless manager for sampling operations.
    Dependencies are injected at method call time via ActorConfig and arguments.
    """
    def __init__(self):
        pass

    # _handle_fsdp_preparation extracted to raylight.comfy_dist.fsdp_utils


    def prepare_for_sampling(
        self, 
        model, 
        config: ActorConfig, 
        latent: dict, 
        state_dict: Optional[dict] = None
    ) -> Tuple[Any, Any, Any, bool, bool]:
        """Common setup for sampling methods."""
        if model is None:
             raise RuntimeError(f"[RayActor {config.local_rank}] Model not loaded! Please use a Load node first.")
        
        work_model = model
        model_was_modified = False

        latent_image = latent["samples"]
        
        latent_image = comfy.sample.fix_empty_latent_channels(work_model, latent_image)

        if config.is_fsdp:
            from raylight.comfy_dist.fsdp_utils import prepare_fsdp_model_for_sampling
            model_was_modified = prepare_fsdp_model_for_sampling(work_model, config, state_dict)
        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if config.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # CRITICAL: Ensure model device references are valid for this worker
        if hasattr(work_model, 'load_device') and work_model.load_device != config.device:
            work_model.load_device = config.device
            
        noise_mask = latent.get("noise_mask", None)

        # DEBUG: Verify GGUF Ops State
        # if getattr(work_model, "gguf_metadata", None):
        #      ops = work_model.model_options.get("custom_operations")
        #      if ops and hasattr(ops, "Linear"):
        #           print(f"[RayActor {config.local_rank}] SAMPLING START | GGUF Config verified: Dequant={getattr(ops.Linear, 'dequant_dtype', 'None')}, Patch={getattr(ops.Linear, 'patch_dtype', 'None')}")

        return work_model, latent_image, noise_mask, disable_pbar, model_was_modified
    
    @contextmanager
    def sampling_context(
        self, 
        model, 
        config: ActorConfig,
        latent: dict, 
        state_dict: Optional[dict] = None,
        name: str = "sampler",
        memory: MemoryPolicy = NULL_POLICY,
    ):
        """Context manager for sampling operations to ensure cleanup."""
        with monitor_memory(f"RayActor {config.local_rank} - {name}", device=config.device):
            # 1. Common Setup (FSDP, Pbar, Device, Latent Fix)
            # We need a copy of latent because _prepare modifies it
            # The caller passes the original dict, we copy it here
            work_latent = latent.copy()
            
            setup_result = self.prepare_for_sampling(model, config, work_latent, state_dict)
            work_model, latent_image, noise_mask, disable_pbar, model_was_modified = setup_result

            # FIX: Propagate the fixed latent back to work_latent so Noise Gen sees correct shape/dtype
            work_latent["samples"] = latent_image
            
            # We yield the setup results along with the work_latent dict 
            # so the caller can update it with samples
            _needs_reload = False
            try:
                # ── Phase 4: Pre-forward deficit check (aimdo stopgap) ──
                # Estimate activation VRAM: latent × 4 (latent + noise + x + x0)
                try:
                    activation_estimate = latent_image.nbytes * 4
                except (AttributeError, RuntimeError):
                    # NestedTensor and some custom tensor types lack .nbytes
                    try:
                        activation_estimate = latent_image.nelement() * latent_image.element_size() * 4
                    except Exception:
                        activation_estimate = 0
                freed = memory.before_inference(needed_bytes=activation_estimate)
                if freed > 0:
                    _needs_reload = True

                # Guard: block C allocator callback from evicting weights
                # during the active forward pass.  Only empty_cache (safe)
                # can fire from the interceptor while this is set.
                memory._in_forward = True
                yield (work_model, latent_image, noise_mask, disable_pbar, work_latent, model_was_modified)
            finally:
                # Unguard: forward pass is done, C callback can evict again.
                memory._in_forward = False
                # 2. Cleanup after sampling
                import gc
                gc.collect()

                try:
                    from raylight.distributed_modules.attention.backends.fusion.main import compact_reset
                    # Reset all global states (layer indices, cache, instrumentation)
                    compact_reset()
                except Exception:
                    pass

                try:
                    from raylight.distributed_modules.tp_compress import clear_all_tp_residual_caches
                    clear_all_tp_residual_caches()
                except Exception:
                    pass
                finally:
                    # CRITICAL: Always defragment GPU allocator to prevent cross-run
                    # slowdown from fragmented ring attention / compact buffers.
                    torch.cuda.empty_cache()

                # ALSO: Clear GGUF dequantization cache to prevent VRAM accumulation
                try:
                    from raylight.expansion.comfyui_gguf.ops import GGMLLayer
                    GGMLLayer.clear_dequant_cache()
                except Exception:
                    pass

                # Stabilize CPU heap to prevent progressive slowdown from fragmentation
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                # ── Phase 4: Restore evicted params after inference ──
                # Covers both pre-forward evictions (_needs_reload) and
                # mid-forward evictions (interceptor freed weights during
                # per-block forward via layer-aware protected-set guard).
                cache = getattr(memory, '_pinned_cache', None)
                diff_module = getattr(work_model, 'model', None)
                if cache is not None and diff_module is not None and cache._partial_freed:
                    try:
                        cache.reload_evicted(diff_module)
                    except Exception as e:
                        print(f"[SamplerManager] Evicted param reload failed: {e}")


                if config.is_fsdp:
                    pass


    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def custom_sampler(
        self,
        model,
        config: ActorConfig,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
        state_dict: Optional[dict] = None,
        memory: MemoryPolicy = NULL_POLICY,
    ):
        # Returns: ((Out Latent, Denoised Out Latent), boolean flag if model was modified/baked)
        
        with self.sampling_context(model, config, latent_image, state_dict, name="custom_sampler", memory=memory) as ctx:
             work_model, final_samples, noise_mask, disable_pbar, work_latent, model_modified = ctx
             
             # 2. Noise Generation
             import comfy.nested_tensor
             import comfy.sample
             if not add_noise:
                 # OPTIMIZATION: Use zeros_like to avoid CPU->GPU transfer if latent is already on GPU
                 # Handle NestedTensor (e.g. LTXVConcatAVLatent output fed with add_noise=False)
                 samples = work_latent["samples"]
                 if isinstance(samples, comfy.nested_tensor.NestedTensor):
                     noise = comfy.nested_tensor.NestedTensor(tuple(torch.zeros_like(t) for t in samples.unbind()))
                 else:
                     noise = torch.zeros_like(samples)
             else:
                 noise = Noise_RandomNoise(noise_seed).generate_noise(work_latent)

             # 3. Sampling
             # Use utility for consistent memory logging
             # if hasattr(model, "mmap_cache"):
             #     print(f"[RayActor {config.local_rank}] Mmap Cache Len: {len(model.mmap_cache) if model.mmap_cache else 0}")
             
             with torch.no_grad():
                 # Correctly initialize CompactFusion step counter and clear indices
                 try:
                     from raylight.distributed_modules.attention.backends.fusion.main import compact_set_step, compact_reset
                     # Essential: Always reset state before sampling to clear indices/state from previous runs
                     compact_reset()
                     compact_set_step(0) 
                 except Exception:
                     pass

                 try:
                     from raylight.distributed_modules.tp_compress import clear_all_tp_residual_caches
                     clear_all_tp_residual_caches()
                 except Exception:
                     pass



                 # Stats + Compact Callback
                 last_step_time = time.perf_counter()
                 last_denoised = None

                 def sampling_callback(step, x0, x, total_steps):
                    nonlocal last_step_time, last_denoised
                    current_time = time.perf_counter()

                    last_step_time = current_time
                    
                    # Record step time (approximate, excludes callback overhead for next step)
                    # Step timing available via duration variable if needed
                    
                    try:
                        from raylight.distributed_modules.attention.backends.fusion.main import compact_set_step
                        # Always set step to reset the global ATTN_LAYER_IDX counter
                        compact_set_step(step)
                    except (ImportError, NameError, AttributeError):
                        pass

                    # Keep the final x0 estimate so nodes can expose denoised_output.
                    if x0 is not None and (step + 1) >= total_steps:
                        last_denoised = x0.detach().clone()
                        
                 profile_enabled = config.raylight_config.debug.profile_sampler
                 with CProfileContext(enabled=profile_enabled, sort_by='cumulative', top_k=5, name="custom_ksampler (Comfy)"):
                     samples = comfy.sample.sample_custom(
                             work_model,
                             noise,
                             cfg,
                             sampler,
                             sigmas,
                             positive,
                             negative,
                          final_samples,
                          noise_mask=noise_mask,
                          disable_pbar=disable_pbar,
                          seed=noise_seed,
                          callback=sampling_callback,
                      )
                 samples_cpu = samples.to("cpu")
                 denoised_cpu = last_denoised.to("cpu") if last_denoised is not None else samples_cpu

                 out = work_latent.copy()
                 out["samples"] = samples_cpu
                 denoised_out = work_latent.copy()
                 denoised_out["samples"] = denoised_cpu
                 del samples # Drop GPU reference immediately

        return (out, denoised_out), model_modified

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def common_ksampler(
        self,
        model,
        config: ActorConfig,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        sigmas=None,
        state_dict: Optional[dict] = None,
        memory: MemoryPolicy = NULL_POLICY,
    ):
        # Returns: ((Out Latent, Denoised Out Latent), boolean flag if model was modified/baked)

        with self.sampling_context(model, config, latent, state_dict, name="common_ksampler", memory=memory) as ctx:
            work_model, final_samples, noise_mask, disable_pbar, work_latent, model_modified = ctx

            # 2. Noise Generation
            if disable_noise:
                noise = torch.zeros(
                    final_samples.size(),
                    dtype=final_samples.dtype,
                    layout=final_samples.layout,
                    device="cpu",
                )
            else:
                batch_inds = work_latent.get("batch_index", None)
                noise = comfy.sample.prepare_noise(
                    final_samples, seed, batch_inds
                )

            # 3. Sampler resolution logic for custom sigmas
            sampler_obj = sampler_name
            if sigmas is not None:
                 if isinstance(sampler_name, str):
                     sampler_obj = comfy.samplers.ksampler(sampler_name)

            # 4. Sampling
            with torch.no_grad():
                # Correctly initialize CompactFusion step counter and clear indices
                try:
                    from raylight.distributed_modules.attention.backends.fusion.main import compact_set_step, compact_reset
                    # Essential: Always reset state before sampling to clear indices/state from previous runs
                    compact_reset()
                    compact_set_step(start_step if start_step is not None else 0)
                except Exception:
                    pass

                try:
                    from raylight.distributed_modules.tp_compress import clear_all_tp_residual_caches
                    clear_all_tp_residual_caches()
                except Exception:
                    pass
                
                # Stats + Compact Callback
                last_step_time = time.perf_counter()
                last_denoised = None

                def sampling_callback(step, x0, x, total_steps):
                    nonlocal last_step_time, last_denoised
                    current_time = time.perf_counter()

                    last_step_time = current_time
                    # Step timing is now logged inline if needed
                    pass

                    try:
                        from raylight.distributed_modules.attention.backends.fusion.main import compact_set_step
                        # Always set step to reset the global ATTN_LAYER_IDX counter
                        compact_set_step(step)
                    except (ImportError, NameError, AttributeError):
                        pass

                    # Keep the final x0 estimate so nodes can expose denoised_output.
                    if x0 is not None and (step + 1) >= total_steps:
                        last_denoised = x0.detach().clone()

                profile_enabled = config.raylight_config.debug.profile_sampler
                with CProfileContext(enabled=profile_enabled, sort_by='cumulative', top_k=5, name="common_ksampler (Comfy)"):
                    if sigmas is None:
                        samples = comfy.sample.sample(
                            work_model,
                            noise,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            positive,
                            negative,
                            final_samples,
                            denoise=denoise,
                            disable_noise=disable_noise,
                            start_step=start_step,
                            last_step=last_step,
                            force_full_denoise=force_full_denoise,
                            noise_mask=noise_mask,
                            disable_pbar=disable_pbar,
                            seed=seed,
                            callback=sampling_callback,
                       )
                    else:
                             samples = comfy.sample.sample_custom(
                                work_model,
                                noise,
                                cfg,
                                sampler_obj,
                                sigmas,
                                positive,
                                negative,
                                final_samples,
                                noise_mask=noise_mask,
                                disable_pbar=disable_pbar,
                                seed=seed,
                                callback=sampling_callback,
                            )
                
            samples_cpu = samples.to("cpu")
            denoised_cpu = last_denoised.to("cpu") if last_denoised is not None else samples_cpu

            out = work_latent.copy()
            out["samples"] = samples_cpu
            denoised_out = work_latent.copy()
            denoised_out["samples"] = denoised_cpu
            del samples # Drop GPU reference immediately

        return (out, denoised_out), model_modified
