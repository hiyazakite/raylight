from __future__ import annotations
import torch
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_patcher
from contextlib import contextmanager
from typing import Optional, Tuple, Any, TYPE_CHECKING
from raylight.utils.memory import monitor_memory
from raylight.utils.common import Noise_RandomNoise, patch_ray_tqdm
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
import time

if TYPE_CHECKING:
    from raylight.distributed_worker.worker_config import WorkerConfig


class SamplerManager:
    """
    Stateless manager for sampling operations.
    Dependencies are injected at method call time via WorkerConfig and arguments.
    """
    def __init__(self):
        pass

    # _handle_fsdp_preparation extracted to raylight.comfy_dist.fsdp_utils


    def prepare_for_sampling(
        self, 
        model, 
        config: WorkerConfig, 
        latent: dict, 
        state_dict: Optional[dict] = None
    ) -> Tuple[Any, Any, Any, bool, bool]:
        """Common setup for sampling methods."""
        if model is None:
             raise RuntimeError(f"[RayWorker {config.local_rank}] Model not loaded! Please use a Load node first.")
        
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
        if getattr(work_model, "gguf_metadata", None):
             ops = work_model.model_options.get("custom_operations")
             if ops and hasattr(ops, "Linear"):
                  print(f"[RayWorker {config.local_rank}] SAMPLING START | GGUF Config verified: Dequant={getattr(ops.Linear, 'dequant_dtype', 'None')}, Patch={getattr(ops.Linear, 'patch_dtype', 'None')}")

        return work_model, latent_image, noise_mask, disable_pbar, model_was_modified
    
    @contextmanager
    def sampling_context(
        self, 
        model, 
        config: WorkerConfig,
        latent: dict, 
        state_dict: Optional[dict] = None,
        name: str = "sampler"
    ):
        """Context manager for sampling operations to ensure cleanup."""
        with monitor_memory(f"RayWorker {config.local_rank} - {name}", device=config.device):
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
            yield (work_model, latent_image, noise_mask, disable_pbar, work_latent, model_was_modified)


    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def custom_sampler(
        self,
        model,
        config: WorkerConfig,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
        state_dict: Optional[dict] = None
    ):
        # Returns: (Out Latent, boolean flag if model was modified/baked)
        
        with self.sampling_context(model, config, latent_image, state_dict, name="custom_sampler") as ctx:
             work_model, final_samples, noise_mask, disable_pbar, work_latent, model_modified = ctx
             
             # 2. Noise Generation
             if not add_noise:
                 # OPTIMIZATION: Use zeros_like to avoid CPU->GPU transfer if latent is already on GPU
                 noise = torch.zeros_like(work_latent["samples"])
             else:
                 noise = Noise_RandomNoise(noise_seed).generate_noise(work_latent)

             # 3. Sampling
             # Use utility for consistent memory logging
             if hasattr(model, "mmap_cache"):
                 print(f"[RayWorker {config.local_rank}] Mmap Cache Len: {len(model.mmap_cache) if model.mmap_cache else 0}")
             
             with torch.no_grad():
                 # Correctly initialize CompactFusion step counter
                 try:
                     from raylight.distributed_modules.compact.main import compact_set_step, compact_config, compact_reset
                     config = compact_config()
                     if config is not None and config.enabled:
                         compact_reset() # Ensure cache is clean for new generation
                         compact_set_step(0) # custom_sampler usually implies 0 start unless specified usually, but sigmas handle it. 
                         # Actually Comfy's sample_custom doesn't pass step to callback straightforwardly in all versions, 
                         # but let's assume standard behavior.
                 except (ImportError, NameError, AttributeError):
                     pass

                 except (ImportError, NameError, AttributeError):
                     pass

                 # Stats + Compact Callback
                 last_step_time = time.perf_counter()
                 
                 def sampling_callback(step, x0, x, total_steps):
                    nonlocal last_step_time
                    current_time = time.perf_counter()
                    duration = current_time - last_step_time
                    last_step_time = current_time
                    
                    # Record step time (approximate, excludes callback overhead for next step)
                    # Step timing available via duration variable if needed
                    
                    try:
                        from raylight.distributed_modules.compact.main import compact_set_step, compact_config
                        config = compact_config()
                        if config is not None and config.enabled:
                            compact_set_step(step)
                    except (ImportError, NameError, AttributeError):
                        pass

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
                 out = work_latent.copy()
                 out["samples"] = samples

        return out, model_modified

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def common_ksampler(
        self,
        model,
        config: WorkerConfig,
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
        state_dict: Optional[dict] = None
    ):
        # Returns: ((Out Latent,), boolean flag if model was modified/baked)

        with self.sampling_context(model, config, latent, state_dict, name="common_ksampler") as ctx:
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
                # Correctly initialize CompactFusion step counter
                try:
                    from raylight.distributed_modules.compact.main import compact_set_step, compact_config
                    config = compact_config()
                    if config is not None and config.enabled:
                        compact_set_step(start_step if start_step is not None else 0)
                except (ImportError, NameError, AttributeError):
                    pass
                
                # Stats + Compact Callback
                last_step_time = time.perf_counter()

                def sampling_callback(step, x0, x, total_steps):
                    nonlocal last_step_time
                    current_time = time.perf_counter()
                    duration = current_time - last_step_time
                    last_step_time = current_time
                    # Step timing is now logged inline if needed
                    pass

                    try:
                        from raylight.distributed_modules.compact.main import compact_set_step, compact_config
                        config = compact_config()
                        if config is not None and config.enabled:
                            compact_set_step(step)
                    except (ImportError, NameError, AttributeError):
                        pass
                
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
            out = work_latent.copy()
            out["samples"] = samples

        return (out,), model_modified
