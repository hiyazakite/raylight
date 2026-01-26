import torch
import comfy.sample
import comfy.utils
import comfy.samplers
import comfy.model_patcher
from contextlib import contextmanager
from raylight.utils.memory import monitor_memory
from raylight.utils.common import Noise_EmptyNoise, Noise_RandomNoise, patch_ray_tqdm, cleanup_memory
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops

class SamplerManager:
    def __init__(self, worker):
        self.worker = worker

    def _prepare_for_sampling(self, latent):
        """Common setup for sampling methods."""
        if self.worker.model is None:
             raise RuntimeError(f"[RayWorker {self.worker.local_rank}] Model not loaded! Please use a Load node first.")
        work_model = self.worker.model

        latent_image = latent["samples"]
        # Handle dict copy in caller if needed, or here if we want to be safe
        # common_ksampler makes a copy of latent later (out = latent.copy())
        # custom_sampler makes a copy of latent early. 
        # For now, we return modified latent_image.

        latent_image = comfy.sample.fix_empty_latent_channels(work_model, latent_image)

        if self.worker.parallel_dict["is_fsdp"] is True:
            # Propagate state_dict to patcher if not present (crucial for lazy/parallel path)
            if getattr(work_model, "fsdp_state_dict", None) is None:
                if hasattr(self.worker, "state_dict") and self.worker.state_dict is not None:
                     print(f"[RayWorker {self.worker.local_rank}] Injecting saved FSDP state dict into model patcher...")
                     work_model.set_fsdp_state_dict(self.worker.state_dict)

            # NOTE: Logic to force model to CPU before FSDP was removed here (Legacy)
            # We now handle device placement explicitly in the FSDP shard/load functions.
            
            # CRITICAL FOR FSDP: Bake LoRAs into weights before wrapping!
            # Since 'use_orig_params=True' is not supported in this torch version, FSDP flattens params
            # which breaks standard ComfyUI soft-patching (hooks). We must hard-patch (bake) first.
            if hasattr(work_model, "patches") and work_model.patches:
                 print(f"[RayWorker {self.worker.local_rank}] FSDP: Baking {len(work_model.patches)} patches into weights before sharding...")
                 
                 # Force in-place update so the underlying model instance (which FSDP wraps) is modified
                 prev_inplace = work_model.weight_inplace_update
                 work_model.weight_inplace_update = True
                 
                 # force_patch_weights=True permanently modifies the weights in work_model.model
                 work_model.load(device_to="cpu", force_patch_weights=True)
                 
                 # Free memory: We don't need backups of the unbaked weights since we are committing to them
                 if hasattr(work_model, "backup"):
                     work_model.backup.clear()

                 # Restore inplace flag
                 work_model.weight_inplace_update = prev_inplace
                 
                 # CRITICAL: Prevent FSDP wrapper from reloading stale state_dict over our baked weights
                 if hasattr(work_model, "set_fsdp_state_dict"):
                     print(f"[RayWorker {self.worker.local_rank}] FSDP: Clearing fsdp_state_dict to preserve baked weights.")
                     work_model.set_fsdp_state_dict(None)
                 
                 # Clear patches to prevent double application or tracking issues
                 work_model.patches.clear()
                 if hasattr(work_model, "patches_uuid"):
                     import uuid
                     work_model.patches_uuid = uuid.uuid4()
            
            work_model.patch_fsdp()
            # FSDP cleanup
            # if hasattr(self.worker, "state_dict") and self.worker.state_dict is not None:
            #     del self.worker.state_dict
            #     self.worker.state_dict = None
            #     cleanup_memory()

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if self.worker.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # CRITICAL: Ensure model device references are valid for this worker
        if hasattr(work_model, 'load_device') and work_model.load_device != self.worker.device:
            work_model.load_device = self.worker.device
            
        noise_mask = latent.get("noise_mask", None)
            
        return work_model, latent_image, noise_mask, disable_pbar
    
    @contextmanager
    def sampling_context(self, latent):
        """Context manager for sampling operations to ensure cleanup."""
        try:
            # 1. Common Setup (FSDP, Pbar, Device, Latent Fix)
            # We need a copy of latent because _prepare modifies it
            # The caller passes the original dict, we copy it here
            work_latent = latent.copy()
            
            setup_result = self._prepare_for_sampling(work_latent)
            # setup_result is (work_model, latent_image, noise_mask, disable_pbar)
            
            # We yield the setup results along with the work_latent dict 
            # so the caller can update it with samples
            yield setup_result + (work_latent,)
            
        finally:
            # Always clean up memory, even if sampling fails
            cleanup_memory()

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def custom_sampler(
        self,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
    ):
        # NOTE: Reload is now handled by coordinator (sampler node) BEFORE dispatching.

        with self.sampling_context(latent_image) as (work_model, final_samples, noise_mask, disable_pbar, work_latent):
            # 2. Noise Generation
            if not add_noise:
                noise = Noise_EmptyNoise().generate_noise(work_latent)
            else:
                noise = Noise_RandomNoise(noise_seed).generate_noise(work_latent)

            # 3. Sampling
            # Use utility for consistent memory logging
            with monitor_memory(f"RayWorker {self.worker.local_rank} - custom_sampler", device=self.worker.device):
                if hasattr(self.worker.model, "mmap_cache"):
                     print(f"[RayWorker {self.worker.local_rank}] Mmap Cache Len: {len(self.worker.model.mmap_cache) if self.worker.model.mmap_cache else 0}")
                
                with torch.no_grad():
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
                )
                out = work_latent.copy()
                out["samples"] = samples

        return out

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def common_ksampler(
        self,
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
    ):
        with monitor_memory(f"RayWorker {self.worker.local_rank} - common_ksampler", device=self.worker.device):
            # NOTE: Reload is now handled by coordinator (sampler node) BEFORE dispatching.
            
            with self.sampling_context(latent) as (work_model, final_samples, noise_mask, disable_pbar, work_latent):
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
                        )
                    
                    out = work_latent.copy()
                    out["samples"] = samples

        return (out,)
