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

if TYPE_CHECKING:
    from raylight.distributed_worker.worker_config import WorkerConfig

class SamplerManager:
    """
    Stateless manager for sampling operations.
    Dependencies are injected at method call time via WorkerConfig and arguments.
    """
    def __init__(self):
        pass

    def _handle_fsdp_preparation(self, work_model, config: WorkerConfig, state_dict: Optional[dict] = None) -> bool:
        """
        Handles FSDP-specific model preparation including weight baking.
        Returns True if the model weights were modified (baked).
        """
        model_was_modified = False
        if config.is_fsdp:
            # Propagate state_dict to patcher if not present (crucial for lazy/parallel path)
            if getattr(work_model, "fsdp_state_dict", None) is None:
                if state_dict is not None:
                     print(f"[RayWorker {config.local_rank}] Injecting saved FSDP state dict into model patcher...")
                     work_model.set_fsdp_state_dict(state_dict)

            # CRITICAL FOR FSDP: Bake LoRAs into weights before wrapping!
            # Since 'use_orig_params=True' is not supported in this torch version, FSDP flattens params
            # which breaks standard ComfyUI soft-patching (hooks). We must hard-patch (bake) first.
            if hasattr(work_model, "patches") and work_model.patches:
                 print(f"[RayWorker {config.local_rank}] FSDP: Baking {len(work_model.patches)} patches into weights before sharding...")
                 model_was_modified = True
                 
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
                     print(f"[RayWorker {config.local_rank}] FSDP: Clearing fsdp_state_dict to preserve baked weights.")
                     work_model.set_fsdp_state_dict(None)
                 
                 # Clear patches to prevent double application or tracking issues
                 work_model.patches.clear()
                 if hasattr(work_model, "patches_uuid"):
                     import uuid
                     work_model.patches_uuid = uuid.uuid4()
            
            work_model.patch_fsdp()
            
        return model_was_modified

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
            model_was_modified = self._handle_fsdp_preparation(work_model, config, state_dict)

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if config.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # CRITICAL: Ensure model device references are valid for this worker
        if hasattr(work_model, 'load_device') and work_model.load_device != config.device:
            work_model.load_device = config.device
            
        noise_mask = latent.get("noise_mask", None)
            
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

        return (out,), model_modified
