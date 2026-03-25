from __future__ import annotations

import gc
from typing import Any

import ray
import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils

from raylight.comfy_dist.utils import cancellable_get


class RayIDLoraKSampler:
    """Distributed ID-LoRA sampler for LTXAV.

    Accepts a combined AV latent from LTXVConcatAVLatent plus standard
    positive/negative CONDITIONING.  All patchification, noise scheduling, and
    audio/video separation happen inside the workers via model.apply_model —
    no ltx_core dependency required.

    Output LATENT is a NestedTensor compatible with LTXVSeparateAVLatent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray cluster with LTXAV model loaded"}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "audio_guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "identity_guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sigmas": ("SIGMAS", {"tooltip": "Sigmas schedule from LTXVScheduler"}),
                "latent_image": ("LATENT", {"tooltip": "Combined AV latent from LTXVConcatAVLatent"}),
            },
            "optional": {
                "reference_audio_latent": ("LATENT", {"tooltip": "Encoded reference audio from LTXVAudioVAEEncode — speaker identity"}),
                "stg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "stg_blocks": ("STRING", {"default": "0,1,2,3"}),
                "av_bimodal_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "sample"
    CATEGORY = "Raylight"

    def sample(
        self,
        ray_actors,
        add_noise,
        noise_seed,
        positive,
        negative,
        video_guidance_scale,
        audio_guidance_scale,
        identity_guidance_scale,
        sigmas,
        latent_image,
        reference_audio_latent=None,
        stg_scale=0.0,
        stg_blocks="",
        av_bimodal_scale=0.0,
    ):
        from raylight.distributed_worker.managers.idlora_manager import IDLoraDenoiseConfig
        from raylight.comfy_dist.utils import clear_ray_cluster

        try:
            # Free main-process VRAM so workers can access pinned caches
            gc.collect()
            from raylight.nodes import _orig_unload_all_models
            _orig_unload_all_models()
            comfy.model_management.soft_empty_cache()
            comfy.model_management.soft_empty_cache()

            gpu_actors = ray_actors["workers"]
            dtype = torch.bfloat16

            ray.get([actor.reload_model_if_needed.remote() for actor in gpu_actors])

            # ----------------------------------------------------------------
            # 1. Split the combined AV NestedTensor
            # ----------------------------------------------------------------
            av_latents = latent_image["samples"].unbind()
            video_samples = av_latents[0]  # (B, C, F, H, W)
            audio_samples = av_latents[1]  # (B, C, T, freq)

            v_mask = None
            a_mask = None
            av_masks = None
            if "noise_mask" in latent_image and latent_image["noise_mask"] is not None:
                av_masks = latent_image["noise_mask"].unbind()
                v_mask = av_masks[0].cpu()  # (B, 1, F, H, W)
                a_mask = av_masks[1].cpu()

            # ----------------------------------------------------------------
            # 2. Add noise (flow matching: x_t = (1-σ)·x_0 + σ·noise)
            #    Where mask=0 (conditioned, e.g. first frame), keep clean.
            #    Where mask=1, interpolate toward noise.
            #    Reference: GaussianNoiser in ltx_core uses
            #      x_t = noise * (mask * σ) + x_0 * (1 - mask * σ)
            # ----------------------------------------------------------------
            sigmas_cpu = sigmas.float().cpu()
            sigma_max = sigmas_cpu[0].item()

            gen = torch.Generator()
            gen.manual_seed(noise_seed)

            if add_noise and sigma_max > 0:
                v_clean = video_samples.to(dtype=dtype, device="cpu")
                v_noise = torch.randn(v_clean.shape, generator=gen, dtype=dtype)
                v_mask_eff = v_mask.to(dtype=dtype) if v_mask is not None else torch.ones(1, 1, 1, 1, 1, dtype=dtype)
                v_scaled_mask = v_mask_eff * sigma_max
                v_f = torch.lerp(v_clean, v_noise, v_scaled_mask)
                del v_noise, v_mask_eff, v_scaled_mask

                a_clean = audio_samples.to(dtype=dtype, device="cpu")
                a_noise = torch.randn(a_clean.shape, generator=gen, dtype=dtype)
                a_mask_eff = a_mask.to(dtype=dtype) if a_mask is not None else torch.ones(1, 1, 1, 1, dtype=dtype)
                a_scaled_mask = a_mask_eff * sigma_max
                a_f = torch.lerp(a_clean, a_noise, a_scaled_mask)
                del a_noise, a_mask_eff, a_scaled_mask
            else:
                v_clean = video_samples.to(dtype=dtype, device="cpu")
                a_clean = audio_samples.to(dtype=dtype, device="cpu")
                v_f = v_clean.clone()
                a_f = a_clean.clone()

            # ----------------------------------------------------------------
            # 3. Prepare reference audio tokens for in-context conditioning
            # Following ComfyUI's native ID-LoRA implementation: reshape
            # (B, C, T, freq) → (B, T, C*freq) and pass as ref_audio kwarg.
            # The model injects tokens in _process_input, sets their timestep
            # to 0, and trims them from the output automatically.
            # ----------------------------------------------------------------
            ref_audio = None
            if reference_audio_latent is not None:
                ref_a = reference_audio_latent["samples"].to(dtype=dtype, device="cpu")
                b, c, t, f = ref_a.shape
                ref_tokens = ref_a.permute(0, 2, 1, 3).reshape(b, t, c * f)
                ref_audio = {"tokens": ref_tokens}

            # ----------------------------------------------------------------
            # 4. Pack tensors for Ray transport
            # ----------------------------------------------------------------
            def _to_dict(samples, mask):
                d: dict = {"samples": samples}
                if mask is not None:
                    d["noise_mask"] = mask
                return d

            v_mask_send = v_mask.to(dtype=dtype) if v_mask is not None else None
            video_ref = ray.put(_to_dict(v_f, v_mask_send))
            audio_ref = ray.put(_to_dict(a_f, a_mask))
            sigmas_ref = ray.put(sigmas_cpu)
            ref_audio_ref = ray.put(ref_audio)  # None or {"tokens": (B, T_ref, C*freq)}

            # Clean latents for post_process_latent blending
            v_clean_ref = ray.put(v_clean)
            a_clean_ref = ray.put(a_clean)

            # Eagerly free huge CPU tensors in main process now that Ray Object Store has them
            del v_f, a_f, v_clean, a_clean, v_mask_send, v_mask, a_mask, sigmas_cpu, ref_audio
            del video_samples, audio_samples, av_latents
            if "noise_mask" in latent_image and latent_image["noise_mask"] is not None:
                del av_masks
            gc.collect()

            stg_block_idx = None
            if stg_scale > 0.0 and stg_blocks:
                stg_block_idx = [int(x.strip()) for x in stg_blocks.split(",") if x.strip().isdigit()]

            denoise_config = IDLoraDenoiseConfig(
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                identity_guidance_scale=identity_guidance_scale,
                stg_scale=stg_scale,
                stg_block_idx=stg_block_idx,
                av_bimodal_scale=av_bimodal_scale,
            )
            config_ref = ray.put(denoise_config)

            lora_config_hash = ray_actors.get("lora_config_hash")
            if lora_config_hash is not None:
                ray.get([actor.reapply_loras_for_config.remote(lora_config_hash) for actor in gpu_actors])

            futures = [
                actor.idlora_sample.remote(
                    video_ref, audio_ref, positive, negative,
                    sigmas_ref, ref_audio=ref_audio_ref, denoise_config=config_ref,
                    v_clean=v_clean_ref, a_clean=a_clean_ref,
                )
                for actor in gpu_actors
            ]

            results = cancellable_get(futures)
            video_out, audio_out = results[0]  # both on CPU

            output = {"samples": comfy.nested_tensor.NestedTensor((video_out, audio_out))}

            gc.collect()
            torch.cuda.empty_cache()

            return (output,)

        except Exception as e:
            clear_ray_cluster(ray_actors, reason=f"sampling error in RayIDLoraKSampler: {type(e).__name__}")
            raise
