from __future__ import annotations

import copy
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import comfy.model_patcher
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils

from raylight.utils.memory import monitor_memory

if TYPE_CHECKING:
    from raylight.distributed_worker.worker_config import WorkerConfig


@dataclass(frozen=True)
class IDLoraDenoiseConfig:
    """Serializable config for the worker-side ID-LoRA denoising loop."""

    video_guidance_scale: float = 5.0
    audio_guidance_scale: float = 3.0
    identity_guidance_scale: float = 1.0
    stg_scale: float = 0.0
    av_bimodal_scale: float = 0.0
    stg_block_idx: Optional[list] = None


class IDLoraManager:
    """Worker-side multi-pass denoising loop for ID-LoRA on LTXAV.

    Uses model.model.apply_model() with the SAME kwargs that the standard
    ComfyUI sampling infrastructure (``comfy.sample.sample_custom`` →
    ``CFGGuider`` → ``_calc_cond_batch``) would produce.

    Key kwargs that MUST be present (matching LTXAV.extra_conds output):
      - ``denoise_mask``       — video denoise mask (B,1,F,H,W) full spatial
      - ``audio_denoise_mask`` — audio denoise mask (B,C,T,freq)
      - ``frame_rate``         — float
      - ``latent_shapes``      — list of shapes for pack/unpack
      - ``transformer_options``— dict with cond_or_uncond, sigmas, sample_sigmas

    Patchification, timestep embedding, and unpatchification all happen
    inside apply_model via ComfyUI's native LTXAV implementation.
    """

    def __init__(self):
        pass

    def _prepare_model(self, model, config: "WorkerConfig", state_dict=None):
        if model is None:
            raise RuntimeError(
                f"[RayWorker {config.local_rank}] Model not loaded! Use a Load node first."
            )
        if config.is_fsdp:
            from raylight.comfy_dist.fsdp_utils import prepare_fsdp_model_for_sampling
            prepare_fsdp_model_for_sampling(model, config, state_dict)
        if hasattr(model, "load_device") and model.load_device != config.device:
            model.load_device = config.device
        return model

    @contextmanager
    def sampling_context(self, model, config: "WorkerConfig", state_dict=None, name="idlora_sampler"):
        with monitor_memory(f"RayWorker {config.local_rank} - {name}", device=config.device):
            self._prepare_model(model, config, state_dict)
            try:
                model.pre_run()
                yield model
            finally:
                try:
                    model.cleanup()
                except Exception:
                    pass

                try:
                    if hasattr(model, "clean_hooks"): # Prevent hook bloat on repeated generations
                        model.clean_hooks()
                except Exception:
                    pass

                # Reset CompactFusion state
                try:
                    from raylight.distributed_modules.attention.backends.fusion.main import compact_reset
                    compact_reset()
                except Exception:
                    pass

                gc.collect()
                torch.cuda.empty_cache()

                # Clear GGUF dequantization cache
                try:
                    from raylight.expansion.comfyui_gguf.ops import GGMLLayer
                    GGMLLayer.clear_dequant_cache()
                except Exception:
                    pass

                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

    def idlora_denoise(
        self,
        model,
        config: "WorkerConfig",
        video_dict: dict,
        audio_dict: dict,
        positive,
        negative,
        sigmas: torch.Tensor,
        ref_audio: Optional[dict] = None,
        denoise_config: Optional[IDLoraDenoiseConfig] = None,
        v_clean: Optional[torch.Tensor] = None,
        a_clean: Optional[torch.Tensor] = None,
    ):
        """Run the ID-LoRA Euler denoising loop on the worker.

        The model is called via ``model.model.apply_model()`` with kwargs
        matching what ComfyUI's ``_calc_cond_batch`` → ``LTXAV.extra_conds``
        pipeline would produce.  This ensures ``process_timestep`` receives
        both ``denoise_mask`` AND ``audio_denoise_mask`` for correct per-patch
        timestep computation.

        Args:
            model: ComfyUI ModelPatcher (LTXAV).
            config: Worker config (device, rank, etc.).
            video_dict: {"samples": (B,C,F,H,W), "noise_mask": optional}.
            audio_dict: {"samples": (B,C,T,freq), "noise_mask": optional}.
            positive/negative: Standard ComfyUI CONDITIONING lists.
            sigmas: 1-D float tensor of noise levels (decreasing, ends at 0).
            ref_audio: Optional dict {"tokens": (B, T_ref, C*freq)}.  Passed
                as ``ref_audio`` to apply_model; the model (via idlora_patches)
                injects tokens inside _process_input with negative temporal
                positions, sets timestep=0, and trims from output.  None = no
                reference audio.
            denoise_config: Guidance scales etc.
            v_clean: Clean video latent (B,C,F,H,W) for post_process blending.
            a_clean: Clean audio latent (B,C,T,freq) for post_process blending.

        Returns:
            (video_out_cpu, audio_out_cpu) — (B,C,F,H,W) and (B,C,T,freq) on CPU.
        """
        if denoise_config is None:
            denoise_config = IDLoraDenoiseConfig()

        # Apply ID-LoRA patches to LTXAVModel (idempotent, skips if PR merged)
        from raylight.comfy_extra_dist.idlora_patches import ensure_patches_applied
        ensure_patches_applied()

        with self.sampling_context(model, config):
            device = config.device
            dtype = torch.bfloat16

            # ------------------------------------------------------------------
            # Initialize CompactFusion state (matching SamplerManager)
            # ------------------------------------------------------------------
            try:
                from raylight.distributed_modules.attention.backends.fusion.main import compact_set_step, compact_reset
                compact_reset()
                compact_set_step(0)
            except Exception:
                pass

            if config.local_rank == 0:
                pos_keys = sorted(positive[0][1].keys())
                neg_keys = sorted(negative[0][1].keys())
                print(f"[RayWorker 0] ID-LoRA cond keys | pos={pos_keys} neg={neg_keys}", flush=True)
                if "keyframe_idxs" in positive[0][1] and hasattr(positive[0][1]["keyframe_idxs"], "shape"):
                    print(f"[RayWorker 0] ID-LoRA keyframe_idxs shape: {tuple(positive[0][1]['keyframe_idxs'].shape)}", flush=True)
                if "guide_attention_entries" in positive[0][1]:
                    print(f"[RayWorker 0] ID-LoRA guide_attention_entries: {len(positive[0][1]['guide_attention_entries'])}", flush=True)

            # ------------------------------------------------------------------
            # Load tensors onto device
            # ------------------------------------------------------------------
            v_noisy = video_dict["samples"].to(device=device, dtype=dtype)
            a_noisy = audio_dict["samples"].to(device=device, dtype=dtype)
            v_mask = video_dict.get("noise_mask")
            a_mask = audio_dict.get("noise_mask")
            if v_mask is not None:
                v_mask = v_mask.to(device=device, dtype=dtype)
            if a_mask is not None:
                a_mask = a_mask.to(device=device, dtype=dtype)

            # Expand broadcast video mask to full video latent spatial dims.
            # LTXVImgToVideoInplace creates (B,1,F,1,1) per-frame masks but
            # process_timestep → patchify needs full (B,*,F,H,W) to produce
            # the correct number of per-patch timestep values.
            if v_mask is not None and v_mask.dim() == 5:
                _, _, _, H, W = v_noisy.shape
                v_mask = v_mask.expand(-1, -1, -1, H, W).contiguous()

            # ref_audio stays as dict — tensors are moved to device inside
            # the patched _process_input when apply_model is called

            # Clean latents for post_process_latent blending.
            if v_clean is not None:
                v_clean = v_clean.to(device=device, dtype=dtype)
            if a_clean is not None:
                a_clean = a_clean.to(device=device, dtype=dtype)

            sigmas = sigmas.float().to(device=device)
            total_steps = len(sigmas) - 1
            batch_size = v_noisy.shape[0]

            # ------------------------------------------------------------------
            # Pack video + audio into flat tensor for apply_model
            # apply_model will unpack via latent_shapes before calling the model.
            # ------------------------------------------------------------------
            packed, latent_shapes = comfy.utils.pack_latents([v_noisy, a_noisy])
            packed_mask = None
            if v_mask is not None or a_mask is not None:
                # Match CFGGuider.sample(): masks must be reshaped to each
                # latent's full shape BEFORE packing so LTXAV.extra_conds()
                # can unpack them using latent_shapes.
                mask_video = v_mask if v_mask is not None else torch.ones_like(v_noisy[:, :1])
                mask_audio = a_mask if a_mask is not None else torch.ones_like(a_noisy[:, :1])
                mask_video = comfy.sampler_helpers.prepare_mask(mask_video, v_noisy.shape, device)
                mask_audio = comfy.sampler_helpers.prepare_mask(mask_audio, a_noisy.shape, device)
                packed_mask, _ = comfy.utils.pack_latents([mask_video, mask_audio])

            # Build the same conditioning structures the standard ComfyUI
            # sampler path uses. This lets LTXAV.extra_conds() create the exact
            # kwargs for apply_model, including text preprocessing, denoise mask
            # unpacking, and any future model-specific conditioning keys.
            model_options = comfy.model_patcher.create_model_options_clone(model.model_options)
            model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
            processed_conds = {
                "positive": comfy.sampler_helpers.convert_cond(positive),
                "negative": comfy.sampler_helpers.convert_cond(negative),
            }
            comfy.sampler_helpers.prepare_model_patcher(model, processed_conds, model_options)
            processed_conds = comfy.samplers.process_conds(
                model.model,
                packed,
                processed_conds,
                device,
                latent_image=packed,
                denoise_mask=packed_mask,
                seed=None,
                latent_shapes=latent_shapes,
            )

            if config.local_rank == 0:
                pos_model_cond_keys = sorted(processed_conds["positive"][0].get("model_conds", {}).keys())
                neg_model_cond_keys = sorted(processed_conds["negative"][0].get("model_conds", {}).keys())
                print(
                    f"[RayWorker 0] ID-LoRA model_conds | pos={pos_model_cond_keys} neg={neg_model_cond_keys}",
                    flush=True,
                )

            def _calc_batch(x_packed, sigma, conds_to_run, ref_audio_override=None, model_options_override=None):
                s = sigma.to(device=device)
                if s.dim() == 0:
                    s = s.expand(batch_size)

                if model_options_override is not None:
                    step_model_options = comfy.model_patcher.create_model_options_clone(model_options_override)
                else:
                    step_model_options = comfy.model_patcher.create_model_options_clone(model_options)
                step_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas

                if ref_audio_override is not None:
                    def _model_function_wrapper(apply_model_fn, args):
                        c = args["c"].copy()
                        c["ref_audio"] = ref_audio_override
                        return apply_model_fn(args["input"], args["timestep"], **c)

                    step_model_options["model_function_wrapper"] = _model_function_wrapper

                return comfy.samplers.calc_cond_batch(
                    model.model,
                    conds_to_run,
                    x_packed,
                    s,
                    step_model_options,
                )

            # ------------------------------------------------------------------
            # Add STG hooks to transformer blocks dynamically
            # ------------------------------------------------------------------
            if getattr(denoise_config, "stg_scale", 0.0) > 0.0:
                def make_stg_wrapper(orig_fn, block_idx):
                    def wrapper(x, *args, **kwargs):
                        opts = kwargs.get("transformer_options", {})
                        skips = opts.get("stg_skip_blocks", None)
                        if skips is not None and block_idx in skips:
                            return torch.zeros_like(x)
                        return orig_fn(x, *args, **kwargs)
                    return wrapper

                try:
                    blocks = model.model.diffusion_model.transformer_blocks
                    for i, block in enumerate(blocks):
                        if not hasattr(block.attn1, "_orig_forward_stg"):
                            block.attn1._orig_forward_stg = block.attn1.forward
                            block.attn1.forward = make_stg_wrapper(block.attn1.forward, i)
                        if hasattr(block, "audio_attn1") and not hasattr(block.audio_attn1, "_orig_forward_stg"):
                            block.audio_attn1._orig_forward_stg = block.audio_attn1.forward
                            block.audio_attn1.forward = make_stg_wrapper(block.audio_attn1.forward, i)
                except Exception as e:
                    print(f"[RayWorker 0] Warning: Failed to attach STG wrappers: {e}", flush=True)

            # ------------------------------------------------------------------
            # Euler denoising loop
            # ------------------------------------------------------------------
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                for step_idx in range(total_steps):
                    sigma = sigmas[step_idx]
                    sigma_next = sigmas[step_idx + 1]

                    # Update CompactFusion step counter
                    try:
                        from raylight.distributed_modules.attention.backends.fusion.main import compact_set_step
                        compact_set_step(step_idx)
                    except Exception:
                        pass

                    if config.local_rank == 0:
                        print(
                            f"[RayWorker 0] ID-LoRA step {step_idx + 1}/{total_steps}"
                            f"  σ={sigma.item():.4f} → {sigma_next.item():.4f}",
                            flush=True,
                        )

                    do_vcfg = denoise_config.video_guidance_scale > 1.0
                    do_acfg = denoise_config.audio_guidance_scale > 1.0

                    # ---- Positive / negative batch via real ComfyUI cond path ----
                    x0_neg_flat = None
                    if do_vcfg or do_acfg:
                        x0_pos_flat, x0_neg_flat = _calc_batch(
                            packed,
                            sigma,
                            [processed_conds["positive"], processed_conds["negative"]],
                            ref_audio_override=ref_audio,
                        )
                    else:
                        x0_pos_flat = _calc_batch(
                            packed,
                            sigma,
                            [processed_conds["positive"]],
                            ref_audio_override=ref_audio,
                        )[0]

                    v_x0_pos, a_x0_pos = comfy.utils.unpack_latents(x0_pos_flat, latent_shapes)
                    v_x0, a_x0 = v_x0_pos.clone(), a_x0_pos.clone()

                    # ---- CFG (video + audio can have different scales) ----
                    if do_vcfg or do_acfg:
                        v_x0_neg, a_x0_neg = comfy.utils.unpack_latents(x0_neg_flat, latent_shapes)
                        if do_vcfg:
                            v_x0 = v_x0 + (denoise_config.video_guidance_scale - 1.0) * (v_x0_pos - v_x0_neg)
                        if do_acfg:
                            a_x0 = a_x0 + (denoise_config.audio_guidance_scale - 1.0) * (a_x0_pos - a_x0_neg)
                        try:
                            del x0_neg_flat, v_x0_neg, a_x0_neg
                        except Exception:
                            pass

                    # ---- STG (Spatio-Temporal Guidance) ----
                    stg_blocks = getattr(denoise_config, "stg_block_idx", None)
                    if getattr(denoise_config, "stg_scale", 0.0) > 0.0 and stg_blocks is not None:
                        stg_opts = comfy.model_patcher.create_model_options_clone(model_options)
                        stg_opts.setdefault("transformer_options", {})["stg_skip_blocks"] = set(stg_blocks)
                        x0_stg_flat = _calc_batch(
                            packed,
                            sigma,
                            [processed_conds["positive"]],
                            ref_audio_override=ref_audio,
                            model_options_override=stg_opts
                        )[0]
                        v_x0_stg, a_x0_stg = comfy.utils.unpack_latents(x0_stg_flat, latent_shapes)
                        v_x0 = v_x0 + denoise_config.stg_scale * (v_x0_pos - v_x0_stg)
                        a_x0 = a_x0 + denoise_config.stg_scale * (a_x0_pos - a_x0_stg)
                        del x0_stg_flat, v_x0_stg, a_x0_stg

                    # ---- AV Bimodal Guidance ----
                    if getattr(denoise_config, "av_bimodal_scale", 0.0) > 1.0:
                        iso_opts = comfy.model_patcher.create_model_options_clone(model_options)
                        iso_t_opts = iso_opts.setdefault("transformer_options", {})
                        iso_t_opts["a2v_cross_attn"] = False
                        iso_t_opts["v2a_cross_attn"] = False
                        x0_iso_flat = _calc_batch(
                            packed,
                            sigma,
                            [processed_conds["positive"]],
                            ref_audio_override=ref_audio,
                            model_options_override=iso_opts
                        )[0]
                        v_x0_iso, a_x0_iso = comfy.utils.unpack_latents(x0_iso_flat, latent_shapes)
                        v_x0 = v_x0 + (denoise_config.av_bimodal_scale - 1.0) * (v_x0_pos - v_x0_iso)
                        a_x0 = a_x0 + (denoise_config.av_bimodal_scale - 1.0) * (a_x0_pos - a_x0_iso)
                        del x0_iso_flat, v_x0_iso, a_x0_iso

                    # ---- Identity guidance (extra no-ref pass) ----
                    # Forward pass without ref_audio, then amplify the delta
                    # introduced by the reference on audio only.
                    if getattr(denoise_config, "identity_guidance_scale", 0.0) > 0.0 and ref_audio is not None:
                        x0_noref_flat = _calc_batch(
                            packed,
                            sigma,
                            [processed_conds["positive"]],
                            ref_audio_override=None,
                        )[0]
                        _, a_x0_noref = comfy.utils.unpack_latents(x0_noref_flat, latent_shapes)
                        id_scale = denoise_config.identity_guidance_scale
                        a_x0 = a_x0 + id_scale * (a_x0_pos - a_x0_noref)
                        del x0_noref_flat, a_x0_noref
                    
                    del v_x0_pos, a_x0_pos, x0_pos_flat

                    # ---- State cleanup between passes ----
                    # Clear trivial mask cache to ensure next pass (with or without ref) 
                    # doesn't reuse incorrect triviality assumptions.
                    if hasattr(model.model.diffusion_model, "_trivial_mask_cache_key"):
                        del model.model.diffusion_model._trivial_mask_cache_key

                    # ---- post_process_latent: restore conditioned regions ----
                    # Reference: denoised * mask + clean * (1 - mask)
                    # Ensures mask=0 regions (e.g. encoded first frame) are
                    # clamped to known-clean values before Euler step.
                    if v_clean is not None and v_mask is not None:
                        v_x0 = v_x0 * v_mask + v_clean * (1.0 - v_mask)
                    if a_clean is not None and a_mask is not None:
                        a_x0 = a_x0 * a_mask + a_clean * (1.0 - a_mask)

                    # ---- Euler step ----
                    # velocity = (x_t - x0) / sigma
                    # x_{t-1} = x_t + velocity * (sigma_next - sigma)
                    dt = sigma_next - sigma  # negative (sigma decreases)
                    v_noisy = v_noisy + dt * (v_noisy - v_x0) / sigma
                    a_noisy = a_noisy + dt * (a_noisy - a_x0) / sigma

                    # Repack for next step
                    packed, _ = comfy.utils.pack_latents([v_noisy, a_noisy])

            return v_noisy.cpu(), a_noisy.cpu()
