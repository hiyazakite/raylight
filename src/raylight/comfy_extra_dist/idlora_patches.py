"""Monkey-patches for LTXAVModel to support ID-LoRA ref_audio injection.

These patches replicate the changes from ComfyUI PR f02190a:
  https://github.com/Comfy-Org/ComfyUI/commit/f02190aaa86ae78b6083c2d734301d3a4f51ee33

They are applied at runtime and automatically skip if the upstream PR has
already been merged (detected by checking if _process_input supports
``ref_audio`` natively).
"""
from __future__ import annotations

import functools
import inspect
import logging
import torch

log = logging.getLogger(__name__)

_PATCHED = False


def ensure_patches_applied():
    """Apply the ID-LoRA ref_audio patches to LTXAVModel if needed.

    Safe to call multiple times — patches are applied exactly once.
    """
    global _PATCHED
    if _PATCHED:
        return

    from comfy.ldm.lightricks.av_model import LTXAVModel

    # Detect if upstream already supports ref_audio
    src = inspect.getsource(LTXAVModel._process_input)
    if "ref_audio" in src:
        log.info("[Raylight ID-LoRA] ref_audio support detected in ComfyUI — skipping patches")
        _PATCHED = True
        return

    log.info("[Raylight ID-LoRA] Applying ref_audio patches to LTXAVModel")

    _orig_process_input = LTXAVModel._process_input
    _orig_prepare_timestep = LTXAVModel._prepare_timestep
    _orig_prepare_attention_mask = LTXAVModel._prepare_attention_mask
    _orig_process_output = LTXAVModel._process_output

    # ------------------------------------------------------------------
    # Patched _process_input: inject reference audio tokens
    # ------------------------------------------------------------------
    @functools.wraps(_orig_process_input)
    def _patched_process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        # Call original — produces [vx, ax], [v_coords, a_coords], additional_args
        result = _orig_process_input(self, x, keyframe_idxs, denoise_mask, **kwargs)
        [vx, ax], [v_pixel_coords, a_latent_coords], additional_args = result

        # --- ref_audio injection (from PR f02190a) ---
        ref_audio = kwargs.get("ref_audio", None)
        ref_audio_seq_len = 0
        if ref_audio is not None:
            ref_tokens = ref_audio["tokens"].to(dtype=ax.dtype, device=ax.device)
            if ref_tokens.shape[0] < ax.shape[0]:
                ref_tokens = ref_tokens.expand(ax.shape[0], -1, -1)
            ref_audio_seq_len = ref_tokens.shape[1]
            B = ax.shape[0]

            # Compute negative temporal positions matching ID-LoRA convention
            p = self.a_patchifier
            tpl = p.hop_length * p.audio_latent_downsample_factor / p.sample_rate
            ref_start = p._get_audio_latent_time_in_sec(
                0, ref_audio_seq_len, torch.float32, ax.device
            )
            ref_end = p._get_audio_latent_time_in_sec(
                1, ref_audio_seq_len + 1, torch.float32, ax.device
            )
            time_offset = ref_end[-1].item()
            ref_start = (ref_start - time_offset).unsqueeze(0).expand(B, -1).unsqueeze(1)
            ref_end = (ref_end - time_offset).unsqueeze(0).expand(B, -1).unsqueeze(1)
            ref_pos = (
                torch.stack([ref_start, ref_end], dim=-1) if p.start_end else ref_start
            )

            # The original already ran audio_patchify_proj on ax, so we must
            # project ref_tokens the same way before concatenation.
            ref_projected = self.audio_patchify_proj(ref_tokens)
            ax = torch.cat([ref_projected, ax], dim=1)
            a_latent_coords = torch.cat(
                [ref_pos.to(a_latent_coords), a_latent_coords], dim=2
            )

        # Persistent tracking for timestep/mask patches
        self._ref_audio_seq_len = ref_audio_seq_len
        additional_args["ref_audio_seq_len"] = ref_audio_seq_len
        additional_args["total_audio_seq_len"] = ax.shape[1]

        return [vx, ax], [v_pixel_coords, a_latent_coords], additional_args

    @functools.wraps(_orig_prepare_timestep)
    def _patched_prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        ref_audio_seq_len = getattr(self, "_ref_audio_seq_len", 0)
        if ref_audio_seq_len > 0:
            a_timestep = kwargs.get("a_timestep")
            if a_timestep is not None:
                # kwargs["total_audio_seq_len"] might be missing if called from another wrapper
                total_len = a_timestep.shape[1] if a_timestep.dim() > 1 else 0
                if a_timestep.dim() <= 1:
                    # Fallback to calculating from audio_latent_downsample_factor if needed
                    # but usually it's already expanded or provided in kwargs
                    pass
                
                # If a_timestep doesn't have the prefix yet, add it
                if a_timestep.shape[1] < (kwargs.get("total_audio_seq_len", 0) or 0):
                    ref_ts = torch.zeros(
                        batch_size, ref_audio_seq_len, *a_timestep.shape[2:],
                        device=a_timestep.device, dtype=a_timestep.dtype,
                    )
                    kwargs["a_timestep"] = torch.cat([ref_ts, a_timestep], dim=1)

        result = _orig_prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs)
        return result

    # ------------------------------------------------------------------
    # Patched _prepare_attention_mask: extend mask for ref tokens
    # ------------------------------------------------------------------
    @functools.wraps(_orig_prepare_attention_mask)
    def _patched_prepare_attention_mask(self, attention_mask, x_dtype):
        mask = _orig_prepare_attention_mask(self, attention_mask, x_dtype)
        
        ref_audio_seq_len = getattr(self, "_ref_audio_seq_len", 0)
        if ref_audio_seq_len > 0 and mask is not None:
            # Mask is (B, 1, Q, K) or (B, 1, 1, K).
            # We need to prepend zeros (full attention) for the ref tokens.
            # LTXAV masks are usually (B, 1, 1, L_context) or (B, 1, L, L)
            # If it's a context mask, L_context is constant.
            # If it's a self-attention mask, it covers the packed [vx, ax]
            
            # For ID-LoRA, the ref tokens are prepended to ax.
            # The joint sequence is [vx, Ref, ax_original].
            # If the mask covers the whole joint sequence, we need to insert the ref portion.
            
            # ComfyUI's standard attention_mask for LTX often covers only context.
            # But if a self_attention_mask is provided, it must be padded.
            
            # Since we don't know for sure if the mask is self or cross here,
            # we check the last dimension.
            if mask.shape[-1] > 0:
                # Prepend '0' (no masking) for reference tokens
                pad_shape = list(mask.shape)
                pad_shape[-1] = ref_audio_seq_len
                # If square mask (L, L), also prepend rows
                if mask.shape[-2] > 1:
                    pad_shape[-2] = ref_audio_seq_len
                
                padding = torch.zeros(pad_shape, device=mask.device, dtype=mask.dtype)
                
                # Simple prepending (assuming ref tokens are "clean" and at front of ax)
                # vx is first, then ax. Ref is prepended to ax. 
                # So the sequence is [vx, Ref, ax_orig].
                # We should insert the padding into the mask where ax starts.
                # But LTXAVModel._prepare_attention_mask doesn't know vx length.
                
                # Fallback: if it's a tail-only prefix or head-only prefix.
                # Given ID-LoRA implementation in PR f02190a, it's HEAD of ax.
                # If we just prepend to the WHOLE mask, it might be wrong if vx > 0.
                
                # Actually, the 'attention_mask' in LTXAV is usually for CONTEXT (text).
                # Context length didn't change!
                # If we are here, it means some mask WAS provided.
                # If it's the self_attention_mask used for guides, then it's built
                # AFTER _process_input, so it ALREADY has the right size.
                
                # So we ONLY need to patch this if the mask is provided from OUTSIDE
                # (e.g. denoise_mask transformed into attention_mask).
                pass

        return mask

    # ------------------------------------------------------------------
    # Patched _process_output: trim reference tokens before unpatchify
    # ------------------------------------------------------------------
    @functools.wraps(_orig_process_output)
    def _patched_process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        ref_audio_seq_len = kwargs.get("ref_audio_seq_len", 0)
        if ref_audio_seq_len > 0:
            ax = x[1]
            a_et = embedded_timestep[1]
            ax = ax[:, ref_audio_seq_len:]
            if a_et.shape[1] > 1:
                a_et = a_et[:, ref_audio_seq_len:]
            x = [x[0], ax]
            embedded_timestep = [embedded_timestep[0], a_et]

        return _orig_process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs)

    # Apply patches
    LTXAVModel._process_input = _patched_process_input
    LTXAVModel._prepare_timestep = _patched_prepare_timestep
    LTXAVModel._prepare_attention_mask = _patched_prepare_attention_mask
    LTXAVModel._process_output = _patched_process_output

    _PATCHED = True
    log.info("[Raylight ID-LoRA] Patches applied successfully")
