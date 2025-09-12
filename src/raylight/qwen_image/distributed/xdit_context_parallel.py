import torch
from typing import Optional, Tuple
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
attn_type = xfuser_attn.get_attn_type()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type)


def pad_freqs(original_tensor, target_len):
    """
    original_tensor: [B, L_global, 1, D/2, 2, 2] — full freq tensor
    """
    b, seq_len, z, dim, a, c = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    padding_tensor = torch.ones(
        b,
        pad_size,
        z,
        dim,
        a,
        c,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=1)
    return padded_tensor


def apply_rotary_emb_sp(xq, xk, freqs_cis):
    """
    xq, xk:       [B, L_local, 1, D]
    freqs_cis:    [B, L_global, 1, D/2, 2, 2] — full freq tensor
    Returns:      RoPE-applied local xq, xk
    """

    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()

    B, L_local, _, D = xq.shape
    L_global = L_local * sp_size

    # Ensure freqs_cis has length L_global
    freqs_cis = pad_freqs(freqs_cis, L_global)

    # Slice the correct frequency chunk for this rank
    start = sp_rank * L_local
    end = start + L_local
    freqs_local = freqs_cis[:, start:end]  # [B, L_local, 1, D/2, 2, 2]

    # Prepare xq/xk for RoPE (split real/imag)
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

    # Apply RoPE using local frequencies
    xq_out = freqs_local[..., 0] * xq_[..., 0] + freqs_local[..., 1] * xq_[..., 1]
    xk_out = freqs_local[..., 0] * xk_[..., 0] + freqs_local[..., 1] * xk_[..., 1]

    return xq_out.reshape_as(xq), xk_out.reshape_as(xk)


def usp_dit_forward(
    self,
    x,
    timesteps,
    context,
    attention_mask=None,
    guidance: torch.Tensor = None,
    ref_latents=None,
    transformer_options={},
    control=None,
    **kwargs
):
    timestep = timesteps
    encoder_hidden_states = context
    encoder_hidden_states_mask = attention_mask

    hidden_states, img_ids, orig_shape = self.process_img(x)
    num_embeds = hidden_states.shape[1]

    if ref_latents is not None:
        h = 0
        w = 0
        index = 0
        index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
        for ref in ref_latents:
            if index_ref_method:
                index += 1
                h_offset = 0
                w_offset = 0
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            hidden_states = torch.cat([hidden_states, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
    txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
    del ids, txt_ids, img_ids

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    # Context parallel
    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()
    hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]
    encoder_hidden_states = torch.chunk(encoder_hidden_states, sp_world_size, dim=1)[sp_rank]

    patches_replace = transformer_options.get("patches_replace", {})
    patches = transformer_options.get("patches", {})
    blocks_replace = patches_replace.get("dit", {})

    for i, block in enumerate(self.transformer_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"])
                return out
            out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb}, {"original_block": block_wrap})
            hidden_states = out["img"]
            encoder_hidden_states = out["txt"]
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    hidden_states[:, :add.shape[1]] += add

    hidden_states = get_sp_group().all_gather(hidden_states, dim=1)
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
    hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
    return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]


def usp_attn_forward(
    self,
    hidden_states: torch.FloatTensor,  # Image stream
    encoder_hidden_states: torch.FloatTensor = None,  # Text stream
    encoder_hidden_states_mask: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_txt = encoder_hidden_states.shape[1]

    img_query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
    img_key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
    img_value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

    txt_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
    txt_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
    txt_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))

    img_query = self.norm_q(img_query)
    img_key = self.norm_k(img_key)
    txt_query = self.norm_added_q(txt_query)
    txt_key = self.norm_added_k(txt_key)

    joint_query = torch.cat([txt_query, img_query], dim=1)
    joint_key = torch.cat([txt_key, img_key], dim=1)
    joint_value = torch.cat([txt_value, img_value], dim=1)

    joint_query, joint_key = apply_rotary_emb_sp(joint_query, joint_key, image_rotary_emb)

    joint_query = joint_query.flatten(start_dim=2)
    joint_key = joint_key.flatten(start_dim=2)
    joint_value = joint_value.flatten(start_dim=2)

    joint_hidden_states = xfuser_optimized_attention(joint_query, joint_key, joint_value, self.heads, attention_mask)

    txt_attn_output = joint_hidden_states[:, :seq_txt, :]
    img_attn_output = joint_hidden_states[:, seq_txt:, :]

    img_attn_output = self.to_out[0](img_attn_output)
    img_attn_output = self.to_out[1](img_attn_output)
    txt_attn_output = self.to_add_out(txt_attn_output)

    return img_attn_output, txt_attn_output
