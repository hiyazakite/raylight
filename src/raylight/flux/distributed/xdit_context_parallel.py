import math

import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
attn_type = xfuser_attn.get_attn_type()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type)


def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    if modulation_dims is None:
        if m_add is not None:
            return torch.addcmul(m_add, tensor, m_mult)
        else:
            return tensor * m_mult
    else:
        for d in modulation_dims:
            tensor[:, d[0]:d[1]] *= m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0]:d[1]] += m_add[:, d[2]]
        return tensor


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def pad_freqs(original_tensor, target_len):
    """
    original_tensor: [B, 1, L_global, D/2, 2, 2] — full freq tensor
    """
    b, z, seq_len, dim, a, c = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    padding_tensor = torch.ones(
        b,
        z,
        pad_size,
        dim,
        a,
        c,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=1)
    return padded_tensor


def apply_rope_sp(xq, xk, freqs_cis):
    """
    xq, xk:       [B, 1, L_local, D]
    freqs_cis:    [B, 1, L_global, D/2, 2, 2] — full freq tensor
    Returns:      RoPE-applied local xq, xk
    """

    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()

    B, _, L_local, D = xq.shape
    L_global = L_local * sp_size

    # Ensure freqs_cis has length L_global
    freqs_cis = pad_freqs(freqs_cis, L_global)

    # Slice the correct frequency chunk for this rank
    start = sp_rank * L_local
    end = start + L_local
    freqs_local = freqs_cis[:, :, start:end]  # [B, 1, L_local, D]

    # Prepare xq/xk for RoPE (split real/imag)
    xq_ = xq.to(dtype=freqs_local.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_local.dtype).reshape(*xk.shape[:-1], -1, 1, 2)

    # Apply RoPE using local frequencies
    xq_out = freqs_local[..., 0] * xq_[..., 0] + freqs_local[..., 1] * xq_[..., 1]
    xk_out = freqs_local[..., 0] * xk_[..., 0] + freqs_local[..., 1] * xk_[..., 1]

    return xq_out.reshape_as(xq).type_as(xq), xk_out.reshape_as(xk).type_as(xk)


def attention_join(q, k, v, join_q, join_k, join_v, mask=None) -> Tensor:
    heads = q.shape[1]
    x = xfuser_optimized_attention(
        q,
        k,
        v,
        heads,
        join_q=join_q,
        join_k=join_k,
        join_v=join_v,
        skip_reshape=True
    )
    return x


def attention(q, k, v, mask=None) -> Tensor:
    heads = q.shape[1]
    x = xfuser_optimized_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True
    )
    return x


def usp_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:

    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
    txt = self.txt_in(txt)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = get_sp_group().all_gather(img, dim=1)
    img = torch.cat((txt, img), 1)

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    for i, block in enumerate(self.single_blocks):
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]:, ...] += add

    # Context parallel
    img = get_sp_group().all_gather(img, dim=1)

    img = img[:, txt.shape[1]:, ...]
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


def usp_single_stream_forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims=None) -> Tensor:
    mod, _ = self.modulation(vec)
    qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)

    if pe is not None:
        q, k = apply_rope_sp(q, k, pe)
    attn = attention(q, k, v)
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def usp_double_stream_forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None):
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    # For context parallel, q = B, _, L, D
    img_seq_len = img_q.shape[2]
    txt_seq_len = txt_q.shape[2]

    # I shoul've just divided the RoPE before and inject them in model forward instead of block forward
    if self.flipped_img_txt:
        q = torch.cat((img_q, txt_q), dim=2)
        k = torch.cat((img_k, txt_k), dim=2)
        if pe is not None:
            q[:, :, :img_seq_len, :], k[:, :, :img_seq_len, :] = apply_rope_sp(
                q[:, :, :img_seq_len, :],
                k[:, :, :img_seq_len, :],
                pe)

        img_q, txt_q = q.split([img_seq_len, txt_seq_len], dim=2)
        img_k, txt_k = k.split([img_seq_len, txt_seq_len], dim=2)

        # run actual attention
        attn = attention_join(img_q, img_k, img_v, txt_q, txt_k, txt_v, mask=attn_mask)
        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
    else:
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        q[:, :, txt_seq_len:, :], k[:, :, txt_seq_len:, :] = apply_rope_sp(
            q[:, :, txt_seq_len:, :],
            k[:, :, txt_seq_len:, :],
            pe)

        txt_q, img_q = q.split([txt_seq_len, img_seq_len], dim=2)
        txt_k, img_k = k.split([txt_seq_len, img_seq_len], dim=2)
        # run actual attention
        attn = attention_join(txt_q, txt_k, txt_v, img_q, img_k, img_v, mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

    # calculate the txt bloks
    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt
