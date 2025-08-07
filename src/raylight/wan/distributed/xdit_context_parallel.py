# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from comfy.ldm.flux.math import apply_rope


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculationk
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def pad_freqs(original_tensor, target_len):
    b, seq_len, z, dim, a, c  = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    padding_tensor = torch.ones(
        b, pad_size, z, dim, a, c, dtype=original_tensor.dtype, device=original_tensor.device
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=1)
    return padded_tensor


def apply_rope_sp(xq, xk, freqs_cis):
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
    xq_ = xq.to(dtype=freqs_local.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_local.dtype).reshape(*xk.shape[:-1], -1, 1, 2)

    # Apply RoPE using local frequencies
    xq_out = freqs_local[..., 0] * xq_[..., 0] + freqs_local[..., 1] * xq_[..., 1]
    xk_out = freqs_local[..., 0] * xk_[..., 0] + freqs_local[..., 1] * xk_[..., 1]

    return xq_out.reshape_as(xq).type_as(xq), xk_out.reshape_as(xk).type_as(xk)



# def apply_rope_sp(xq_ori, xk_ori, freqs_cis):
#     """
#     Applies RoPE on sequence-parallel chunk only.

#     xq, xk: [B, L, 1, D]
#     freqs_cis: [B, L, 1, D/2, 2, 2] or [L, 1, D/2, 2, 2]
#     Returns: local chunk of RoPE-applied xq, xk
#     """

#     sp_rank = get_sequence_parallel_rank()
#     sp_size = get_sequence_parallel_world_size()

#     B, L, _, D = xq_ori.shape
#     s_per_rank = L // sp_size
#     start = sp_rank * s_per_rank
#     end = (sp_rank + 1) * s_per_rank

#     xq = xq_ori[:, start:end]
#     xk = xk_ori[:, start:end]

#     freqs_padded = pad_freqs(freqs_cis, L * sp_size)
#     freqs_local = freqs_padded[:, start:end]

#     # RoPE application
#     xq_ = xq.to(dtype=freqs_local.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
#     xk_ = xk.to(dtype=freqs_local.dtype).reshape(*xk.shape[:-1], -1, 1, 2)

#     xq_out = freqs_local[..., 0] * xq_[..., 0] + freqs_local[..., 1] * xq_[..., 1]
#     xk_out = freqs_local[..., 0] * xk_[..., 0] + freqs_local[..., 1] * xk_[..., 1]

#     # Write back the RoPE-applied chunk to the original tensors
#     xq_ori[:, start:end] = xq_out.reshape_as(xq).type_as(xq_ori)
#     xk_ori[:, start:end] = xk_out.reshape_as(xk).type_as(xk_ori)

#     return xq_ori, xk_ori



def usp_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    freqs=None,
    transformer_options={},
    **kwargs,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (Tensor):
            List of input video tensors with shape [B, C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [B, L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    # embeddings

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    # time embeddings
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype)
    )
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # context
    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    print("before chunk")
    print(x.size())
    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[
        get_sequence_parallel_rank()
    ]

    print("after chunk")
    print(x.size())

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.blocks):
        if ("double_block", i) in blocks_replace:

            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                )
                return out

            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len
            )

    # head
    x = self.head(x, e)

    print("forward")
    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)


    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    return x


def usp_attn_forward(self, x, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = apply_rope_sp(q, k, freqs)

    x = xFuserLongContextAttention()(
        None, query=q, key=k, value=v, window_size=self.window_size
    )

    x = x.flatten(2)
    x = self.o(x)
    return x


