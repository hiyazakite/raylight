import math

import torch
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn
from raylight.distributed_modules.sequence_parallel import extract_local_tensor
xfuser_optimized_attention = xfuser_attn.make_lazy_attention()


def _get_ring_impl_type():
    """Read ring_impl_type lazily so config changes take effect between runs."""
    return getattr(xfuser_attn, "get_ring_impl_type", lambda: "basic")()


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


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q, k, v, pe, mask=None, **kwargs) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)

    heads = q.shape[2]
    x = xfuser_optimized_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        **kwargs
    )
    return x


def _invert_slices(slices, length):
    sorted_slices = sorted(slices)
    result = []
    current = 0
    for start, end in sorted_slices:
        if current < start:
            result.append((current, start))
        current = max(current, end)
    if current < length:
        result.append((current, length))
    return result


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
    timestep_zero_index=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img, img_orig_size = pad_to_world_size(img, dim=1)
    img_ids, _ = pad_to_world_size(img_ids, dim=1)
    txt, txt_orig_size = pad_to_world_size(txt, dim=1)
    txt_ids, _ = pad_to_world_size(txt_ids, dim=1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    txt = self.txt_in(txt)

    vec_orig = vec
    txt_vec = vec
    modulation_dims = None
    double_extra_kwargs = {}
    single_extra_kwargs = {}
    if timestep_zero_index is not None:
        modulation_dims = []
        batch = vec.shape[0] // 2
        vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
        invert = _invert_slices(timestep_zero_index, img.shape[1])
        for s in invert:
            modulation_dims.append((s[0], s[1], 0))
        for s in timestep_zero_index:
            modulation_dims.append((s[0], s[1], 1))
        double_extra_kwargs["modulation_dims_img"] = modulation_dims
        txt_vec = vec[:batch]

    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe_combine = self.pe_embedder(ids)
        pe_image = self.pe_embedder(img_ids)
        pe_combine = pe_combine.transpose(1, 2)
        pe_image = pe_image.transpose(1, 2)
        # seq parallel
        pe_combine = torch.chunk(pe_combine, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
        pe_image = torch.chunk(pe_image, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
        pe_combine = pe_combine.contiguous()
        pe_image = pe_image.contiguous()
    else:
        pe_combine = None
        pe_image = None

    img = extract_local_tensor(img, ring_impl_type=_get_ring_impl_type())
    txt = extract_local_tensor(txt, ring_impl_type=_get_ring_impl_type())
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               **double_extra_kwargs)
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe_image,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe_image,
                             attn_mask=attn_mask,
                             **double_extra_kwargs)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    # NOTE: synchronize() removed — all_gather is stream-ordered via NCCL and
    # already sees all preceding CUDA work on the current stream.
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    txt = get_sp_group().all_gather(txt.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    txt = txt[:, :txt_orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img, img_orig_size = pad_to_world_size(img, dim=1)

    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)
    img = extract_local_tensor(img, ring_impl_type=_get_ring_impl_type())
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    if modulation_dims is not None:
        txt_len = txt.shape[1]
        modulation_dims_combined = [
            (0 if x[0] == 0 else x[0] + txt_len, x[1] + txt_len, x[2])
            for x in modulation_dims
        ]
        single_extra_kwargs = {"modulation_dims": modulation_dims_combined}

    # Pre-compute the number of text tokens that fall within this rank's
    # local chunk.  After extract_local_tensor the combined (txt+img)
    # sequence is split across SP ranks, so the global txt↔img boundary
    # maps to a different local offset per rank.
    _sp_world = get_sequence_parallel_world_size()
    if _sp_world > 1:
        _sp_rank = get_sequence_parallel_rank()
        _local_seq = img.shape[1]  # local chunk length
        _txt_len = txt.shape[1]    # full (gathered) text length
        _rim = _get_ring_impl_type()
        if _rim == "zigzag":
            _half = _local_seq // 2
            _front_start = _sp_rank * _half
            _local_txt_tokens = max(0, min(_txt_len - _front_start, _half))
        else:  # basic
            _global_start = _sp_rank * _local_seq
            _local_txt_tokens = max(0, min(_txt_len - _global_start, _local_seq))
    else:
        _local_txt_tokens = txt.shape[1]

    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    for i, block in enumerate(self.single_blocks):
        # Add offset for single blocks if needed, usually Flux follows doubles then singles
        transformer_options["block_index"] = i + len(self.double_blocks)
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   **single_extra_kwargs)
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe_combine,
                                                       "attn_mask": attn_mask},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe_combine, attn_mask=attn_mask, **single_extra_kwargs)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, _local_txt_tokens:, ...] += add

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    # NOTE: synchronize() removed — all_gather is stream-ordered via NCCL.
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    img = img[:, :img_orig_size, :]
    # ======================== ADD SEQUENCE PARALLEL ========================= #
    img = img[:, txt.shape[1]:, ...]

    final_extra_kwargs = {}
    if modulation_dims is not None:
        final_extra_kwargs["modulation_dims"] = modulation_dims
    img = self.final_layer(img, vec_orig, **final_extra_kwargs)  # (N, T, patch_size ** 2 * out_channels)
    return img


def usp_single_stream_forward(
    self,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims=None,
    **kwargs
) -> Tensor:
    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec

    qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
    del qkv
    q, k = self.norm(q, k, v)

    # compute attention
    mod_idx = kwargs.get("modifier", {}).get("block_index")
    if mod_idx is None:
        mod_idx = kwargs.get("transformer_options", {}).get("block_index")
        
    attn = attention(q, k, v, pe=pe, mask=attn_mask, mod_idx=mod_idx)
    del q, k, v

    # compute activation in mlp stream, cat again and run second linear layer
    if self.yak_mlp:
        mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
    else:
        mlp = self.mlp_act(mlp)

    output = self.linear2(torch.cat((attn, mlp), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def usp_double_stream_forward(
    self,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    **kwargs
):
    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
    img_qkv = self.img_attn.qkv(img_modulated)
    del img_modulated
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
    del img_qkv
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    del txt_modulated
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)
    del txt_qkv
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    if getattr(self, "flipped_img_txt", False):
        img_q, img_k = apply_rope(img_q, img_k, pe)
        q = torch.cat((img_q, txt_q), dim=1)
        del img_q, txt_q
        k = torch.cat((img_k, txt_k), dim=1)
        del img_k, txt_k
        v = torch.cat((img_v, txt_v), dim=1)
        del img_v, txt_v
        # run actual attention
        mod_idx = kwargs.get("modifier", {}).get("block_index")
        if mod_idx is None:
            mod_idx = kwargs.get("transformer_options", {}).get("block_index")
            
        attn = attention(q, k, v, pe=None, mask=attn_mask, mod_idx=mod_idx)
        del q, k, v

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]
    else:
        img_q, img_k = apply_rope(img_q, img_k, pe)
        q = torch.cat((txt_q, img_q), dim=1)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=1)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=1)
        del txt_v, img_v
        # run actual attention
        mod_idx = kwargs.get("modifier", {}).get("block_index")
        if mod_idx is None:
            mod_idx = kwargs.get("transformer_options", {}).get("block_index")
            
        attn = attention(q, k, v, pe=None, mask=attn_mask, mod_idx=mod_idx)
        del q, k, v

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img += apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    del img_attn
    img += apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

    # calculate the txt bloks
    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    del txt_attn
    txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt
