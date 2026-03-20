import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

from xfuser.core.cache_manager.cache_manager import get_cache_manager
try:
    from yunchang.ring.utils import RingComm, update_out_and_lse
    from yunchang.kernels import select_flash_attn_impl, AttnType
except ImportError:
    RingComm = None
    update_out_and_lse = None
    select_flash_attn_impl = None
    AttnType = None

from typing import Optional, Any, Tuple
import torch.nn as nn

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask, AuxRequest
    _HAS_FLEX_ATTN = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    AuxRequest = None
    _HAS_FLEX_ATTN = False

try:
    import xformers
    from xformers.ops.fmha import memory_efficient_attention_forward, LowerTriangularMask
except ImportError:
    xformers = None
    memory_efficient_attention_forward = None
    LowerTriangularMask = None


# ────────────────────────────────────────────────────────────────────────────
# Shared utility: chunked LSE computation for SDPA fallback
# ────────────────────────────────────────────────────────────────────────────

def compute_chunked_lse(q_t, k_t, softmax_scale, mask=None, causal=False):
    """Compute log-sum-exp in chunks to avoid O(N^2) memory spikes.

    Args:
        q_t: Query tensor in (B, H, Q, D) layout.
        k_t: Key tensor in (B, H, K, D) layout.
        softmax_scale: Attention scaling factor.
        mask: Optional attention mask broadcastable to (B, H, Q, K).
        causal: Whether to apply causal masking.

    Returns:
        LSE tensor of shape (B, H, Q).
    """
    bs, heads, q_len, _ = q_t.shape
    k_len = k_t.shape[2]
    CHUNK_SIZE = max(1, 1024 * 32 // k_len) if k_len > 0 else 1024
    lse = torch.empty(bs, heads, q_len, device=q_t.device, dtype=torch.float32)

    with torch.no_grad():
        kt_v = k_t.transpose(-2, -1)
        for i in range(0, q_len, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, q_len)
            q_chunk = q_t[:, :, i:end, :]
            attn_chunk = torch.matmul(q_chunk, kt_v) * softmax_scale

            if mask is not None:
                m_chunk = mask[:, :, i:end, :] if mask.shape[2] > 1 else mask
                attn_chunk = attn_chunk + m_chunk

            if causal:
                row_indices = torch.arange(i, end, device=q_t.device).view(1, 1, -1, 1)
                col_indices = torch.arange(k_len, device=q_t.device).view(1, 1, 1, -1)
                attn_chunk.masked_fill_(row_indices < col_indices, float("-inf"))

            lse[:, :, i:end] = torch.logsumexp(attn_chunk.float(), dim=-1)
            del attn_chunk

    return lse


def manual_block_attention(
    q, k, v, 
    dropout_p=0.0, 
    softmax_scale=None, 
    causal=False, 
    mask=None
):
    """
    Fallback attention for non-trivial masks.
    Tries FlexAttention -> xformers -> SDPA with chunked LSE for memory efficiency.
    """
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    # 1. Try FlexAttention (Fastest, avoids materializing large masks)
    if _HAS_FLEX_ATTN and mask is not None:
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + mask[b, h % mask.shape[1], q_idx, kv_idx]

        if flex_attention is None or AuxRequest is None:
            raise ImportError("FlexAttention components not available despite _HAS_FLEX_ATTN being True")

        # We need to ensure they match (B, H, S, D).
        if q.dim() == 3:
            b, l, dim = q.shape
            h = mask.shape[1]
            d = dim // h
            q_flex = q.view(b, l, h, d).transpose(1, 2)
            k_flex = k.view(b, k.shape[1], h, d).transpose(1, 2)
            v_flex = v.view(b, v.shape[1], h, d).transpose(1, 2)
        elif q.dim() == 4:
            if q.shape[1] > q.shape[2]:  # (B, S, H, D)
                q_flex = q.transpose(1, 2)
                k_flex = k.transpose(1, 2)
                v_flex = v.transpose(1, 2)
            else:  # already B, H, S, D
                q_flex = q
                k_flex = k
                v_flex = v
        else:
            raise ValueError(f"Expected 3D or 4D query, got {q.dim()}D")

        res = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod, return_aux=AuxRequest(lse=True))
        out, aux = res
        lse = getattr(aux, "lse", None)
        out = out.transpose(1, 2).contiguous()
        return out, lse

    # 2. Try xformers
    if xformers is not None and memory_efficient_attention_forward is not None:
        attn_bias = mask
        if causal and attn_bias is None and LowerTriangularMask is not None:
            attn_bias = LowerTriangularMask()

        if attn_bias is not None and (LowerTriangularMask is None or not isinstance(attn_bias, LowerTriangularMask)):
            if hasattr(attn_bias, "shape") and getattr(attn_bias, "shape")[1] == 1 and q.shape[2] > 1:
                attn_bias = getattr(attn_bias, "expand")(-1, q.shape[2], -1, -1)

        res = memory_efficient_attention_forward(
            q, k, v,
            attn_bias=attn_bias,
            p=dropout_p,
            scale=softmax_scale,
        )
        if isinstance(res, tuple):
            out, lse = res[0], res[1]
        else:
            out, lse = res, None
        return out, lse

    # 3. SDPA Fallback
    q_t = q.transpose(1, 2)  # (B, H, L, D)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal and mask is None,
        scale=softmax_scale
    )
    
    lse = compute_chunked_lse(q_t, k_t, softmax_scale, mask=mask, causal=causal)
    out = out.transpose(1, 2).contiguous()
    return out, lse

def get_zigzag_mask_chunk(mask, q_rank, k_rank, world_size, block_seq_len, q_slice=None, k_slice=None):
    """
    Extract the attention mask chunk for non-contiguous zigzag blocks.
    q_slice: 0 for first half (B_rank), 1 for second half (B_2N-rank-1), None for both.
    k_slice: 0 for first half (B_rank), 1 for second half (B_2N-rank-1), None for both.
    """
    if mask is None: return None
    
    q_indices = []
    if q_slice == 0 or q_slice is None:
        q_indices.append((q_rank * block_seq_len, (q_rank + 1) * block_seq_len))
    if q_slice == 1 or q_slice is None:
        q_indices.append(((2 * world_size - q_rank - 1) * block_seq_len, (2 * world_size - q_rank) * block_seq_len))
    
    k_indices = []
    if k_slice == 0 or k_slice is None:
        k_indices.append((k_rank * block_seq_len, (k_rank + 1) * block_seq_len))
    if k_slice == 1 or k_slice is None:
        k_indices.append(((2 * world_size - k_rank - 1) * block_seq_len, (2 * world_size - k_rank) * block_seq_len))
    
    q_chunks = []
    for qs, qe in q_indices:
        k_chunks = []
        for ks, ke in k_indices:
            k_chunks.append(mask[:, :, qs:qe, ks:ke])
        q_chunks.append(torch.cat(k_chunks, dim=-1))
    
    return torch.cat(q_chunks, dim=-2)


def xdit_ring_zigzag_flash_attn_forward_patched(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    attn_type: Optional[Any] = None,
    attn_processor: Optional[nn.Module] = None,
    attn_layer: Optional[Any] = None,
    joint_tensor_value: Optional[torch.Tensor] = None,
    joint_strategy: str = "none",
    mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zigzag Ring Flash Attention with Mask and Joint Tensor support.
    Follows Yunchang's zigzag logic for load-balancing causal attention.
    """
    assert causal == True, "zigzag ring is only supported for causal=True"
    
    if RingComm is None:
        raise ImportError("yunchang.ring.utils.RingComm is required for xfuser ring attention")
    comm = RingComm(process_group)
    rank = comm.rank
    world_size = comm.world_size

    # In zigzag, query/key/value are each [rank, 2*world_size - rank - 1] blocks
    block_seq_len = q.shape[1] // 2
    
    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()

    from raylight.distributed_modules.attention import check_mask_nontrivial
    has_nontrivial_mask = check_mask_nontrivial(mask)

    def iterate_attn(_q, _k, _v, _causal, _mask_chunk):
        if has_nontrivial_mask and _mask_chunk is not None:
             return manual_block_attention(
                _q, _k, _v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=_causal, mask=_mask_chunk
            )
        else:
            if AttnType is None:
                raise ImportError("yunchang.kernels.AttnType is required for flash attention")
            if select_flash_attn_impl is None:
                raise ImportError("yunchang.kernels.select_flash_attn_impl is required for flash attention")
            _attn_type = getattr(AttnType, "FA") if attn_type is None else attn_type
            fn: Any = select_flash_attn_impl(_attn_type, stage="fwd-only", attn_processor=attn_processor) # type: ignore
            return fn(
                _q, _k, _v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=_causal, window_size=window_size, softcap=0.0,
                alibi_slopes=alibi_slopes, return_softmax=True and dropout_p > 0
            )

    curr_k: torch.Tensor = k
    curr_v: torch.Tensor = v
    next_k: Optional[torch.Tensor] = None
    next_v: Optional[torch.Tensor] = None

    for step in range(world_size):
        if step + 1 != world_size:
            next_k = comm.send_recv(curr_k) # type: ignore
            next_v = comm.send_recv(curr_v) # type: ignore
            comm.commit() # type: ignore

        recv_rank = (rank - step) % world_size
        
        if step == 0:
            # Step 0: Causal attn on full local Q vs full local K
            m_chunk = get_zigzag_mask_chunk(mask, rank, rank, world_size, block_seq_len) if has_nontrivial_mask else None
            block_out, block_lse = iterate_attn(q, curr_k, curr_v, _causal=True, _mask_chunk=m_chunk)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse) # type: ignore
        elif step <= rank:
            # Attend to the first half of the received blocks (K_recv_rank)
            k0 = curr_k[:, :block_seq_len] # type: ignore
            v0 = curr_v[:, :block_seq_len] # type: ignore
            m_chunk = get_zigzag_mask_chunk(mask, rank, recv_rank, world_size, block_seq_len, k_slice=0) if has_nontrivial_mask else None
            block_out, block_lse = iterate_attn(q, k0, v0, _causal=False, _mask_chunk=m_chunk)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse) # type: ignore
        else:
            # Only the second half of Q attends to the received blocks
            q1 = q[:, block_seq_len:]
            m_chunk = get_zigzag_mask_chunk(mask, rank, recv_rank, world_size, block_seq_len, q_slice=1) if has_nontrivial_mask else None
            block_out, block_lse = iterate_attn(q1, curr_k, curr_v, _causal=False, _mask_chunk=m_chunk)
            out, lse = update_out_and_lse(
                out, lse, block_out, block_lse, # type: ignore
                slice_=(slice(None), slice(block_seq_len, None))
            )
            
        if step + 1 != world_size:
            comm.wait() # type: ignore
            curr_k, curr_v = next_k, next_v # type: ignore

    if out is None:
        out = torch.zeros_like(q)
    if lse is None:
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device, dtype=torch.float32)
    else:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        
    out = out.to(q.dtype)
    return out, lse

def xdit_ring_zigzag_flash_attn_func_patched(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
    alibi_slopes=None, deterministic=False, return_attn_probs=False, group=None,
    attn_type=None, attn_processor=None, attn_layer=None, joint_tensor_key=None,
    joint_tensor_value=None, joint_strategy="none", mask=None, **kwargs,
):
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    out, softmax_lse = xdit_ring_zigzag_flash_attn_forward_patched(
        group, q, k, v, softmax_scale=softmax_scale, dropout_p=dropout_p,
        causal=causal, window_size=window_size, alibi_slopes=alibi_slopes,
        deterministic=deterministic, attn_type=attn_type, attn_processor=attn_processor,
        attn_layer=attn_layer, joint_tensor_key=joint_tensor_key,
        joint_tensor_value=joint_tensor_value, joint_strategy=joint_strategy,
        mask=mask, **kwargs
    )
    return out if not return_attn_probs else (out, softmax_lse, None)


def xdit_ring_flash_attn_forward_patched(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    attn_type: Optional[Any] = None,
    attn_processor: Optional[nn.Module] = None,
    attn_layer: Optional[Any] = None,
    joint_tensor_key: Optional[torch.Tensor] = None,
    joint_tensor_value: Optional[torch.Tensor] = None,
    joint_strategy: str = "none",
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Patched xfuser ring attention forward that supports masks.
    """
    is_joint = (joint_tensor_key is not None and joint_tensor_value is not None)
    
    if RingComm is None:
        raise ImportError("yunchang is required for xfuser ring attention")
    comm = RingComm(process_group)
    rank = comm.rank
    world_size = comm.world_size

    out = None
    lse = None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()

    # OPT 1: Avoid GPU-CPU sync — use cached triviality check
    from raylight.distributed_modules.attention import check_mask_nontrivial
    has_nontrivial_mask = check_mask_nontrivial(mask)

    next_k = None
    next_v = None

    for step in range(world_size):
        if step + 1 != world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        # Handle joint
        block_k, block_v = k, v
        if is_joint:
            assert joint_tensor_key is not None and joint_tensor_value is not None
            if joint_strategy == "rear" and step + 1 == world_size:
                block_k = torch.cat([k, joint_tensor_key], dim=1)
                block_v = torch.cat([v, joint_tensor_value], dim=1)
            elif joint_strategy == "front" and step == 0:
                block_k = torch.cat([joint_tensor_key, k], dim=1)
                block_v = torch.cat([joint_tensor_value, v], dim=1)

        # Slice mask for current ring step
        mask_slice = None
        if has_nontrivial_mask:
            assert mask is not None
            q_len, k_len_block = q.shape[1], block_k.shape[1]
            q_start = rank * q_len
            recv_rank = (rank - step) % world_size
            k_start = recv_rank * k.shape[1]
            
            q_idx = slice(q_start, q_start + q_len) if mask.shape[2] > 1 else slice(None)
            k_idx = slice(k_start, k_start + k_len_block) if mask.shape[3] > 1 else slice(None)
            mask_slice = mask[:, :, q_idx, k_idx]

        if not causal or step <= rank:
            if mask_slice is not None:
                block_out, block_lse = manual_block_attention(
                    q, block_k, block_v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal and step == 0,
                    mask=mask_slice
                )
            else:
                if AttnType is None or select_flash_attn_impl is None:
                    raise ImportError("yunchang is required for xfuser ring attention without masks")
                _attn_type = getattr(AttnType, "FA") if attn_type is None else attn_type
                # type ignore because select_flash_attn_impl return type is dynamic and signature varies by stage
                fn: Any = select_flash_attn_impl(_attn_type, stage="fwd-only", attn_processor=attn_processor) # type: ignore
                block_out, block_lse = fn(
                    q, block_k, block_v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal and step == 0,
                    window_size=window_size,
                    softcap=0.0,
                    alibi_slopes=alibi_slopes,
                    return_softmax=True and dropout_p > 0,
                )

            if update_out_and_lse is None:
                raise ImportError("yunchang is required for xfuser ring attention updates")
            if block_lse is None:
                raise RuntimeError("LSE is None from block attention, but required for ring attention update.")
            out, lse = update_out_and_lse(out, lse, block_out, block_lse) # type: ignore

        if step + 1 != world_size:
            comm.wait()
            k = next_k # type: ignore
            v = next_v # type: ignore

    if out is None:
        out = torch.zeros_like(q)
    if lse is None:
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device, dtype=torch.float32)
    else:
        # OPT 6: Single LSE post-processing (was duplicated before)
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        
    out = out.to(q.dtype)
    return out, lse


# OPT 5: Removed autograd.Function wrapper — backward was not implemented,
# and save_for_backward was leaking Q/K/V/out/LSE memory under torch.no_grad().

def xdit_ring_flash_attn_func_patched(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type=None,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    mask=None,
    **kwargs,
):
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    out, softmax_lse = xdit_ring_flash_attn_forward_patched(
        group,
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        attn_type=attn_type,
        attn_processor=attn_processor,
        attn_layer=attn_layer,
        joint_tensor_key=joint_tensor_key,
        joint_tensor_value=joint_tensor_value,
        joint_strategy=joint_strategy,
        mask=mask,
    )
    return out if not return_attn_probs else (out, softmax_lse, None)
