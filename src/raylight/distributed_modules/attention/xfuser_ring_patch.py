import torch
import torch.nn.functional as F
import math
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
            # Use h % mask.shape[1] to handle broadcasting across heads safely in Triton
            return score + mask[b, h % mask.shape[1], q_idx, kv_idx]

        if flex_attention is None or AuxRequest is None:
            raise ImportError("FlexAttention components not available despite _HAS_FLEX_ATTN being True")

        logger.info("[Raylight] manual_block_attention: Using FlexAttention")

        # We need to ensure they match (B, H, S, D).
        if q.dim() == 3:
            # (B, L, H*D) -> (B, L, H, D) -> (B, H, L, D)
            b, l, dim = q.shape
            
            # Identify heads from k shape or context if possible, 
            # normally we'd pass heads, but we can try to guess from mask
            # mask is (B, H, Q_L, K_L)
            h = mask.shape[1]
            d = dim // h
            
            q_flex = q.view(b, l, h, d).transpose(1, 2)
            k_flex = k.view(b, k.shape[1], h, d).transpose(1, 2)
            v_flex = v.view(b, v.shape[1], h, d).transpose(1, 2)
        elif q.dim() == 4:
            if q.shape[2] == q.shape[1] or q.shape[2] > q.shape[1]: 
                # Already (B, H, S, D) or something similar. 
                # Check for (bs, seq, heads, dim). If seq > heads, it's (B, S, H, D)
                if q.shape[1] > getattr(q, 'shape')[2]: # usually S > H
                    q_flex = q.transpose(1, 2)
                    k_flex = k.transpose(1, 2)
                    v_flex = v.transpose(1, 2)
                else: # already B, H, S, D
                    q_flex = q
                    k_flex = k
                    v_flex = v
            else: # (B, S, H, D)
                q_flex = q.transpose(1, 2)
                k_flex = k.transpose(1, 2)
                v_flex = v.transpose(1, 2)
        else:
            raise ValueError(f"Expected 3D or 4D query, got {q.dim()}D")        
        # Call flex_attention directly (avoiding torch.compile OOMs on some hardware limits)
        flex_fn = flex_attention
        if flex_fn is None:
            raise ImportError("FlexAttention not available")
        
        # Use AuxRequest to get LSE natively from the kernel
        res = flex_fn(q_flex, k_flex, v_flex, score_mod=score_mod, return_aux=AuxRequest(lse=True))
        
        out, aux = res
        lse = getattr(aux, "lse", None)
        
        # Transpose back to (B, S, H, D)
        out = out.transpose(1, 2).contiguous()
        return out, lse

    # 2. Try xformers
    if xformers is not None and memory_efficient_attention_forward is not None:
        attn_bias = mask
        if causal:
            if attn_bias is None:
                if LowerTriangularMask is not None:
                    attn_bias = LowerTriangularMask()
                else:
                    pass
            else:
                pass

        logger.info("[Raylight] Using xformers for masked block attention.")

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
    logger.info("[Raylight] Falling back to manual PyTorch attention for block (xformers not available). Using SDPA.")
    
    q_t = q.transpose(1, 2) # (B, H, L, D)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal and mask is None,
        scale=softmax_scale
    )
    
    # Calculate LSE in chunks to avoid O(N^2) memory spikes
    bs, heads, q_len, dim = q_t.shape
    k_len = k_t.shape[2]
    CHUNK_SIZE = max(1, 1024 * 32 // k_len) if k_len > 0 else 1024
    lse = torch.empty(bs, heads, q_len, device=q.device, dtype=torch.float32)
    
    with torch.no_grad():
        kt_v = k_t.transpose(-2, -1)
        for i in range(0, q_len, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, q_len)
            q_chunk = q_t[:, :, i:end, :]
            attn_chunk = torch.matmul(q_chunk, kt_v) * softmax_scale
            
            if mask is not None:
                # Slice mask if it's not a singleton broadcast
                m_chunk = mask[:, :, i:end, :] if mask.shape[2] > 1 else mask
                attn_chunk.add_(m_chunk)
            
            if causal:
                row_indices = torch.arange(i, end, device=q.device).view(1, 1, -1, 1)
                col_indices = torch.arange(k_len, device=q.device).view(1, 1, 1, -1)
                attn_chunk.masked_fill_(row_indices < col_indices, float("-inf"))
            
            lse[:, :, i:end] = torch.logsumexp(attn_chunk.float(), dim=-1)
            del attn_chunk
            
    return out.transpose(1, 2).contiguous(), lse

def xdit_ring_flash_attn_forward_patched(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    attn_type=None,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    q_descale=None,
    k_descale=None,
    v_descale=None,
    # Added mask support
    mask=None,
):
    """
    Patched xfuser ring attention forward that supports masks.
    """
    is_joint = False
    if (joint_tensor_key is not None and joint_tensor_value is not None):
        is_joint = True
    
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

    # Pre-check for trivial mask
    is_trivial_mask = False
    if mask is not None:
        is_trivial_mask = (mask == 0).all()
        if is_trivial_mask:
            logger.info("[Raylight] xdit_ring_flash_attn_forward: Mask is trivial (all zeros), using fast path")

    next_k = None  # type: ignore
    next_v = None  # type: ignore

    for step in range(world_size):
        if step + 1 != world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
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

        # Slice mask
        mask_slice = None
        if mask is not None and not is_trivial_mask:
            q_len, k_len_block = q.shape[1], block_k.shape[1]
            q_start = rank * q_len
            recv_rank = (rank - step) % world_size
            k_start = recv_rank * k.shape[1] # original block length
            
            q_idx = slice(q_start, q_start + q_len) if mask.shape[2] > 1 else slice(None)
            k_idx = slice(k_start, k_start + k_len_block) if mask.shape[3] > 1 else slice(None)
            mask_slice = mask[:, :, q_idx, k_idx]

        if not causal or step <= rank:
            if mask_slice is not None:
                logger.info(f"[Raylight] xdit_ring_flash_attn_forward (rank {rank}, step {step}): Using manual_block_attention (Masked/Fallback)")
                # Manual fallback for masked block
                block_out, block_lse = manual_block_attention(
                    q, block_k, block_v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal and step == 0,
                    mask=mask_slice
                )
            else:
                if step == 0:
                    logger.info(f"[Raylight] xdit_ring_flash_attn_forward (rank {rank}, step {step}): Using FlashAttention")
                if AttnType is None or select_flash_attn_impl is None:
                    raise ImportError("yunchang is required for xfuser ring attention without masks")
                # FlashAttn
                _attn_type = getattr(AttnType, "FA") if attn_type is None else attn_type
                fn = select_flash_attn_impl(_attn_type, stage="fwd-only", attn_processor=attn_processor)  # type: ignore
                block_out, block_lse = fn(  # type: ignore
                    q, block_k, block_v,
                    dropout_p=dropout_p,  # type: ignore
                    softmax_scale=softmax_scale,  # type: ignore
                    causal=causal and step == 0,  # type: ignore
                    window_size=window_size,  # type: ignore
                    softcap=0.0,  # type: ignore
                    alibi_slopes=alibi_slopes,  # type: ignore
                    return_softmax=True and dropout_p > 0,  # type: ignore
                )

            if update_out_and_lse is None:
                raise ImportError("yunchang is required for xfuser ring attention updates")
                
            # Type guard for block_lse being None from xformers/flex_attention fallback paths
            if block_lse is None:
                # If a fallback doesn't provide LSE, we can't reliably do ring attention update
                raise RuntimeError("LSE is None from block attention, but required for ring attention update.")
            
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != world_size:
            comm.wait()
            k = next_k
            v = next_v

    if out is None:
        out = torch.zeros_like(q)
    if lse is None:
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], device=q.device, dtype=torch.float32)
    else:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

class xFuserRingFlashAttnFuncPatched(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
        mask=None,
    ):
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.size(-1))

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
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
            mask=mask,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("Backward not implemented for patched ring attention.")

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
    return xFuserRingFlashAttnFuncPatched.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
        mask,
    )
