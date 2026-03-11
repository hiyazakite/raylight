"""
NOTE: code from yunchang
"""

import torch
import os
import logging

logger = logging.getLogger(__name__)
from yunchang.ring.utils import RingComm, update_out_and_lse




from raylight.distributed_modules.compact.utils import COMPACT_COMPRESS_TYPE
from raylight.distributed_modules.compact.context import compact_config, compact_cache
from raylight.distributed_modules.compact.ops import compact_compress, compact_decompress
from raylight.distributed_modules.compact.prof import Profiler
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask, AuxRequest
    _HAS_FLEX_ATTN = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    AuxRequest = None
    _HAS_FLEX_ATTN = False

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None

try:
    import xformers
    import xformers.ops
    from xformers.ops import memory_efficient_attention_forward, LowerTriangularMask
except ImportError:
    xformers = None
    memory_efficient_attention_forward = None
    LowerTriangularMask = None

_VERBOSE_ATTN = os.getenv("RAYLIGHT_VERBOSE_ATTN", "0") == "1"

def compact_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    return_attn_probs=None,
    deterministic=False,
    attn_layer=None,
    group=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    mod_idx=None,
    current_iter=None,
    mask=None,
    key_suffix="",
):
    """
    Compact ring attention forward pass.
    """
    if _VERBOSE_ATTN:
        mask_info = f"Shape={mask.shape}" if mask is not None else "None"
        print(f"📦 [Raylight] compact_fwd called (Layer: {mod_idx}, Iter: {current_iter}, mask={mask_info})")
    # Trust that higher-level logic (usp_dit_forward) has already 
    # stripped trivial masks to avoid GPU-CPU sync here.
    from xfuser.core.distributed import get_sequence_parallel_rank
    get_sequence_parallel_rank() # Just to ensure it's imported

    config = compact_config()
    gather = config.override_with_patch_gather_fwd if config else False
    
    if gather:
        from raylight.distributed_modules.compact.patchpara.fwd import patch_gather_fwd
        return patch_gather_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter,
            mask=mask, key_suffix=key_suffix
        )
    else:
        return _compact_ring_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter,
            mask=mask, key_suffix=key_suffix
        )

# env var USE_AWL
AWL=os.getenv("USE_AWL", "0") == "1"
AWL_RAND=False

def compact_update_awl_scale(q, k, v):
    # (bs, seq_len, head_cnt, head_size)
    """
    Calculates key token importance by sampling queries and computing attention scores.

    Args:
        q: Query tensor (bs, seq_len, head_cnt, head_size)
        k: Key tensor (bs, seq_len, head_cnt, head_size)
    """
    from raylight.distributed_modules.compact.slowpath import set_current_lowrank_scale
    if not AWL:
        return
    with torch.no_grad(): # No need to track gradients for importance calculation
        if not AWL_RAND:
            bs, seq_len, head_cnt, head_size = q.shape
            # q_2d = q.view(bs * seq_len, head_cnt * head_size)
            # q_chan_mean = torch.mean(q_2d.abs(), dim=0).flatten().float()
            # q_chan_mean = q_chan_mean / q_chan_mean.norm()
            # lowrank_scale_k = q_chan_mean
            # print percentile
            # print(f"q_chan_mean 1%: {q_chan_mean.quantile(0.01):.4f}, 99%: {q_chan_mean.quantile(0.99):.4f}")
            v_2d = v.view(bs * seq_len, head_cnt * head_size)
            v_norm = torch.norm(v_2d, dim=-1).flatten()
            # smaller the v norm, typically larger the attn score
            lowrank_scale_k = v_norm.mean() / v_norm
            lowrank_scale_k = lowrank_scale_k.flatten()
            set_current_lowrank_scale(lowrank_scale_k, None)
            return
        else:
            bs, seq_len, head_cnt, head_size = q.shape
            random_scale = torch.ones(bs * seq_len)
            # Randomly select 10% of the elements and set them to 10
            mask = torch.zeros(bs * seq_len, device=q.device, dtype=torch.bool)
            num_elements = bs * seq_len
            num_to_select = int(0.1 * num_elements)  # 10% of elements
            # Get random indices
            indices = torch.randperm(num_elements, device=q.device)[:num_to_select]
            mask.scatter_(0, indices, True)
            # Create a tensor with 10s where mask is True, and original values elsewhere
            random_scale = torch.where(mask, torch.tensor(10.0, device=q.device), torch.ones_like(mask, device=q.device, dtype=torch.float32))
            set_current_lowrank_scale(random_scale, random_scale)



def compact_attn_forward(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, mask=None
):
    """
    Custom attention forward using SDPA or xformers to support masking in CompactFusion.
    """
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    # FlexAttention is the fastest and most flexible for complex masks
    if _HAS_FLEX_ATTN and mask is not None:
        if _VERBOSE_ATTN:
            print("[Raylight] Using FlexAttention for masked attention.")
        
        # FlexAttention expects a score_mod or mask_mod.
        # If mask is [B, 1, Q, K], we can pass it as a bias via score_mod.
        def score_mod(score, b, h, q_idx, kv_idx):
            # Use h % mask.shape[1] to handle broadcasting across heads safely in Triton
            return score + mask[b, h % mask.shape[1], q_idx, kv_idx]

        # Use torch.compile for the fused kernel. 
        # Note: We don't want to compile every time, so we wrap it.
        # For Ring Attention, we also need LSE.
        # If flex_attention doesn't return LSE directly, we can get it by 
        # using SDPA for LSE only, which is still faster than materializing 
        # the full scores for 'out'.
        
        if flex_attention is None or AuxRequest is None:
            raise ImportError("FlexAttention components not available despite _HAS_FLEX_ATTN being True")

        logger.info("[Raylight] compact_attn_forward: Using FlexAttention")

        # In Ring Attention, q is (bs, seq_len/N, head_cnt, head_size) or (B, H, L, D_h).
        # We need to ensure they match (B, H, S, D).
        if q.dim() == 4:
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
            raise ValueError(f"Expected 4D query, got {q.dim()}D")
        
        # Call flex_attention directly (avoiding torch.compile OOMs on some hardware limits)
        flex_fn = flex_attention
        if flex_fn is None:
            raise ImportError("FlexAttention not available")
        
        # Use AuxRequest to get LSE natively from the kernel
        res = flex_fn(q_flex, k_flex, v_flex, score_mod=score_mod, return_aux=AuxRequest(lse=True))
        
        # flex_attention returns (out, aux_outputs) when return_aux is specified
        out, aux = res
        lse = getattr(aux, "lse", None)
        
        # flex_attention 'out' is (B, H, S, D). Transpose back to (B, S, H, D) for Ring Attention.
        out = out.transpose(1, 2).contiguous()
        # lse is (B, H, S). Ring attention code usually wants it in a specific layout
        # for update_out_and_lse. We'll keep it as (B, H, S) here as it's the most common internal format.
        return out, lse

    # xformers is very fast for masked attention and returns LSE
    if xformers is not None and memory_efficient_attention_forward is not None:
        # q, k, v are (bs, seq_len, head_cnt, head_size)
        # xformers expects (B, L, H, D)
        
        # Handle causal mask if needed
        attn_bias = mask
        if causal:
            if attn_bias is None:
                if LowerTriangularMask is not None:
                    attn_bias = LowerTriangularMask()
                else:
                    # Fallback if LowerTriangularMask is missing but xformers is active
                    pass
            else:
                # Merge causal into existing mask if needed
                pass

        print("[Raylight] Using xformers for masked attention.")

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
            out, lse = res, None # LSE might not be returned by some xformers versions/ops
        return out, lse

    logger.info("[Raylight] Falling back to manual PyTorch attention (xformers not available). Using SDPA.")

    # Optimized fallback using SDPA (Memory efficient for 'out')
    q_t = q.transpose(1, 2) # (bs, heads, q_len, dim)
    k_t = k.transpose(1, 2) # (bs, heads, k_len, dim)
    v_t = v.transpose(1, 2) # (bs, heads, k_len, dim)
    
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal and mask is None,
        scale=softmax_scale
    )
    
    # SDPA doesn't return LSE directly. For ring attention online softmax, LSE is MANDATORY.
    # To avoid O(N^2) memory bottlenecks during high-res upscaling, we chunk the LSE calculation.
    bs, heads, q_len, dim = q_t.shape
    k_len = k_t.shape[2]
    
    # Use a chunk size that fits comfortably in VRAM
    # If k_len is huge, we need smaller Q-chunks.
    CHUNK_SIZE = max(1, 1024 * 32 // k_len) if k_len > 0 else 1024
    lse = torch.empty(bs, heads, q_len, device=q.device, dtype=torch.float32)
    
    with torch.no_grad():
        kt_v = k_t.transpose(-2, -1)
        for i in range(0, q_len, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, q_len)
            q_chunk = q_t[:, :, i:end, :]
            
            # Compute raw attention scores for this chunk: (bs, heads, chunk_len, k_len)
            attn_chunk = torch.matmul(q_chunk, kt_v) * softmax_scale
            
            if mask is not None:
                if mask.shape[2] > 1:
                    m_chunk = mask[:, :, i:end, :]
                else:
                    m_chunk = mask
                attn_chunk.add_(m_chunk)
            
            if causal:
                # Optimized causal mask for the local block
                row_indices = torch.arange(i, end, device=q.device).view(1, 1, -1, 1)
                col_indices = torch.arange(k_len, device=q.device).view(1, 1, 1, -1)
                attn_chunk.masked_fill_(row_indices < col_indices, float("-inf"))
            
            # logsumexp over the K-dimension (dim=-1)
            lse[:, :, i:end] = torch.logsumexp(attn_chunk.float(), dim=-1)
            del attn_chunk
            
    out = out.transpose(1, 2).contiguous()
    return out, lse

@Profiler.prof_func("compact._compact_ring_fwd")
def _compact_ring_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    return_attn_probs=None,
    deterministic=False,
    attn_layer=None,
    group=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    mod_idx=None,
    current_iter=None,
    mask=None,
    key_suffix="",
):
    if _VERBOSE_ATTN:
        mask_info = f"Shape={mask.shape}" if mask is not None else "None"
        print(f"🐳 [Raylight] _compact_ring_fwd loop (Layer: {mod_idx}, Step: {current_iter}, Q: {q.shape}, mask={mask_info})")
    # (bs, seq_len, head_cnt, head_size)
    assert alibi_slopes is None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    is_joint = False
    if (joint_tensor_key is not None and 
        joint_tensor_value is not None):
        supported_joint_strategy = ["front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        else:
            is_joint = True
    elif (joint_tensor_key is None and 
        joint_tensor_value is None):
        pass
    else:
        raise ValueError(
            "joint_tensor_key and joint_tensor_value should be None or not None simultaneously."
        )
    if attn_layer is not None:
        # XXX: we dont need KV Cache in ring
        # k, v = get_cache_manager().update_and_get_kv_cache(
        #     new_kv=[k, v],
        #     layer=attn_layer,
        #     slice_dim=1,
        #     layer_type="attn",
        # )
        # k = k.contiguous()
        # v = v.contiguous()
        pass
    process_group = group if group is not None else torch.distributed.group.WORLD
    from typing import cast
    comm = RingComm(cast(torch.distributed.ProcessGroup, process_group))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = None
    lse = None

    # PERFORMANCE: Detect trivial mask once for the entire global mask.
    # If the global mask is all zeros, we can use the FlashAttention fast-path for all blocks.
    global_trivial_mask = False
    if mask is not None:
        # Singleton zero or empty mask is trivial
        if mask.numel() == 1:
            global_trivial_mask = (mask == 0).all()
        # Avoid full scan for huge masks if possible, or do it once.
        # For LTX 2.3, masks are usually manageable in size.
        elif (mask == 0).all():
            global_trivial_mask = True

    config = compact_config()
    if config:
        compress_type_k = config.compress_func(mod_idx, current_iter)
        compress_type_v = config.compress_func(mod_idx, current_iter)
    else:
        compress_type_k = COMPACT_COMPRESS_TYPE.IDENTITY
        compress_type_v = COMPACT_COMPRESS_TYPE.IDENTITY
    
    # Pre-calculate cache keys and ranks
    world_size = comm.world_size
    rank = comm.rank
    
    from xfuser.core.distributed import get_sequence_parallel_rank
    sp_rank = get_sequence_parallel_rank()
    q_len = q.shape[1]
    q_start = sp_rank * q_len
    # Pre-format base keys to avoid redundant f-string formatting in loops
    my_rank_mod = rank % world_size
    k_my_cache_key = f"{mod_idx}-{my_rank_mod}-k{key_suffix}"
    v_my_cache_key = f"{mod_idx}-{my_rank_mod}-v{key_suffix}"
    
    # Pre-calculate all recv_ranks and their cache keys
    recv_ranks = [(rank - s) % world_size for s in range(world_size)]
    k_recv_keys = [f"{mod_idx}-{r}-k{key_suffix}" for r in recv_ranks]
    v_recv_keys = [f"{mod_idx}-{r}-v{key_suffix}" for r in recv_ranks]

    original_k_shape = k.shape 
    original_v_shape = v.shape
    k_to_send = compact_compress(k_my_cache_key, k, compress_type_k, update_cache=True)
    v_to_send = compact_compress(v_my_cache_key, v, compress_type_v, update_cache=True)
    
    # Ensure dtypes match for kernels once before loop
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    for step in range(world_size):
        buf_k: torch.Tensor = torch.empty(0)
        buf_v: torch.Tensor = torch.empty(0)
        if step + 1 != world_size:
            buf_k = comm.send_recv(k_to_send)
            buf_v = comm.send_recv(v_to_send)
            comm.commit()
        
        if step != 0:
            k_recv_cache_key = k_recv_keys[step]
            v_recv_cache_key = v_recv_keys[step]
            k = compact_decompress(
                k_recv_cache_key, k_to_send, compress_type_k, original_k_shape, update_cache=True
            ).contiguous()
            if k.dtype != q.dtype:
                k = k.to(q.dtype)
            v = compact_decompress(
                v_recv_cache_key, v_to_send, compress_type_v, original_v_shape, update_cache=True
            ).contiguous()
            if v.dtype != q.dtype:
                v = v.to(q.dtype)

        if is_joint and joint_strategy == "rear":
            if step + 1 == comm.world_size:
                assert joint_tensor_key is not None and joint_tensor_value is not None
                key_to_use = torch.cat([k, joint_tensor_key], dim=1)
                value_to_use = torch.cat([v, joint_tensor_value], dim=1)
            else:
                key_to_use, value_to_use = k, v
        elif is_joint and joint_strategy == "front":
            if step == 0:
                assert joint_tensor_key is not None and joint_tensor_value is not None
                key_to_use = torch.cat([joint_tensor_key, k], dim=1)
                value_to_use = torch.cat([joint_tensor_value, v], dim=1)
            else:
                key_to_use, value_to_use = k, v
        else:
            key_to_use, value_to_use = k, v
        # (bs, seq_len, head_cnt, head_size)
        # (bs, seq_len, head_cnt, head_size)
        if not causal or step <= comm.rank:
            # Calculate mask slice for the current K block
            # mask handling
            mask_slice = None
            if mask is not None:
                # mask: [B, 1, 1, seq_len] or [B, 1, seq_len, seq_len]
                # Handle broadcastable masks (singleton dimensions)
                q_len = q.shape[1]
                k_len_block = k.shape[1] # Length of the current K block
                
                # Calculate global start/end for the current Q and K blocks
                recv_rank = (comm.rank - step) % comm.world_size
                q_start = comm.rank * q_len
                k_start = recv_rank * k_len_block
                
                # Slicing:
                # If mask is [B, 1, 1, S_global], we slice dim 3.
                # If mask is [B, 1, S_global, S_global], we slice dim 2 (Q) and dim 3 (K).
                # Note: Q is ALWAYS local in our Ring implementation (only K rotates).
                q_idx = slice(q_start, q_start + q_len) if (len(mask.shape) > 2 and mask.shape[2] > 1) else slice(None)
                k_idx = slice(k_start, k_start + k_len_block) if (len(mask.shape) > 3 and mask.shape[3] > 1) else slice(None)
                
                mask_slice = mask[:, :, q_idx, k_idx]

            # PERFORMANCE: Trivial Mask Fast-Path
            # Using pre-calculated global_trivial_mask to avoid synchronous inner-loop .all() calls.
            is_trivial_mask = global_trivial_mask
            if not is_trivial_mask and mask_slice is not None:
                # Double check for singleton mask if it was not globally trivial
                if mask_slice.numel() == 1:
                    is_trivial_mask = (mask_slice == 0).all()

            if (mask is not None and not is_trivial_mask) or flash_attn is None:
                if step == 0 and not is_trivial_mask:
                    if _VERBOSE_ATTN:
                        logger.info(f"[Raylight] Side-stepping FlashAttn: mask={mask is not None}, is_trivial={is_trivial_mask}, flash_attn={flash_attn is not None}")
                
                logger.info(f"[Raylight] _compact_ring_fwd (step {step}): Using compact_attn_forward (Masked/Fallback)")
                
                # Use custom manual attention if mask is present and non-trivial or FlashAttn unavailable
                # This ensures we respect the mask (Critical for padding/mangled output issues)
                block_out, block_lse = compact_attn_forward(
                    q,
                    key_to_use,
                    value_to_use,
                    dropout_p,
                    softmax_scale,
                    causal=causal and step == 0,
                    mask=mask_slice
                )
            else:
                if step == 0:
                    logger.info(f"[Raylight] _compact_ring_fwd (step {step}): Using FlashAttention")
                assert _flash_attn_forward is not None
                v_str = getattr(flash_attn, "__version__", "0.0.0")
                
                # Robust version check for flash_attn signature
                # 2.7.0+ changed signature to use separate window_size params and return 4 values
                use_legacy = False
                if v_str == "0.0.0":
                    use_legacy = False # Assume modern if version unknown
                else:
                    try:
                        v_parts = [int(p) for p in v_str.split('.') if p.isdigit()]
                        if v_parts and v_parts[0] < 2:
                            use_legacy = True
                        elif v_parts and v_parts[0] == 2 and len(v_parts) > 1 and v_parts[1] <= 6:
                            # 2.6.3 and below
                            use_legacy = True
                    except Exception:
                        use_legacy = False

                # Use kwargs to avoid static analysis errors for different versions of flash_attn
                fa_kwargs = {
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "softcap": 0.0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
                
                if use_legacy: 
                    # 2.6.3 and below use 'window_size'
                    fa_kwargs["window_size"] = window_size
                    fa_res = _flash_attn_forward(q, key_to_use, value_to_use, **fa_kwargs)  # type: ignore
                    block_out = fa_res[0]  # type: ignore
                    block_lse = fa_res[5]  # type: ignore
                else:
                    # 2.7.0+ use 'window_size_left' and 'window_size_right'
                    fa_kwargs["window_size_left"] = window_size[0]
                    fa_kwargs["window_size_right"] = window_size[1]
                    fa_res = _flash_attn_forward(q, key_to_use, value_to_use, **fa_kwargs)
                    block_out = fa_res[0]
                    block_lse = fa_res[1]
            if update_out_and_lse is None:
                raise ImportError("yunchang not available, cannot update_out_and_lse")
            if block_lse is None:
                raise RuntimeError("LSE is None from block attention, but required for ring attention update.")
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            with Profiler.scope("compact.ring.wait"):
                comm.wait()
            k_to_send = buf_k 
            v_to_send = buf_v
    
    if out is None:
        out = torch.zeros_like(q)
    if lse is None:
        # (bs, head, seq, 1) or (bs, head, block_seq)
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], 1, device=q.device, dtype=torch.float32)

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    if config and config.check_cache_consistency:
        cache = compact_cache()
        if cache:
            cache.check_consistency(group=process_group)
    return out, lse, None