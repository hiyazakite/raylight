"""
NOTE: code from yunchang
"""

import torch



from yunchang.ring.utils import RingComm, update_out_and_lse




from raylight.distributed_modules.compact.utils import COMPACT_COMPRESS_TYPE
from raylight.distributed_modules.compact.main import (
    compact_config,
    compact_cache,
    compact_compress,
    compact_decompress,
)
from raylight.distributed_modules.compact.prof import Profiler

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None


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
):
    """
    Compact ring attention forward pass.
    """
    config = compact_config()
    gather = config.override_with_patch_gather_fwd if config else False
    
    if gather:
        from raylight.distributed_modules.compact.patchpara.fwd import patch_gather_fwd
        return patch_gather_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter,
            mask=mask
        )
    else:
        return _compact_ring_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter
        )

# env var USE_AWL
import os
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

import torch.nn.functional as F

def compact_attn_forward(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, mask=None
):
    """
    Custom attention forward using SDPA to support masking in CompactFusion.
    """
    # SDPA expects query (N, ..., L, E), key (N, ..., S, E), value (N, ..., S, E)
    # q, k, v are (bs, seq_len, head_cnt, head_size) -> (bs, head_cnt, seq_len, head_size)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Handle mask broadcasting
    # mask is (B, 1, 1, S_block) or similar.
    # SDPA attn_mask: (B, H, L, S) or (B, 1, L, S).
    # If mask is boolean, user expects True=Keep? Or True=Mask?
    # ComfyUI usually uses a Bias Mask (0, -inf) or Boolean (True/False).
    # SDPA documentation: 
    #   attn_mask (optional): binary mask (True for ignore) or float mask (bias).
    # We should perform basic check or just pass it if compatible.
    
    # Handle causal
    # If `causal` is True, SDPA handles it. If `mask` is also provided, SDPA combines them?
    # Only if mask is None, is_causal=True works best.
    # If mask is provided, is_causal usually must be False and mask includes causal?
    # But here `causal` argument handles Ring causal logic.
    # If `causal` (Step 0), we want causal behavior.
    
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    # If mask is provided, we merge causal into mask or rely on SDPA to handle both?
    # SDPA: "is_causal and attn_mask cannot be set at the same time" (in some versions).
    # We should use manual causal mask if mask is present.
    
    if mask is not None and causal:
        pass
        
        
    # Fallback to simple SDPA without is_causal if mask is present (assuming mask handles it or not needed?)
    # But `causal` here means "Lower Triangular".
    # If `attn_mask` is passed, we must manually apply causal if needed.
    
    # Let's just try running SDPA.
    # If causal=True and mask is not None, we must Construct a Causal Mask and merge it.
    
    op_causal = causal
    op_mask = mask
    
    if causal and op_mask is not None:
        assert isinstance(op_mask, torch.Tensor)
        op_causal = False
        # Merge causal mask into op_mask
        L, S = q.size(-2), k.size(-2)
        c_mask = torch.ones(L, S, device=q.device, dtype=torch.bool).tril().view(1, 1, L, S)
        # Convert c_mask to bias (0, -inf) match op_mask type?
        # Assume op_mask is Bias (float).
        if op_mask.dtype == torch.bool:
             # Boolean mask.
             op_mask = op_mask & c_mask
        else:
             # Additive mask. 0 for keep, -inf for mask.
             # c_mask (True keep, False mask) -> (0, -inf)
             c_bias = torch.zeros_like(op_mask)
             c_bias.masked_fill_(~c_mask, float("-inf"))
             op_mask = op_mask + c_bias

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=op_mask,
        dropout_p=dropout_p,
        is_causal=op_causal,
        scale=softmax_scale
    )
    
    out = out.transpose(1, 2) # Back to (bs, seq, head, head_size)
    
    # Return out, lse (stub LSE as None or 0? Ring needs LSE for correction!)
    # SDPA does NOT return LSE.
    # !! CRITICAL !!
    # Ring Attention requires LSE (LogSumExp) to combine blocks!
    # SDPA returns only Output.
    # If we use SDPA, we cannot act as a Ring Block!
    # We obtain the final output for this block, but we cannot merge it with previous blocks correctly regarding Softmax normalization!
    
    # Ring Algorithm:
    # Out_new = (Out_old * exp(LSE_old - LSE_new) + Out_block * exp(LSE_block - LSE_new))
    # We NEED LSE_block.
    
    # If SDPA doesn't return LSE, we CANNOT use it for Ring Attention.
    # We must use a kernel that returns LSE (like flash_attn_func or yunchang's) or Compute it manually.
    
    # Manual Computation (Slow but returns LSE):
    # Attn = (Q @ K.T) * scale
    # LSE = logsumexp(Attn + mask)
    # Out = Softmax(Attn) @ V
    
    # Since we need Correctness for "Mangled Output" diagnosis, Slow Manual Implementation is acceptable.
    # AND "CompactFusion" is mostly for sequence extension.
    # If we fallback to Pytorch manual, it works.
    
    # Re-implement manual attention here.
    attn = (q @ k.transpose(-2, -1)) * softmax_scale
    if op_mask is not None:
         attn = attn + op_mask # Assume additive mask
    if op_causal:
         L, S = q.size(-2), k.size(-2)
         c_mask = torch.ones(L, S, device=q.device, dtype=torch.bool).tril()
         attn = attn.masked_fill(~c_mask, float("-inf"))
    
    lse = torch.logsumexp(attn, dim=-1, keepdim=True)
    attn_probs = torch.exp(attn - lse)
    # Dropout?
    if dropout_p > 0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        
    out = attn_probs @ v
    out = out.transpose(1, 2)
    lse = lse.squeeze(-1) # (bs, head, seq)
    # lse shape in ring: (bs, head, seq)? 
    # _compact_ring_fwd expects LSE: (bs, seq, head) -> line 281: lse = lse.squeeze(dim=-1).transpose(1, 2)
    # If my lse is (bs, head, seq, 1) -> squeeze -> (bs, head, seq).
    # Ring expects (bs, head, seq) inside loop?
    # update_out_and_lse expects (bs, head, seq, 1)?
    # Let's match yunchang.kernsl.attention.pytorch_attn_forward return signature.
    # It returns block_out, block_lse.
    
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
):
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
    process_group = group
    comm = RingComm(process_group)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = None
    lse = None

    config = compact_config()
    if config:
        compress_type_k = config.compress_func(mod_idx, current_iter)
        compress_type_v = config.compress_func(mod_idx, current_iter)
    else:
        compress_type_k = COMPACT_COMPRESS_TYPE.IDENTITY
        compress_type_v = COMPACT_COMPRESS_TYPE.IDENTITY
    
    # Cache keys match reference implementation (no shape in key)
    # Shape mismatches are handled gracefully in get_base by returning None
    k_my_cache_key = f"{mod_idx}-{comm.rank%comm.world_size}-k"
    v_my_cache_key = f"{mod_idx}-{comm.rank%comm.world_size}-v"
    original_k_shape = k.shape 
    original_v_shape = v.shape
    k_to_send = compact_compress(k_my_cache_key, k, compress_type_k, update_cache=True)
    v_to_send = compact_compress(v_my_cache_key, v, compress_type_v, update_cache=True)
    
    for step in range(comm.world_size):
        buf_k: torch.Tensor = torch.empty(0)
        buf_v: torch.Tensor = torch.empty(0)
        if step + 1 != comm.world_size:
            buf_k = comm.send_recv(k_to_send)
            buf_v = comm.send_recv(v_to_send)
            comm.commit()
        
        if step != 0:
            recv_rank = (comm.rank - step) % comm.world_size
            k_recv_cache_key = f"{mod_idx}-{recv_rank}-k"
            v_recv_cache_key = f"{mod_idx}-{recv_rank}-v"
            k = compact_decompress(
                k_recv_cache_key, k_to_send, compress_type_k, original_k_shape, update_cache=True
            )
            v = compact_decompress(
                v_recv_cache_key, v_to_send, compress_type_v, original_v_shape, update_cache=True
            )
        k = k.contiguous() 
        v = v.contiguous()

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
            # Ensure dtypes match for flash_attn
            if key_to_use.dtype != q.dtype:
                key_to_use = key_to_use.to(q.dtype)
            if value_to_use.dtype != q.dtype:
                value_to_use = value_to_use.to(q.dtype)

            # Calculate mask slice for the current K block
            # mask handling
            mask_slice = None
            if mask is not None:
                # mask: [B, 1, 1, seq_len] or [B, 1, seq_len, seq_len]
                # Handle broadcastable masks (singleton dimensions)
                q_len = q.shape[1]
                k_len_block = k.shape[1] # Length of the current K block
                
                # Calculate global start/end for the current K block
                recv_rank = (comm.rank - step) % comm.world_size
                # Since Ring rotates K, we need to know which global chunk we are looking at.
                # seq_len_local = k_len_block (assuming all chunks equal size)
                k_start = recv_rank * k_len_block
                
                # Slicing:
                # If mask is [B, 1, 1, S_global], we slice dim 3.
                # If mask is [B, 1, S_global, S_global], we slice dim 2 (Q) and dim 3 (K).
                # Note: Q is ALWAYS local in our Ring implementation (only K rotates).
                # So Q index is always [sp_rank * q_len : (sp_rank + 1) * q_len].
                from xfuser.core.distributed import get_sequence_parallel_rank
                sp_rank = get_sequence_parallel_rank()
                q_start = sp_rank * q_len
                
                q_idx = slice(q_start, q_start + q_len) if mask.shape[2] > 1 else slice(None)
                k_idx = slice(k_start, k_start + k_len_block) if mask.shape[3] > 1 else slice(None)
                
                mask_slice = mask[:, :, q_idx, k_idx]

            if mask is not None or flash_attn is None:
                # Use custom manual attention if mask is present or FlashAttn unavailable
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

                if use_legacy: 
                    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                        q,
                        key_to_use,
                        value_to_use,
                        dropout_p,
                        softmax_scale,
                        causal=causal and step == 0,
                        window_size=window_size,
                        softcap=0.0,
                        alibi_slopes=alibi_slopes,
                        return_softmax=True and dropout_p > 0,
                    )
                else:
                    block_out, block_lse, _, _ = _flash_attn_forward(
                        q,
                        key_to_use,
                        value_to_use,
                        dropout_p,
                        softmax_scale,
                        causal=causal and step == 0,
                        window_size_left=window_size[0],
                        window_size_right=window_size[1],
                        softcap=0.0,
                        alibi_slopes=alibi_slopes,
                        return_softmax=True and dropout_p > 0,
                    )
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