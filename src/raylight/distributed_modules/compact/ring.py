"""
NOTE: code from yunchang
"""

import torch
import os



from yunchang.ring.utils import RingComm, update_out_and_lse




from raylight.distributed_modules.compact.utils import COMPACT_COMPRESS_TYPE
from raylight.distributed_modules.compact.main import (
    compact_config,
    compact_cache,
    compact_compress,
    compact_decompress,
)
from raylight.distributed_modules.compact.prof import Profiler
import torch.nn.functional as F

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
            mask=mask
        )
    else:
        return _compact_ring_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter,
            mask=mask
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

        out, lse = memory_efficient_attention_forward(
            q, k, v,
            attn_bias=attn_bias,
            p=dropout_p,
            scale=softmax_scale,
        )
        return out, lse

    print("[Raylight] Falling back to manual PyTorch attention (xformers not available).")

    # Fallback to manual implementation (Matches yunchang return signature)
    # q, k, v are (bs, seq_len, head_cnt, head_size)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    
    attn = (q_t @ k_t.transpose(-2, -1)) * softmax_scale
    if mask is not None:
         attn = attn + mask 
    if causal:
         L, S = q_t.size(-2), k_t.size(-2)
         c_mask = torch.ones(L, S, device=q.device, dtype=torch.bool).tril()
         attn = attn.masked_fill(~c_mask, float("-inf"))
    
    lse = torch.logsumexp(attn, dim=-1, keepdim=True)
    attn_probs = torch.exp(attn - lse)
    if dropout_p > 0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        
    out = attn_probs @ v_t
    out = out.transpose(1, 2)
    lse = lse.squeeze(-1) # (bs, head, seq)
    
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
    k_my_cache_key = f"{mod_idx}-{my_rank_mod}-k"
    v_my_cache_key = f"{mod_idx}-{my_rank_mod}-v"
    
    # Pre-calculate all recv_ranks and their cache keys
    recv_ranks = [(rank - s) % world_size for s in range(world_size)]
    k_recv_keys = [f"{mod_idx}-{r}-k" for r in recv_ranks]
    v_recv_keys = [f"{mod_idx}-{r}-v" for r in recv_ranks]

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
                q_idx = slice(q_start, q_start + q_len) if (len(mask.shape) > 2 and mask.shape[2] > 1) else slice(None)
                k_idx = slice(k_start, k_start + k_len_block) if (len(mask.shape) > 3 and mask.shape[3] > 1) else slice(None)
                
                mask_slice = mask[:, :, q_idx, k_idx]

            if mask is not None or flash_attn is None:
                if step == 0:
                    print(f"[Raylight] Side-stepping FlashAttn: mask={mask is not None}, flash_attn={flash_attn is not None}")
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