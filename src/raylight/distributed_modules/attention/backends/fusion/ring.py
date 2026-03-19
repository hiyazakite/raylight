"""
NOTE: code from yunchang
"""

import torch
import os

from yunchang.ring.utils import RingComm, update_out_and_lse

from raylight.distributed_modules.attention.backends.fusion.utils import COMPACT_COMPRESS_TYPE
from raylight.distributed_modules.attention.backends.fusion.context import compact_config, compact_cache
from raylight.distributed_modules.attention.backends.fusion.ops import compact_compress, compact_decompress
from raylight.distributed_modules.attention.backends.fusion.prof import Profiler
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

# P1: Overlap decompress with NCCL communication on a secondary CUDA stream.
# After comm.wait() receives compressed K/V, decompress is launched on
# _decomp_stream so it runs concurrently with the next step's send_recv.
# Disable with RAYLIGHT_OVERLAP_DECOMP=0 if it causes issues.
_OVERLAP_DECOMP = os.getenv("RAYLIGHT_OVERLAP_DECOMP", "1") == "1"
_decomp_stream_cache: dict[int, torch.cuda.Stream] = {}  # keyed by device index


def _get_decomp_stream(device: torch.device) -> torch.cuda.Stream:
    """Return a cached CUDA stream for decompress overlap on *device*."""
    idx = device.index if device.index is not None else 0
    if idx not in _decomp_stream_cache:
        _decomp_stream_cache[idx] = torch.cuda.Stream(device=device)
    return _decomp_stream_cache[idx]


# OPT 7: Cache RingComm instances per process group
_ring_comm_cache = {}

def _get_ring_comm(process_group):
    """Reuse RingComm for the same process group to avoid repeated allocation."""
    group_key = id(process_group)
    if group_key not in _ring_comm_cache:
        _ring_comm_cache[group_key] = RingComm(process_group)
    return _ring_comm_cache[group_key]

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
        # print(f"📦 [Raylight] compact_fwd called (Layer: {mod_idx}, Iter: {current_iter}, mask={mask_info})")
        pass
    # Trust that higher-level logic (usp_dit_forward) has already 
    # stripped trivial masks to avoid GPU-CPU sync here.
    from xfuser.core.distributed import get_sequence_parallel_rank
    get_sequence_parallel_rank() # Just to ensure it's imported

    config = compact_config()
    gather = config.override_with_patch_gather_fwd if config else False
    
    if gather:
        from raylight.distributed_modules.attention.backends.fusion.patchpara.fwd import patch_gather_fwd
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
    from raylight.distributed_modules.attention.backends.fusion.slowpath import set_current_lowrank_scale
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
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + mask[b, h % mask.shape[1], q_idx, kv_idx]

        if flex_attention is None or AuxRequest is None:
            raise ImportError("FlexAttention components not available despite _HAS_FLEX_ATTN being True")

        if q.dim() == 4:
            if q.shape[1] > q.shape[2]:  # (B, S, H, D)
                q_flex = q.transpose(1, 2)
                k_flex = k.transpose(1, 2)
                v_flex = v.transpose(1, 2)
            else:  # already B, H, S, D
                q_flex = q
                k_flex = k
                v_flex = v
        else:
            raise ValueError(f"Expected 4D query, got {q.dim()}D")
        
        res = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod, return_aux=AuxRequest(lse=True))
        out, aux = res
        lse = getattr(aux, "lse", None)
        out = out.transpose(1, 2).contiguous()
        return out, lse

    # xformers is very fast for masked attention and returns LSE
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

    # OPT 4: SDPA Fallback — use shared chunked LSE utility
    from raylight.distributed_modules.attention.xfuser_ring_patch import compute_chunked_lse

    q_t = q.transpose(1, 2)
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
        # print(f"🐳 [Raylight] _compact_ring_fwd loop (Layer: {mod_idx}, Step: {current_iter}, Q: {q.shape}, mask={mask_info})")
        pass
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
    # OPT 7: Reuse RingComm instance for same process group
    comm = _get_ring_comm(process_group)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = None
    lse = None

    # OPT 1: Avoid GPU-CPU sync — use cached triviality check
    from raylight.distributed_modules.attention import check_mask_nontrivial
    has_nontrivial_mask = check_mask_nontrivial(mask)

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

    # P1: Pre-decompress on a secondary CUDA stream so that decompress
    # of the NEXT step's received K/V overlaps with the current step's
    # NCCL send_recv.  For world_size <= 2, overlap provides no benefit
    # (the decompress cannot run concurrently with useful work).
    use_overlap = _OVERLAP_DECOMP and world_size > 2
    decomp_stream = _get_decomp_stream(q.device) if use_overlap else None
    k_predecomp = None
    v_predecomp = None

    for step in range(world_size):
        # P6: Batch K+V into a single NCCL transfer to halve kernel launches.
        # Instead of two send_recv calls (4 P2P ops), concatenate into one
        # buffer (2 P2P ops) and split after receive.
        buf_kv: torch.Tensor | None = None
        k_split: int = 0
        if step + 1 != world_size:
            k_split = k_to_send.numel()
            kv_to_send = torch.cat([k_to_send.reshape(-1), v_to_send.reshape(-1)])
            buf_kv = comm.send_recv(kv_to_send)
            comm.commit()
        
        if step != 0:
            if k_predecomp is not None:
                # Data was pre-decompressed on the secondary stream at the
                # end of the previous step.  Synchronize so the default
                # stream sees the completed tensors.
                torch.cuda.current_stream().wait_stream(decomp_stream)
                k = k_predecomp
                v = v_predecomp
                k_predecomp = v_predecomp = None
            else:
                # Fallback: decompress synchronously on the default stream
                # (first eligible step when overlap is off, or world_size<=2).
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
            mask_slice = None
            if has_nontrivial_mask:
                q_len = q.shape[1]
                k_len_block = k.shape[1]
                recv_rank = (comm.rank - step) % comm.world_size
                q_start = comm.rank * q_len
                k_start = recv_rank * k_len_block
                q_idx = slice(q_start, q_start + q_len) if (len(mask.shape) > 2 and mask.shape[2] > 1) else slice(None)
                k_idx = slice(k_start, k_start + k_len_block) if (len(mask.shape) > 3 and mask.shape[3] > 1) else slice(None)
                mask_slice = mask[:, :, q_idx, k_idx]

            if mask_slice is not None or flash_attn is None:
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
            if update_out_and_lse is None:
                raise ImportError("yunchang not available, cannot update_out_and_lse")
            if block_lse is None:
                raise RuntimeError("LSE is None from block attention, but required for ring attention update.")
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            with Profiler.scope("compact.ring.wait"):
                comm.wait()
            # P6: Split the received KV buffer back into K and V parts.
            assert buf_kv is not None
            k_to_send = buf_kv[:k_split]
            v_to_send = buf_kv[k_split:]

            # P1: Pre-decompress next step's received K/V on the secondary
            # stream.  This runs concurrently with the next step's NCCL
            # send_recv (posted at the top of the loop) since NCCL and
            # decomp_stream are independent CUDA streams.  Both only READ
            # k_to_send/v_to_send so there is no data race.
            next_step = step + 1
            if use_overlap and next_step > 0:
                # Ensure decomp_stream sees all default-stream work
                # (in particular the buf_k/buf_v contents from comm.wait).
                decomp_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(decomp_stream):
                    k_predecomp = compact_decompress(
                        k_recv_keys[next_step], k_to_send,
                        compress_type_k, original_k_shape, update_cache=True,
                    ).contiguous()
                    if k_predecomp.dtype != q.dtype:
                        k_predecomp = k_predecomp.to(q.dtype)
                    v_predecomp = compact_decompress(
                        v_recv_keys[next_step], v_to_send,
                        compress_type_v, original_v_shape, update_cache=True,
                    ).contiguous()
                    if v_predecomp.dtype != q.dtype:
                        v_predecomp = v_predecomp.to(q.dtype)
    
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