from typing import Optional
from raylight.distributed_modules.attention.backends.fusion.context import compact_config
from raylight.distributed_modules.attention.backends.fusion.prof import Profiler
import torch
import torch.distributed as dist
from raylight.distributed_modules.attention.backends.fusion.utils import COMPACT_COMPRESS_TYPE

# Import necessary components for DistriFusion
from raylight.distributed_modules.attention.backends.fusion.patchpara.df_cache import AllGatherCache, DummyHandle
from raylight.distributed_modules.attention.backends.fusion.patchpara.df_utils import PatchConfig

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None

from raylight.distributed_modules.attention.backends.fusion.ring import compact_attn_forward
_buffers = {}

def clear_buffers():
    global _buffers
    _buffers = {}

@Profiler.prof_func("patch_gather_fwd.gather_patch_fwd")
def patch_gather_fwd(
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
    Attention forward pass using all_gather for K/V tensors (no compression).
    Supports synchronous or asynchronous (DistriFusion) communication.
    """
    assert alibi_slopes is None, "Alibi slopes not supported in this basic gather impl."
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    config_obj = compact_config()
    override = config_obj.override_with_patch_gather_fwd if config_obj else False
    assert override, "Patch gather fwd is not enabled"
    config: Optional[PatchConfig] = config_obj.patch_gather_fwd_config if config_obj else None
    assert config is not None, "patch_gather_fwd_config is required"
    assert mod_idx is not None, "mod_idx is required for caching"
    assert current_iter is not None, "current_iter is required for async logic"

    # --- Joint Tensor Handling --- # (Keep this logic as is)
    is_joint = False
    if (joint_tensor_key is not None and
        joint_tensor_value is not None):
        supported_joint_strategy = ["front", "rear", "none"] # Added none for clarity
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        elif joint_strategy != "none":
            is_joint = True
    elif (joint_tensor_key is None and
          joint_tensor_value is None):
        joint_strategy = "none" # Ensure strategy is none if tensors are none
    else:
        raise ValueError(
            f"joint_tensor_key and joint_tensor_value should be None or not None simultaneously."
        )
    # --- End Joint Tensor Handling ---

    process_group = group
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    
    cache = None
    if config.async_comm:
        from raylight.distributed_modules.attention.backends.fusion.main import allgather_cache
        cache = allgather_cache()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    k_list_for_computation = None
    v_list_for_computation = None

    # --- Communication Step (Sync or Async) ---
    if config.use_compact:
        from raylight.distributed_modules.attention.backends.fusion.ops import compact_all_gather
        comp_type_k = config_obj.compress_func(mod_idx, current_iter) if (config_obj and config_obj.compress_func) else COMPACT_COMPRESS_TYPE.IDENTITY
        comp_type_v = config_obj.compress_func(mod_idx, current_iter) if (config_obj and config_obj.compress_func) else COMPACT_COMPRESS_TYPE.IDENTITY
        k_list_for_computation = compact_all_gather(
            f"{mod_idx}-k",
            k,
            comp_type=comp_type_k,
            group=process_group,
        )
        v_list_for_computation = compact_all_gather(
            f"{mod_idx}-v",
            v,
            comp_type=comp_type_v,
            group=process_group,
        )
    elif not config.async_comm:
        # --- Synchronous Communication --- #
        k_list = [torch.empty_like(k) for _ in range(world_size)]
        v_list = [torch.empty_like(v) for _ in range(world_size)]
        with Profiler.scope("compact.gather.all_gather_sync"):
            dist.all_gather(k_list, k, group=process_group)
            dist.all_gather(v_list, v, group=process_group)
        k_list_for_computation = k_list
        v_list_for_computation = v_list
        # --- End Synchronous --- #
    else:
        # --- Asynchronous Communication (DistriFusion) --- #
        k_cache_key = f'{mod_idx}-k'
        v_cache_key = f'{mod_idx}-v'
        # print(f"k_cache_key: {k_cache_key}, v_cache_key: {v_cache_key}")
        with Profiler.scope("df.all_gather"):
            if current_iter < config.async_warmup:
                if _buffers.get(k_cache_key) is None:
                    _buffers[k_cache_key] = [torch.empty_like(k).contiguous() for _ in range(world_size)]
                    _buffers[v_cache_key] = [torch.empty_like(v).contiguous() for _ in range(world_size)]
                
                # --- Warmup Phase (Sync Gather + Cache Dummy Handle) --- #
                k_list = _buffers[k_cache_key]
                v_list = _buffers[v_cache_key]
                # Perform actual sync gather
                dist.all_gather(k_list, k, group=process_group)
                dist.all_gather(v_list, v, group=process_group)

                # Store results with a dummy handle for the *next* step to wait on (no-op)
                if cache is not None:
                    cache.put(k_cache_key, DummyHandle(), k_list, k)
                    cache.put(v_cache_key, DummyHandle(), v_list, v)

                # Use current results for computation in this warmup step
                k_list_for_computation = k_list
                v_list_for_computation = v_list
                # --- End Warmup --- #
            else:
                # --- Async Phase (Wait Previous + Launch Current) --- #
                # Wait for and get results from the *previous* step's async gather
                if cache is None or not cache.contains(k_cache_key) or not cache.contains(v_cache_key):
                        # Should be populated by the last warmup step or previous async step
                        raise RuntimeError(f"DistriFusion cache miss for key {k_cache_key} or {v_cache_key} at iter {current_iter}. Check async_warmup steps.")

                prev_k_handle, prev_k_list, _ = cache.get(k_cache_key)
                prev_v_handle, prev_v_list, _ = cache.get(v_cache_key)

                # Wait for the previous communication to complete
                prev_k_handle.wait()
                prev_v_handle.wait()

                # Use the received data for computation in *this* step
                k_list_for_computation = [buf.clone() for buf in prev_k_list]
                v_list_for_computation = [buf.clone() for buf in prev_v_list]
                
                # update with fresh k v
                k_list_for_computation[rank] = k.clone()
                v_list_for_computation[rank] = v.clone()
                # Launch the *current* step's async gather for the *next* step
                # Prepare receive buffers for the next step's gather
                next_k_list = _buffers[k_cache_key]
                next_v_list = _buffers[v_cache_key]

                # Launch async all_gather
                k_handle = dist.all_gather(next_k_list, k, group=process_group, async_op=True)
                v_handle = dist.all_gather(next_v_list, v, group=process_group, async_op=True)

                # Store the handle and receive buffers for the next step
                if cache is not None:
                    cache.put(k_cache_key, k_handle, next_k_list, k)
                    cache.put(v_cache_key, v_handle, next_v_list, v)
                # --- End Async Phase --- #
    # --- End Communication Step --- #

    # Concatenate gathered tensors for computation
    assert k_list_for_computation is not None and v_list_for_computation is not None, "Gathered tensor lists are None"
    global_k = torch.cat(k_list_for_computation, dim=1).contiguous()
    global_v = torch.cat(v_list_for_computation, dim=1).contiguous()

    # --- Apply Joint Tensors to Global K/V ---
    key_to_use = global_k
    value_to_use = global_v
    if is_joint:
        if joint_strategy == "front":
            key_to_use = torch.cat([joint_tensor_key, global_k], dim=1) # type: ignore
            value_to_use = torch.cat([joint_tensor_value, global_v], dim=1) # type: ignore
        elif joint_strategy == "rear":
            key_to_use = torch.cat([global_k, joint_tensor_key], dim=1) # type: ignore
            value_to_use = torch.cat([global_v, joint_tensor_value], dim=1) # type: ignore
    # --- End Apply Joint Tensors ---

    # PERFORMANCE: Avoid synchronous .all() checks in inner loops.
    # Top-level usp_dit_forward already cleans up trivial masks by setting them to None.
    # If mask is not None here, we assume it's non-trivial.
    is_trivial_mask = False

    # Perform attention with potentially augmented global K and V
    if mask is not None and not is_trivial_mask:
        # Use compact_attn_forward which handles chunked LSE and masks
        out, lse = compact_attn_forward(
            q,
            key_to_use, 
            value_to_use,
            dropout_p,
            softmax_scale,
            causal=causal,
            mask=mask,
        )
    elif flash_attn is None:
        # Use PyTorch attention if flash_attn not available
        from yunchang.kernels.attention import pytorch_attn_forward
        out, lse = pytorch_attn_forward(
            q,
            key_to_use, 
            value_to_use,
            dropout_p,
            softmax_scale,
            causal=causal,
        )
    else:
        # Use flash_attn
        v_str = getattr(flash_attn, "__version__", "0.0.0")
        use_legacy = False
        try:
            v_parts = [int(p) for p in v_str.split('.') if p.isdigit()]
            if v_parts and v_parts[0] < 2:
                use_legacy = True
            elif v_parts and v_parts[0] == 2 and len(v_parts) > 1 and v_parts[1] <= 6:
                use_legacy = True
        except Exception:
            pass

        fa_kwargs = {
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "softcap": 0.0,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }

        if use_legacy:
            fa_kwargs["window_size"] = window_size
            fa_res = _flash_attn_forward(q, key_to_use, value_to_use, **fa_kwargs) # type: ignore
            out = fa_res[0]
            lse = fa_res[5]  # type: ignore
        else:
            fa_kwargs["window_size_left"] = window_size[0]
            fa_kwargs["window_size_right"] = window_size[1]
            fa_res = _flash_attn_forward(q, key_to_use, value_to_use, **fa_kwargs) # type: ignore
            out = fa_res[0]
            lse = fa_res[1]  # type: ignore

    out = out.to(q.dtype) # type: ignore
    lse = lse.squeeze(dim=-1).transpose(1, 2) # type: ignore

    return out, lse, None