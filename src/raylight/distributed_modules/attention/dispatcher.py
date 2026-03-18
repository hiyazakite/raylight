import torch
import logging
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from raylight.distributed_modules.attention.backends.fusion.ring import compact_fwd, _VERBOSE_ATTN
    from raylight.distributed_modules.attention.backends.fusion.context import compact_get_step
except ImportError:
    compact_fwd = None
    _VERBOSE_ATTN = False
    compact_get_step = lambda: 0

try:
    from xfuser.core.long_ctx_attention.ring import xdit_ring_flash_attn_func
except ImportError:
    xdit_ring_flash_attn_func = None

from raylight.distributed_modules.attention.backends.xfuser_ring_patch import (
    xdit_ring_flash_attn_func_patched,
    xdit_ring_zigzag_flash_attn_func_patched
)

_WRAPPED_CACHE = {}

def select_ring_attn_fn(
    use_compact_ring: bool,
    has_mask: bool,
    layer_idx: Optional[int] = None,
    ring_impl_type: str = "basic"
) -> Callable:
    """
    Centralized logic to select the appropriate ring attention function.
    """
    if use_compact_ring:
        if compact_fwd is None:
            raise ImportError("Compact ring backend (compact_fwd) not found but requested.")
        return compact_fwd
    
    # Standard xfuser path
    if ring_impl_type == "zigzag":
        target_fn = xdit_ring_zigzag_flash_attn_func_patched
        if target_fn is not None:
             if _VERBOSE_ATTN:
                print(f"[Raylight] ⚡ Using zigzag ring attention (Layer: {layer_idx})")
             # Wrap to swallow extra kwargs like mod_idx
             if target_fn not in _WRAPPED_CACHE:
                 _fn = target_fn
                 _WRAPPED_CACHE[target_fn] = lambda *args, **kwargs: _fn(*args, **{k: v for k, v in kwargs.items() if k not in ["mod_idx"]})
             return _WRAPPED_CACHE[target_fn]

    if has_mask:
        # Use patched version if available (supports masks)
        target_fn = xdit_ring_flash_attn_func_patched or xdit_ring_flash_attn_func
        if target_fn is not None:
            if _VERBOSE_ATTN:
                print(f"[Raylight] ⚡ Mask present — using patched xfuser ring for mask support (Layer: {layer_idx})")
            # Wrap to swallow extra kwargs like mod_idx
            if target_fn not in _WRAPPED_CACHE:
                _fn = target_fn
                _WRAPPED_CACHE[target_fn] = lambda *args, **kwargs: _fn(*args, **{k: v for k, v in kwargs.items() if k not in ["mod_idx"]})
            return _WRAPPED_CACHE[target_fn]
    
    if xdit_ring_flash_attn_func is not None:
        # Wrap the original function to swallow extra kwargs like mod_idx
        if xdit_ring_flash_attn_func not in _WRAPPED_CACHE:
            _fn = xdit_ring_flash_attn_func
            _WRAPPED_CACHE[xdit_ring_flash_attn_func] = lambda *args, **kwargs: _fn(*args, **{k: v for k, v in kwargs.items() if k not in ["mod_idx"]})
        return _WRAPPED_CACHE[xdit_ring_flash_attn_func]
    
    # Fallback to compact ring if xfuser ring is missing
    logger.warning("[Raylight] Standard xfuser ring not found, falling back to compact ring")
    if compact_fwd is None:
        raise ImportError("No ring attention backend found (tried xfuser and compact).")
    return compact_fwd

def prepare_ring_attn_kwargs(
    ring_fn: Callable,
    layer_idx: int,
    joint_tensor_key: Optional[torch.Tensor] = None,
    joint_tensor_value: Optional[torch.Tensor] = None,
    joint_strategy: str = "none",
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: Any = None,
    attn_layer: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Prepares the keyword arguments for the selected ring attention function.
    Optimized to minimize dictionary overhead.
    """
    is_compact = (ring_fn is compact_fwd)
    
    # Common base
    attn_kwargs = {
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size": window_size,
        "alibi_slopes": alibi_slopes,
        "deterministic": deterministic,
        "return_attn_probs": return_attn_probs,
        "group": group,
        "attn_layer": attn_layer,
        "joint_tensor_key": joint_tensor_key,
        "joint_tensor_value": joint_tensor_value,
        "joint_strategy": joint_strategy,
    }
    
    # Determine mod_idx: prioritize kwargs (from model architecture) over global layer_idx
    mod_idx = kwargs.get("mod_idx")
    if mod_idx is None:
        mod_idx = layer_idx
    
    if is_compact:
        attn_kwargs["mod_idx"] = mod_idx
        attn_kwargs["current_iter"] = kwargs.get("current_iter", compact_get_step())
        attn_kwargs["mask"] = mask
    else:
        # standard xfuser ring or patched ring
        attn_kwargs["mod_idx"] = mod_idx # Pass it anyway for consistency/future-proofing
        if mask is not None:
            attn_kwargs["mask"] = mask

    return attn_kwargs
