import torch
import logging
from typing import Dict, Any, Callable, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from raylight.config import RaylightConfig

logger = logging.getLogger(__name__)

try:
    from raylight.distributed_modules.attention.backends.fusion.ring import compact_fwd
    from raylight.distributed_modules.attention.backends.fusion.context import compact_get_step
except ImportError:
    compact_fwd = None
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

# ---------------------------------------------------------------------------
# Per-process cached loaders for local attention kernels (TP path).
# Each loader tries the import once; subsequent calls return the cached result.
# Input convention for all wrappers: q/k/v are [B, T, H*D], heads=H scalar.
# ---------------------------------------------------------------------------
_sage_attn_fn: Optional[Callable] = None
_sage_attn_checked: bool = False
_flash_attn2_fn: Optional[Callable] = None
_flash_attn2_checked: bool = False
_flash_attn3_fn: Optional[Callable] = None
_flash_attn3_checked: bool = False


def _try_load_sage_attn() -> Optional[Callable]:
    """Load SageAttention (NHD layout wrapper). Warns at most once on failure."""
    global _sage_attn_fn, _sage_attn_checked
    if _sage_attn_checked:
        return _sage_attn_fn
    _sage_attn_checked = True

    # Try the modern `sageattention` package first, then the older `sage_attention`.
    _sageattn = None
    _hnd_layout = True  # default: [B,H,N,D]
    try:
        from sageattention import sageattn as _sageattn_fn  # type: ignore
        _sageattn = _sageattn_fn
        _hnd_layout = False  # sageattn accepts tensor_layout="NHD"
    except ImportError:
        try:
            from sage_attention import sage_attn as _sageattn_fn  # type: ignore
            _sageattn = _sageattn_fn
            _hnd_layout = True  # legacy sage_attn expects [B,H,N,D]
        except ImportError:
            pass

    if _sageattn is None:
        logger.warning(
            "[Raylight] SageAttention requested but not installed. Falling back to Flash Attention."
        )
        _sage_attn_fn = None
        return None

    if _hnd_layout:
        # Reshape [B, T, H*D] → [B, H, T, D], call, reshape back.
        # k/v use -1 for seq so cross-attention (kv_seq ≠ q_seq) works.
        def _wrap_hnd(q, k, v, heads, mask=None, **kwargs):
            b, q_seq, hd = q.shape
            dim_head = hd // heads
            q = q.view(b, q_seq, heads, dim_head).permute(0, 2, 1, 3).contiguous()
            k = k.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
            v = v.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
            out = _sageattn(q, k, v, is_causal=kwargs.get("causal", False))
            return out.permute(0, 2, 1, 3).reshape(b, q_seq, hd)
        _sage_attn_fn = _wrap_hnd
    else:
        # sageattn NHD: reshape [B, T, H*D] → [B, T, H, D], call with tensor_layout="NHD".
        # k/v use -1 for seq so cross-attention (kv_seq ≠ q_seq) works.
        def _wrap_nhd(q, k, v, heads, mask=None, **kwargs):
            b, q_seq, hd = q.shape
            dim_head = hd // heads
            q = q.view(b, q_seq, heads, dim_head).contiguous()
            k = k.view(b, -1, heads, dim_head).contiguous()
            v = v.view(b, -1, heads, dim_head).contiguous()
            out = _sageattn(q, k, v, tensor_layout="NHD", is_causal=kwargs.get("causal", False))
            return out.reshape(b, q_seq, hd)
        _sage_attn_fn = _wrap_nhd

    return _sage_attn_fn


def _try_load_flash_attn2() -> Optional[Callable]:
    """Load Flash Attention 2. Warns at most once on failure."""
    global _flash_attn2_fn, _flash_attn2_checked
    if _flash_attn2_checked:
        return _flash_attn2_fn
    _flash_attn2_checked = True
    try:
        from flash_attn import flash_attn_func as _fa2  # type: ignore
        def _wrap_fa2(q, k, v, heads, mask=None, **kwargs):
            b, q_seq, hd = q.shape
            dim_head = hd // heads
            # flash_attn_func expects [B, T, H, D]; use -1 for kv seq (cross-attn safe)
            q = q.view(b, q_seq, heads, dim_head)
            k = k.view(b, -1, heads, dim_head)
            v = v.view(b, -1, heads, dim_head)
            out = _fa2(q, k, v, causal=kwargs.get("causal", False))
            return out.view(b, q_seq, hd)
        _flash_attn2_fn = _wrap_fa2
    except ImportError:
        logger.warning(
            "[Raylight] flash_attn not available for TP local attention. Falling back to ComfyUI."
        )
        _flash_attn2_fn = None
    return _flash_attn2_fn


def _try_load_flash_attn3() -> Optional[Callable]:
    """Load Flash Attention 3. Warns at most once on failure."""
    global _flash_attn3_fn, _flash_attn3_checked
    if _flash_attn3_checked:
        return _flash_attn3_fn
    _flash_attn3_checked = True
    try:
        from flash_attn_interface import flash_attn_func as _fa3  # type: ignore
        def _wrap_fa3(q, k, v, heads, mask=None, **kwargs):
            b, q_seq, hd = q.shape
            dim_head = hd // heads
            q = q.view(b, q_seq, heads, dim_head)
            k = k.view(b, -1, heads, dim_head)
            v = v.view(b, -1, heads, dim_head)
            out, _ = _fa3(q, k, v, causal=kwargs.get("causal", False))
            return out.view(b, q_seq, hd)
        _flash_attn3_fn = _wrap_fa3
    except ImportError:
        logger.warning(
            "[Raylight] Flash Attention 3 not available for TP local attention. Falling back to FA2."
        )
        _flash_attn3_fn = None
    return _flash_attn3_fn

def select_ring_attn_fn(
    use_compact_ring: bool,
    has_mask: bool,
    layer_idx: Optional[int] = None,
    ring_impl_type: str = "basic",
    raylight_config: Optional['RaylightConfig'] = None
) -> Callable:
    """
    Centralized logic to select the appropriate ring attention function.
    """
    # [SENIOR REFACTOR] Use Unified Config if available
    if raylight_config:
        use_compact_ring = (raylight_config.strategy.attention_backend == "COMPACT")
        ring_impl_type = raylight_config.strategy.ring_impl
        
    if use_compact_ring:
        if compact_fwd is None:
            raise ImportError("Compact ring backend (compact_fwd) not found but requested.")
        return compact_fwd
    
    # Standard xfuser path
    if ring_impl_type == "zigzag":
        target_fn = xdit_ring_zigzag_flash_attn_func_patched
        if target_fn is not None:
             if raylight_config and raylight_config.debug.verbose_attn:
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
            if raylight_config and raylight_config.debug.verbose_attn:
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
    raylight_config: Optional['RaylightConfig'] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Prepares the keyword arguments for the selected ring attention function.
    Optimized to minimize dictionary overhead.
    """
    # [SENIOR REFACTOR] Use Unified Config if available
    is_compact = (ring_fn is compact_fwd)
    if raylight_config:
        is_compact = (raylight_config.strategy.attention_backend == "COMPACT")
    
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
        # Default current_iter to compact_get_step() if not provided in kwargs
        attn_kwargs["current_iter"] = kwargs.get("current_iter", compact_get_step())
        attn_kwargs["mask"] = mask
    else:
        # standard xfuser ring or patched ring
        attn_kwargs["mod_idx"] = mod_idx # Pass it anyway for consistency/future-proofing
        if mask is not None:
            attn_kwargs["mask"] = mask

    return attn_kwargs

def get_local_attn_fn(raylight_config: Optional['RaylightConfig'] = None) -> Callable:
    """
    Selects a local high-performance attention kernel for use in the TP path,
    where no sequence-parallel communication is needed.

    Dispatch priority per configured RaylightAttnType:
      FLASH_ATTN_3            → FA3 → FA2 → ComfyUI
      FLASH_ATTN              → FA2 → ComfyUI
      SAGE_* (any SAGE type)  → Sage → FA2 → ComfyUI
      TORCH                   → torch SDPA → ComfyUI
      *                       → ComfyUI

    All wrappers accept (q, k, v, heads, mask=None, **kwargs) with
    q/k/v in [B, T, H*D] format (ComfyUI linear output convention).
    """
    from raylight.distributed_modules.attention import get_config
    cfg = raylight_config or get_config()
    attn_type_name = cfg.strategy.attention_type.name

    # FA3
    if "FLASH_ATTN_3" in attn_type_name:
        fn = _try_load_flash_attn3()
        if fn is not None:
            return fn
        # FA3 missing → try FA2
        fn = _try_load_flash_attn2()
        if fn is not None:
            return fn

    # FA2
    elif "FLASH_ATTN" in attn_type_name:
        fn = _try_load_flash_attn2()
        if fn is not None:
            return fn

    # Any SAGE variant → try Sage, then FA2
    elif "SAGE" in attn_type_name:
        fn = _try_load_sage_attn()
        if fn is not None:
            return fn
        fn = _try_load_flash_attn2()
        if fn is not None:
            return fn

    # TORCH SDPA
    elif attn_type_name == "TORCH":
        import torch.nn.functional as F
        def _sdpa(q, k, v, heads, mask=None, **kwargs):
            b, q_seq, hd = q.shape
            dim_head = hd // heads
            # SDPA expects [B, H, T, D]; use -1 for kv seq (cross-attn safe)
            q = q.view(b, q_seq, heads, dim_head).permute(0, 2, 1, 3)
            k = k.view(b, -1, heads, dim_head).permute(0, 2, 1, 3)
            v = v.view(b, -1, heads, dim_head).permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=kwargs.get("causal", False))
            return out.permute(0, 2, 1, 3).reshape(b, q_seq, hd)
        return _sdpa

    # ComfyUI fallback (handles its own reshaping internally)
    import comfy.ldm.modules.attention as comfy_attn

    def _comfy_dispatch(q, k, v, heads, mask=None, **kwargs):
        if mask is None:
            return comfy_attn.optimized_attention(q, k, v, heads, **kwargs)
        else:
            return comfy_attn.optimized_attention_masked(q, k, v, heads, mask, **kwargs)

    return _comfy_dispatch
