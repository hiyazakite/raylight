import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Callable
from .registry import AttentionRegistry
from .backends.standard import StandardAttentionBackend
from .backends.compact import CompactAttentionBackend

if TYPE_CHECKING:
    from raylight.config import RaylightConfig

# Register backends
AttentionRegistry.register("STANDARD", StandardAttentionBackend)
AttentionRegistry.register("COMPACT", CompactAttentionBackend)

# [SENIOR REFACTOR] Global Config Synchronization
_CONFIG: Optional['RaylightConfig'] = None

def set_config(config: 'RaylightConfig'):
    """Central point to set the attention configuration."""
    global _CONFIG
    _CONFIG = config
    # Invalidate cache when config changes
    invalidate_attn_cache()

def get_config() -> 'RaylightConfig':
    """Retrieve the current attention configuration."""
    global _CONFIG
    if _CONFIG is None:
        from raylight.config import RaylightConfig
        _CONFIG = RaylightConfig.from_env()
    return _CONFIG

# --- Legacy Backward Compatibility Layer ---
def set_attn_type(attn):
    from raylight.config import RaylightAttnType
    # In a real senior refactor, we would update the config object via replace()
    # For now, we maintain the old globals or update a dummy config
    print(f"[Raylight] Deprecated: set_attn_type({attn}) called. Prefer set_config().")
    cfg = get_config()
    from dataclasses import replace
    try:
        if isinstance(attn, str):
            attn = RaylightAttnType[attn]
        new_cfg = replace(cfg, strategy=replace(cfg.strategy, attention_type=attn))
        set_config(new_cfg)
    except Exception:
        pass

def get_attn_type():
    return get_config().strategy.attention_type

def set_sync_ulysses(is_sync):
    cfg = get_config()
    from dataclasses import replace
    new_cfg = replace(cfg, strategy=replace(cfg.strategy, sync_ulysses=is_sync))
    set_config(new_cfg)

def get_sync_ulysses():
    return get_config().strategy.sync_ulysses

def set_ring_impl_type(impl_type):
    cfg = get_config()
    from dataclasses import replace
    new_cfg = replace(cfg, strategy=replace(cfg.strategy, ring_impl=impl_type))
    set_config(new_cfg)

def get_ring_impl_type():
    return get_config().strategy.ring_impl

def make_xfuser_attention(raylight_config=None, **kwargs):
    """
    Create xFuser attention using the registered backend.
    """
    cfg = raylight_config or get_config()
    backend_name = cfg.strategy.attention_backend
    backend = AttentionRegistry.get(backend_name)
    return backend.create_attention(raylight_config=cfg, **kwargs)


# Cache for attention callables keyed by (backend_name, attn_type, sync_ulysses, ring_impl_type).
# Avoids recreating RaylightAttention (and for COMPACT, re-running compact_init)
# on every single attention forward call.
_attn_callable_cache: Dict[Tuple, Callable] = {}


def invalidate_attn_cache():
    """Clear the attention callable cache.

    Call when switching attention backends, changing COMPACT env-var config
    between runs, or after ``compact_reset()`` to ensure the next attention
    call creates a fresh backend instance.
    """
    _attn_callable_cache.clear()


def make_lazy_attention(ring_impl_type_override=None):
    """Return a callable that resolves the attention backend lazily at call time."""
    def _lazy_attention(*args, **kwargs):
        cfg = get_config()
        
        # Apply local override if requested
        if ring_impl_type_override is not None:
            from dataclasses import replace
            cfg = replace(cfg, strategy=replace(cfg.strategy, ring_impl=ring_impl_type_override))
            
        # Build cache key from relevant strategy fields
        cache_key = (
            cfg.strategy.attention_backend,
            cfg.strategy.attention_type,
            cfg.strategy.sync_ulysses,
            cfg.strategy.ring_impl
        )
        
        fn = _attn_callable_cache.get(cache_key)
        if fn is None:
            fn = make_xfuser_attention(raylight_config=cfg)
            _attn_callable_cache[cache_key] = fn
        return fn(*args, **kwargs)
    return _lazy_attention


# Bounded module-level cache for mask triviality results.  Keyed by
# ``id(mask)`` so the same mask tensor object (passed through many ring
# steps / layers in one forward pass) is only checked once.  The cache is
# very small (typically 1-2 distinct masks per forward) and is cleared when
# it reaches the cap to avoid accumulating stale entries.
_mask_nontrivial_cache: dict[int, bool] = {}
_MASK_CACHE_MAX = 8


def invalidate_mask_cache():
    """Clear the mask triviality cache.

    Call at step boundaries or when mask content is modified in-place to
    prevent stale cached results from silently disabling masking.
    """
    _mask_nontrivial_cache.clear()


def check_mask_nontrivial(mask):
    """Check if an attention mask has any non-zero entries.

    Uses ``torch.any`` for a complete check (no sampling heuristic) with a
    single GPU→CPU scalar transfer, and caches the result per mask object
    in a bounded module-level dict so repeat calls within the same forward
    pass are free.

    Returns ``False`` when *mask* is ``None``.
    """
    if mask is None:
        return False

    key = id(mask)
    result = _mask_nontrivial_cache.get(key)
    if result is not None:
        return result

    if mask.numel() == 0:
        result = False
    elif mask.numel() <= 1:
        result = bool(mask.item() != 0)
    else:
        # torch.any → single-element bool tensor → one .item() GPU→CPU sync.
        # This is a complete, correct check (no sampling false-negatives).
        result = bool(mask.any().item())

    # Bounded cache: evict all entries when full to keep memory and lookup
    # cost trivial.  In practice there are only 1-2 distinct masks.
    if len(_mask_nontrivial_cache) >= _MASK_CACHE_MAX:
        _mask_nontrivial_cache.clear()
    _mask_nontrivial_cache[key] = result
    return result
