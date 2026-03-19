import os
from .registry import AttentionRegistry
from .backends.standard import StandardAttentionBackend
from .backends.compact import CompactAttentionBackend

# Register backends
AttentionRegistry.register("STANDARD", StandardAttentionBackend)
AttentionRegistry.register("COMPACT", CompactAttentionBackend)

# Global variables for backward compatibility (used by get_attn_type/get_sync_ulysses)
_ATTN_TYPE = None
_SYNC_ULYSSES = None
_RING_IMPL_TYPE = "basic"


def set_attn_type(attn):
    global _ATTN_TYPE
    _ATTN_TYPE = attn


def get_attn_type():
    if _ATTN_TYPE is None:
        raise RuntimeError("_ATTN_TYPE is not initialized")
    else:
        return _ATTN_TYPE


def set_sync_ulysses(is_sync):
    global _SYNC_ULYSSES
    _SYNC_ULYSSES = is_sync


def get_sync_ulysses():
    if _SYNC_ULYSSES is None:
        raise RuntimeError("_SYNC_ULYSSES variable is not initialized")
    else:
        return _SYNC_ULYSSES


def set_ring_impl_type(impl_type):
    global _RING_IMPL_TYPE
    _RING_IMPL_TYPE = impl_type


def get_ring_impl_type():
    return _RING_IMPL_TYPE


def make_xfuser_attention(attn_type, sync_ulysses, ring_impl_type=None):
    """
    Create xFuser attention using the registered backend.
    Currently defaults to 'STANDARD' backend.
    """
    backend_name = os.environ.get("RAYLIGHT_ATTN_BACKEND", "STANDARD")
    
    if ring_impl_type is None:
        ring_impl_type = get_ring_impl_type()

    backend = AttentionRegistry.get(backend_name)
    return backend.create_attention(attn_type, sync_ulysses, ring_impl_type=ring_impl_type)


# Cache for attention callables keyed by (backend_name, attn_type, sync_ulysses, ring_impl_type).
# Avoids recreating RaylightAttention (and for COMPACT, re-running compact_init)
# on every single attention forward call.
_attn_callable_cache: dict[tuple, callable] = {}


def invalidate_attn_cache():
    """Clear the attention callable cache.

    Call when switching attention backends, changing COMPACT env-var config
    between runs, or after ``compact_reset()`` to ensure the next attention
    call creates a fresh backend instance.
    """
    _attn_callable_cache.clear()


def make_lazy_attention(ring_impl_type_override=None):
    """Return a callable that resolves the attention backend lazily at call time.

    Unlike ``make_xfuser_attention`` (which captures the backend config at
    creation time), the returned function re-reads ``get_attn_type()``,
    ``get_sync_ulysses()``, ``get_ring_impl_type()`` and the
    ``RAYLIGHT_ATTN_BACKEND`` env-var on **every invocation**.  This means
    that changing the backend between ComfyUI queue items takes effect
    immediately without restarting Ray worker processes.

    The resolved callable is cached by ``(backend_name, attn_type,
    sync_ulysses, ring_impl_type)`` so that repeated calls with the same
    config reuse the same ``RaylightAttention`` instance instead of
    allocating a new one every forward pass.

    ``ring_impl_type_override`` – if not None, pins the ring implementation
    (e.g. ``"basic"`` for non-causal models like LTX / Lumina) instead of
    reading the global setting.
    """
    def _lazy_attention(*args, **kwargs):
        attn_type = get_attn_type()
        sync_ulysses = get_sync_ulysses()
        rim = ring_impl_type_override if ring_impl_type_override is not None else get_ring_impl_type()
        backend_name = os.environ.get("RAYLIGHT_ATTN_BACKEND", "STANDARD")
        cache_key = (backend_name, attn_type, sync_ulysses, rim)
        fn = _attn_callable_cache.get(cache_key)
        if fn is None:
            fn = make_xfuser_attention(attn_type, sync_ulysses, ring_impl_type=rim)
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
