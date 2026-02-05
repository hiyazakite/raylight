from .registry import AttentionRegistry
from .standard import StandardAttentionBackend
from .compact import CompactAttentionBackend

# Register backends
AttentionRegistry.register("STANDARD", StandardAttentionBackend)
AttentionRegistry.register("COMPACT", CompactAttentionBackend)

# Global variables for backward compatibility (used by get_attn_type/get_sync_ulysses)
_ATTN_TYPE = None
_SYNC_ULYSSES = None


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


def make_xfuser_attention(attn_type, sync_ulysses):
    """
    Create xFuser attention using the registered backend.
    Currently defaults to 'STANDARD' backend.
    """
    # In Phase 2, we can switch on a config flag here or let the caller decide
    import os
    backend_name = os.environ.get("RAYLIGHT_ATTN_BACKEND", "STANDARD")
    
    backend = AttentionRegistry.get(backend_name)
    return backend.create_attention(attn_type, sync_ulysses)
