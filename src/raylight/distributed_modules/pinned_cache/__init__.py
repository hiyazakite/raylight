"""
Pinned-RAM caches for fast VRAM offload / reload.

Three cache variants share a common base that handles the
``UntypedStorage.resize_(0)`` / ``resize_(nbytes)`` VRAM lifecycle:

``PinnedParamCache``
    Private per-process pinned RAM for single-GPU (non-FSDP) models.

``SharedPinnedParamCache``
    Cross-process ``/dev/shm`` buffer + ``cudaHostRegister`` for
    data-parallel actors that all hold identical weights.

``FSDPShardPinnedCache``
    FSDP2 DTensor-aware cache that operates on per-rank local shards.

All three expose the same public interface::

    cache.build(module)              # snapshot CUDA → pinned (lazy, called by offload)
    cache.offload_to_cpu(module)     # refresh pinned snapshot, free VRAM
    cache.reload_to_cuda(module)     # re-allocate VRAM, copy pinned → CUDA
    cache.built                      # bool property
    cache.param_count()
    cache.pinned_ram_bytes()

Why ``storage.resize_``?
------------------------
``nn.Parameter.data`` is backed by an ``UntypedStorage``.  Some params may
share storage (e.g. weight-tying, FSDP flat buffers).  ``resize_(0)`` truly
frees the CUDA allocation while ``resize_(nbytes)`` re-allocates it — all
tensor views that reference the storage automatically follow.
"""

from __future__ import annotations

# Re-export helpers so existing internal imports keep working.
from ._helpers import (
    _ALIGNMENT,
    _align_up,
    _compute_layout,
    _cuda_host_register,
    _cuda_host_unregister,
    make_cache_id,
)

from ._base import BasePinnedCache
from ._pinned import PinnedParamCache
from ._dict import DictPinnedCache
from ._contiguous import ContiguousPinnedCache
from ._shared import SharedPinnedParamCache
from ._fsdp import FSDPShardPinnedCache

__all__ = [
    "BasePinnedCache",
    "PinnedParamCache",
    "DictPinnedCache",
    "ContiguousPinnedCache",
    "SharedPinnedParamCache",
    "FSDPShardPinnedCache",
    "make_cache_id",
    # Internal helpers re-exported for backward compat
    "_ALIGNMENT",
    "_align_up",
    "_compute_layout",
    "_cuda_host_register",
    "_cuda_host_unregister",
]
