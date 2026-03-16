"""
FSDPShardPinnedCache — zero-reshard offload/reload for FSDP2 models.

After the first forward pass, each rank holds a local shard of every
FSDP-wrapped parameter (a DTensor backed by a CUDA tensor).  This module
captures those shards into a pinned-CPU slab so the model can be offloaded
to free VRAM and hot-reloaded with only a fast H2D memcpy — no collective
communication, no disk I/O, no re-sharding.

Key design decision — lazy build
---------------------------------
``build()`` is called automatically on the FIRST ``offload_to_cpu()`` call,
not at model instantiation time.  Many FSDP models (e.g. Lightricks) use a
deferred weight-loading path where ``set_model_state_dict`` runs during the
first forward pass rather than at wrap time.  If we snapshot too early the
DTensor params are still CPU mmap references, yielding 0 shards captured.
By deferring to the first offload we guarantee shards are on CUDA.

Lifecycle
---------
1. First ``offload_to_cpu(diffusion_model)``
       - Calls ``build()`` automatically if not yet built.
       - ``build()`` copies every CUDA DTensor local shard to pinned RAM.
       - Frees CUDA memory by calling ``untyped_storage().resize_(0)`` on
         each unique root storage.
       - Calls ``torch.cuda.empty_cache()`` so the caching allocator actually
         returns pages and VRAM visibly drops.

2. Subsequent ``offload_to_cpu()`` calls
       - Refreshes pinned buffers with current CUDA values (LoRA may have
         modified weights).
       - Re-frees CUDA storages via resize_(0) + empty_cache.

3. ``reload_to_cuda(diffusion_model)``
       - Calls ``torch.cuda.empty_cache()`` first to reclaim residual
         allocations (e.g. from a just-released VAE).
       - Restores each freed storage via ``resize_(original_nbytes)``
         — re-allocates CUDA memory; DTensor/narrow views auto-follow.
       - Copies shard data from pinned buffers via ``_local_tensor.copy_()``.
       - Synchronizes once at the end.

Why storage-level resize?
-------------------------
FSDP2 stores each sharded parameter in a padded 1D flat buffer
(``FSDPParam._sharded_param_data``).  The DTensor's ``_local_tensor`` is
a **narrow view** into this buffer — they share the same ``UntypedStorage``.
Replacing ``_local_tensor.data = empty(0)`` only detaches the view but does
NOT free the underlying flat buffer.  ``storage.resize_(0)`` frees the root
CUDA allocation, and ``storage.resize_(nbytes)`` re-allocates it; all
views automatically follow their storage object.

Notes
-----
- Uses the public ``DTensor.to_local()`` API for shard capture.
- Writes directly through ``_local_tensor.copy_()`` for restore (no public
  mutation API exists).
- Buffers are moved via ``.to()`` as they are small and not DTensor-wrapped.
- When ``CPUOffload(offload_params=True)`` is active, FSDP manages shard
  placement; this cache is not used in that case.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import torch
from torch.distributed._tensor import DTensor

log = logging.getLogger(__name__)


def _is_fsdp_dtensor(param: torch.Tensor) -> bool:
    """Return True if this parameter is an FSDP2 DTensor shard."""
    return isinstance(param, DTensor) or isinstance(param.data, DTensor)


def _as_dtensor(param: torch.Tensor) -> DTensor:
    if isinstance(param, DTensor):
        return param
    return param.data  # type: ignore[return-value]


def _local_cuda(param: torch.Tensor) -> "torch.Tensor | None":
    """Return the rank-local CUDA backing tensor via the public to_local() API.

    Returns None if the local tensor is not on CUDA or has zero elements
    (e.g. param not yet materialised by a deferred weight-load path).
    """
    dt = _as_dtensor(param)
    local = dt.to_local()
    if local.device.type != "cuda" or local.numel() == 0:
        return None
    return local


class FSDPShardPinnedCache:
    """Pinned-RAM cache for FSDP2 local shards enabling zero-reshard offload/reload."""

    def __init__(self) -> None:
        # name → pinned-CPU tensor (shape matches the FSDP local shard)
        self._cpu_shards: Dict[str, torch.Tensor] = {}
        # name → (shape, dtype)  saved so we can re-allocate CUDA storage correctly
        self._meta: Dict[str, Tuple[torch.Size, torch.dtype]] = {}
        self.shards_on_cuda: bool = False
        self._built: bool = False
        # CUDA UntypedStorage references + original sizes for resize-based VRAM freeing.
        # FSDP2 stores each param's shard in a padded flat buffer (_sharded_param_data)
        # and DTensor._local_tensor is a narrow VIEW into it.  Replacing the view's .data
        # does NOT free the underlying flat buffer.  Instead, we call
        # storage.resize_(0) to truly release CUDA memory, and storage.resize_(nbytes)
        # to re-allocate it on reload.  Views automatically follow their storage object.
        self._freed_storages: list = []  # [(UntypedStorage, original_nbytes)]

    # ------------------------------------------------------------------ build

    def build(self, diffusion_model: torch.nn.Module) -> None:
        """Snapshot all CUDA DTensor local shards to pinned CPU RAM.

        Called automatically from ``offload_to_cpu()`` the first time it runs.
        At that point the first forward has completed and ``set_model_state_dict``
        has been applied, so all DTensor params are guaranteed to be on CUDA.
        """
        if self._built:
            print("[FSDPShardCache] build() called more than once — skipping.")
            return

        count = 0
        skipped_cpu = 0
        skipped_not_dtensor = 0
        total_params = 0
        for name, param in diffusion_model.named_parameters():
            total_params += 1
            if not _is_fsdp_dtensor(param):
                skipped_not_dtensor += 1
                continue
            local = _local_cuda(param)
            if local is None:
                skipped_cpu += 1
                continue
            cpu_buf = local.detach().to("cpu", non_blocking=False)
            try:
                cpu_buf = cpu_buf.pin_memory()
            except Exception:
                pass  # fall back to pageable — H2D still works, just not async
            self._cpu_shards[name] = cpu_buf
            self._meta[name] = (local.shape, local.dtype)
            count += 1

        if skipped_cpu:
            print(
                f"[FSDPShardCache] WARNING: {skipped_cpu} DTensor params not on CUDA at build time "
                f"— not cached (deferred load not yet triggered?)."
            )

        self._built = True
        self.shards_on_cuda = True
        print(
            f"[FSDPShardCache] Built: {count} CUDA shards captured, "
            f"{self.pinned_ram_bytes() / 1e9:.2f} GB pinned RAM. "
            f"(total_params={total_params}, non_dtensor={skipped_not_dtensor}, cpu_dtensor={skipped_cpu})"
        )

    # --------------------------------------------------------------- offload

    def offload_to_cpu(
        self,
        diffusion_model: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Move all FSDP local shards CUDA → pinned RAM and free VRAM.

        Automatically calls ``build()`` on the first invocation so the deferred-
        load path (weights materialised on first forward) is handled correctly.

        Returns the number of shards offloaded.
        """
        # --- lazy build: defer until we know shards are on CUDA ---
        if not self._built:
            print("[FSDPShardCache] First offload — building shard cache now...")
            self.build(diffusion_model)

        if not self.shards_on_cuda:
            print("[FSDPShardCache] Shards already on CPU — skipping offload.")
            return 0

        offloaded = 0
        freed_ptrs: set = set()
        self._freed_storages = []

        for name, param in diffusion_model.named_parameters():
            if not _is_fsdp_dtensor(param):
                if param.device.type == "cuda":
                    param.data = param.data.to("cpu", non_blocking=non_blocking)
                continue

            local = _local_cuda(param)
            if local is None:
                continue

            # (1) Refresh pinned buffer (LoRA may have modified weights since last build)
            if name in self._cpu_shards:
                self._cpu_shards[name].copy_(local, non_blocking=non_blocking)
            else:
                cpu_buf = local.detach().to("cpu", non_blocking=False)
                try:
                    cpu_buf = cpu_buf.pin_memory()
                except Exception:
                    pass
                self._cpu_shards[name] = cpu_buf
                self._meta[name] = (local.shape, local.dtype)

            # (2) Collect the *root* CUDA UntypedStorage backing this shard.
            #     DTensor._local_tensor is a narrow VIEW into FSDPParam._sharded_param_data.
            #     Both share the same UntypedStorage.  Freeing the storage frees both.
            s = local.untyped_storage()
            ptr = s.data_ptr()
            if ptr != 0 and ptr not in freed_ptrs:
                freed_ptrs.add(ptr)
                self._freed_storages.append((s, s.nbytes()))
            offloaded += 1

        # Sync all async D2H copies BEFORE freeing storages —
        # the copy source (CUDA shard) must remain valid until sync.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # (3) Free CUDA storage at the root level.  resize_(0) releases the
        #     underlying cudaMalloc allocation.  All DTensor/narrow views that
        #     reference this storage become empty (zero bytes) but stay intact
        #     as Python objects — FSDP module structure is preserved.
        for s, _ in self._freed_storages:
            s.resize_(0)

        # Move CUDA buffers (small non-param tensors)
        for buf in diffusion_model.buffers():
            if buf.device.type == "cuda":
                buf.data = buf.data.to("cpu", non_blocking=False)

        # CRITICAL: tell the CUDA caching allocator to return freed blocks.
        torch.cuda.empty_cache()

        self.shards_on_cuda = False
        freed_gb = sum(nb for _, nb in self._freed_storages) / 1e9
        print(
            f"[FSDPShardCache] Offloaded {offloaded} shards, freed {len(self._freed_storages)} "
            f"unique CUDA storages ({freed_gb:.2f} GB). VRAM freed."
        )
        return offloaded

    # --------------------------------------------------------------- reload

    def reload_to_cuda(
        self,
        diffusion_model: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Restore all FSDP local shards from pinned RAM → CUDA.

        Allocates fresh CUDA storage per shard, issues async H2D copies from
        pinned buffers, synchronizes once.  No collective communication.

        Returns the number of shards reloaded.
        """
        if not self._built:
            print("[FSDPShardCache] reload_to_cuda() called but cache not built — no-op.")
            return 0
        if self.shards_on_cuda:
            print("[FSDPShardCache] Shards already on CUDA — skipping reload.")
            return 0

        # Release residual caching-allocator reservations (e.g. from a recently offloaded VAE)
        # before we try to allocate shard storage, to avoid competing with stale reserved blocks.
        torch.cuda.empty_cache()

        # (1) Restore every freed CUDA storage to its original size.
        #     This re-allocates CUDA memory.  All DTensor/narrow views that reference
        #     each storage automatically become valid again (same offsets, new memory).
        restored = 0
        for s, nbytes in self._freed_storages:
            s.resize_(nbytes)
            restored += 1
        self._freed_storages = []
        print(f"[FSDPShardCache] Restored {restored} CUDA storages.")

        # (2) Copy pinned-CPU shard data back into the (now valid) local tensor views.
        reloaded = 0
        for name, param in diffusion_model.named_parameters():
            if not _is_fsdp_dtensor(param):
                if param.device.type != "cuda":
                    param.data = param.data.to("cuda", non_blocking=non_blocking)
                continue

            cpu_buf = self._cpu_shards.get(name)
            if cpu_buf is None:
                print(f"[FSDPShardCache] WARNING: No cached shard for '{name}' — skipping.")
                continue

            # _local_tensor is a narrow view into the restored storage.
            # copy_ writes the real shard data into the correct region.
            dt = _as_dtensor(param)
            dt._local_tensor.copy_(cpu_buf, non_blocking=non_blocking)
            reloaded += 1

        # One sync covers all async H2D copies
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Restore buffers
        for buf in diffusion_model.buffers():
            if buf.device.type != "cuda":
                buf.data = buf.data.to("cuda", non_blocking=False)

        self.shards_on_cuda = True
        print(f"[FSDPShardCache] Reloaded {reloaded} shards to CUDA.")
        return reloaded

    # ---------------------------------------------------------------- helpers

    @property
    def built(self) -> bool:
        return self._built

    def shard_count(self) -> int:
        return len(self._cpu_shards)

    def pinned_ram_bytes(self) -> int:
        return sum(t.nbytes for t in self._cpu_shards.values())
