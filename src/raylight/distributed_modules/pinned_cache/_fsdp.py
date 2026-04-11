"""FSDPShardPinnedCache — FSDP2 DTensor-aware pinned-RAM cache."""

from __future__ import annotations

from typing import Dict

import torch

from ._base import BasePinnedCache

# ---------------------------------------------------------------------------
# DTensor helpers (only used by this module)
# ---------------------------------------------------------------------------

def _is_fsdp_dtensor(param: torch.Tensor) -> bool:
    from torch.distributed.tensor._api import DTensor
    return isinstance(param, DTensor) or isinstance(param.data, DTensor)


def _as_dtensor(param: torch.Tensor):
    from torch.distributed.tensor._api import DTensor
    if isinstance(param, DTensor):
        return param
    return param.data


def _local_cuda(param: torch.Tensor) -> "torch.Tensor | None":
    """Return the rank-local CUDA backing tensor, or None if not on CUDA."""
    dt = _as_dtensor(param)
    local = dt.to_local()  # type: ignore
    if local.device.type != "cuda" or local.numel() == 0:
        return None
    return local


# ---------------------------------------------------------------------------
# FSDPShardPinnedCache
# ---------------------------------------------------------------------------

class FSDPShardPinnedCache(BasePinnedCache):
    """Pinned-RAM cache for FSDP2 local shards — zero-reshard offload/reload.

    DTensor local shards are backed by FSDP's padded flat buffers.
    ``storage.resize_(0/nbytes)`` frees/restores the root allocation and
    all narrow views automatically follow.
    """

    def __init__(self) -> None:
        super().__init__(tag="FSDPShardCache")
        self._cpu_shards: Dict[str, torch.Tensor] = {}

    # -- build --

    def _do_build(self, module: torch.nn.Module) -> None:
        count = skipped_cpu = skipped_not_dt = total = 0
        needs_sync = False

        for name, param in module.named_parameters():
            total += 1
            if not _is_fsdp_dtensor(param):
                skipped_not_dt += 1
                continue
            local = _local_cuda(param)
            if local is None:
                skipped_cpu += 1
                continue
            # Pre-allocate pinned + async DMA (no double-copy)
            pinned = torch.empty_like(local, device="cpu").pin_memory()
            pinned.copy_(local, non_blocking=True)
            self._cpu_shards[name] = pinned
            needs_sync = True
            count += 1

        # Single sync point for all async D2H transfers
        if needs_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        if skipped_cpu:
            print(
                f"[{self._tag}] WARNING: {skipped_cpu} DTensor params not on CUDA "
                f"at build time — not cached."
            )
        print(
            f"[{self._tag}] Built: {count} CUDA shards, "
            f"{self.pinned_ram_bytes() / 1e9:.2f} GB pinned. "
            f"(total={total}, non_dtensor={skipped_not_dt}, cpu={skipped_cpu})"
        )

    # -- offload --

    def _snapshot_to_cpu(self, module, non_blocking, freed_ptrs) -> int:
        offloaded = 0
        for name, param in self._param_list:
            if not _is_fsdp_dtensor(param):
                # Move plain (non-DTensor) params to CPU directly
                if param.device.type == "cuda":
                    param.data = param.data.to("cpu", non_blocking=non_blocking)
                continue

            local = _local_cuda(param)
            if local is None:
                continue

            # Refresh pinned buffer
            if name in self._cpu_shards:
                self._cpu_shards[name].copy_(local, non_blocking=non_blocking)
            else:
                # Cache miss: pre-allocate pinned + async DMA (no double-copy)
                pinned = torch.empty_like(local, device="cpu").pin_memory()
                pinned.copy_(local, non_blocking=True)
                self._cpu_shards[name] = pinned
            self._collect_storage(local, freed_ptrs)
            offloaded += 1

        # Move small non-param buffers to CPU
        for buf in module.buffers():
            if buf.device.type == "cuda":
                buf.data = buf.data.to("cpu", non_blocking=non_blocking)

        return offloaded

    # -- reload --

    def _copy_to_cuda(self, module, non_blocking) -> int:
        reloaded = 0
        for name, param in self._param_list:
            if not _is_fsdp_dtensor(param):
                if param.device.type != "cuda":
                    param.data = param.data.to("cuda", non_blocking=non_blocking)
                continue

            cpu_buf = self._cpu_shards.get(name)
            if cpu_buf is None:
                print(f"[{self._tag}] WARNING: No cached shard for '{name}'.")
                continue

            dt = _as_dtensor(param)
            dt._local_tensor.copy_(cpu_buf, non_blocking=non_blocking)  # type: ignore
            reloaded += 1

        # Restore buffers
        for buf in module.buffers():
            if buf.device.type != "cuda":
                buf.data = buf.data.to("cuda", non_blocking=non_blocking)

        return reloaded

    def _swap_to_cpu(self, module) -> int:
        raise NotImplementedError(
            "FSDPShardPinnedCache does not support partial CUDA load."
        )

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        raise NotImplementedError(
            "FSDPShardPinnedCache does not support selective offload."
        )

    # -- sync cuda → pinned (for bake caching) --

    def sync_from_cuda(self, module: torch.nn.Module) -> int:
        """Copy current CUDA shard data back to pinned CPU buffers.

        Call after in-place weight modification (e.g. LoRA baking) to update
        the pinned cache so that subsequent offload/reload and pressure
        eviction preserves the baked state.

        Returns:
            Number of shards synced.
        """
        synced = 0
        for name, param in self._param_list:
            if not _is_fsdp_dtensor(param):
                continue
            local = _local_cuda(param)
            if local is None:
                continue
            cpu_buf = self._cpu_shards.get(name)
            if cpu_buf is None:
                continue
            cpu_buf.copy_(local, non_blocking=True)
            synced += 1

        if synced > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            # Clear the force-reload flag since pinned is now in sync
            self._force_next_reload = False

        print(f"[{self._tag}] Synced {synced} CUDA shards → pinned (baked state cached).")
        return synced

    # -- info --

    def param_count(self) -> int:
        return len(self._cpu_shards)

    # Alias kept for backward compat
    shard_count = param_count

    def pinned_ram_bytes(self) -> int:
        return sum(t.nbytes for t in self._cpu_shards.values())

    def cleanup(self) -> None:
        """Release all pinned shard tensors and reset state."""
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_shards else 0
        self._cpu_shards.clear()
        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: freed {gb:.2f} GB pinned shard RAM.")
