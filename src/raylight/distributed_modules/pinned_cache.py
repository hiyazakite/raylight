"""
Pinned-RAM caches for fast VRAM offload / reload.

Three cache variants share a common base that handles the
``UntypedStorage.resize_(0)`` / ``resize_(nbytes)`` VRAM lifecycle:

``PinnedParamCache``
    Private per-process pinned RAM for single-GPU (non-FSDP) models.

``SharedPinnedParamCache``
    Cross-process ``/dev/shm`` buffer + ``cudaHostRegister`` for
    data-parallel workers that all hold identical weights.

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

import ctypes
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA host-register helpers (shared by SharedPinnedParamCache & GGUF pinning)
# ---------------------------------------------------------------------------

_ALIGNMENT = 512  # byte alignment per param slot (matches GPU DMA granularity)


def _align_up(offset: int, alignment: int = _ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def _cuda_host_register(ptr: int, size: int) -> bool:
    """Pin existing host pages for CUDA DMA via ``cudaHostRegister``."""
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.current_device()  # lazily initialises CUDA context

        cudart = torch.cuda.cudart()
        # cudaHostRegisterPortable = 1 (visible to all CUDA contexts)
        # pybind11 binding expects plain Python ints, NOT ctypes wrappers.
        err = cudart.cudaHostRegister(int(ptr), int(size), int(1))
        if isinstance(err, tuple):
            err = err[0]
        ok = int(err) == 0
        if not ok:
            print(f"[PinnedCache] cudaHostRegister returned error {err}")
        return ok
    except Exception as e:
        print(f"[PinnedCache] cudaHostRegister exception: {e}")
        return False


def _cuda_host_unregister(ptr: int) -> bool:
    try:
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostUnregister(int(ptr))
        if isinstance(err, tuple):
            err = err[0]
        return int(err) == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Shared-memory layout helper (used by SharedPinnedParamCache)
# ---------------------------------------------------------------------------

def _compute_layout(
    module: torch.nn.Module,
) -> Tuple[List[Tuple[str, int, int, torch.Size, torch.dtype]],
           List[Tuple[str, int, int, torch.Size, torch.dtype]],
           int]:
    """Return (param_layout, buffer_layout, total_bytes).

    Each entry: ``(name, byte_offset, nbytes, shape, dtype)``.
    Deterministic across processes given identical module structure.
    """
    param_layout: list = []
    buffer_layout: list = []
    offset = 0

    for name, param in module.named_parameters():
        if param.device.type != "cuda":
            continue
        aligned = _align_up(offset)
        nb = param.data.nbytes
        param_layout.append((name, aligned, nb, param.data.shape, param.data.dtype))
        offset = aligned + nb

    for name, buf in module.named_buffers():
        if buf.device.type != "cuda":
            continue
        aligned = _align_up(offset)
        nb = buf.data.nbytes
        buffer_layout.append((name, aligned, nb, buf.data.shape, buf.data.dtype))
        offset = aligned + nb

    return param_layout, buffer_layout, _align_up(offset)


def make_cache_id(model_path: str) -> str:
    """Deterministic short hash from model path — identical across ranks."""
    return hashlib.md5(model_path.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class BasePinnedCache(ABC):
    """Abstract base for all pinned-RAM offload/reload caches.

    Subclasses implement ``_do_build``, ``_snapshot_to_cpu``,
    ``_copy_to_cuda`` — the storage-freeing and error-checking boilerplate
    lives here.
    """

    def __init__(self, tag: str = "PinnedCache") -> None:
        self._tag = tag
        self._built: bool = False
        self._on_cuda: bool = False
        # (UntypedStorage, original_nbytes) — collected during offload
        self._freed_storages: List[Tuple[torch.UntypedStorage, int]] = []

    # -------------------------------------------------------------- public

    @property
    def built(self) -> bool:
        return self._built

    @property
    def params_on_cuda(self) -> bool:
        return self._on_cuda

    @params_on_cuda.setter
    def params_on_cuda(self, v: bool) -> None:
        self._on_cuda = v

    # Alias for FSDP compat (code references .shards_on_cuda)
    @property
    def shards_on_cuda(self) -> bool:
        return self._on_cuda

    @shards_on_cuda.setter
    def shards_on_cuda(self, v: bool) -> None:
        self._on_cuda = v

    def build(self, module: torch.nn.Module) -> None:
        if self._built:
            print(f"[{self._tag}] build() called more than once — skipping.")
            return
        self._do_build(module)
        self._built = True
        self._on_cuda = True

    def offload_to_cpu(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Snapshot CUDA → pinned RAM, free VRAM via storage resize."""
        if not self._built:
            print(f"[{self._tag}] First offload — building cache now...")
            self.build(module)

        if not self._on_cuda:
            print(f"[{self._tag}] Already on CPU — skipping offload.")
            return 0

        freed_ptrs: set = set()
        self._freed_storages = []

        offloaded = self._snapshot_to_cpu(module, non_blocking, freed_ptrs)

        # Sync D2H copies BEFORE freeing storages
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._pre_free_hook()

        # Free all unique CUDA storages
        for s, _ in self._freed_storages:
            s.resize_(0)

        torch.cuda.empty_cache()
        self._on_cuda = False

        freed_gb = sum(nb for _, nb in self._freed_storages) / 1e9
        print(
            f"[{self._tag}] Offloaded {offloaded} params, freed "
            f"{len(self._freed_storages)} CUDA storages ({freed_gb:.2f} GB)."
        )
        return offloaded

    def reload_to_cuda(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Re-allocate VRAM and copy pinned → CUDA."""
        if not self._built:
            print(f"[{self._tag}] Not built — cannot reload.")
            return 0
        if self._on_cuda:
            print(f"[{self._tag}] Already on CUDA — skipping reload.")
            return 0

        torch.cuda.empty_cache()

        # Re-allocate all freed CUDA storages
        restored = 0
        for s, nbytes in self._freed_storages:
            s.resize_(nbytes)
            restored += 1
        self._freed_storages = []
        print(f"[{self._tag}] Restored {restored} CUDA storages.")

        reloaded = self._copy_to_cuda(module, non_blocking)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._on_cuda = True
        print(f"[{self._tag}] Reloaded {reloaded} params to CUDA.")
        return reloaded

    @abstractmethod
    def param_count(self) -> int: ...

    @abstractmethod
    def pinned_ram_bytes(self) -> int: ...

    # ------------------------------------------------------------ overrides

    @abstractmethod
    def _do_build(self, module: torch.nn.Module) -> None:
        """Subclass: snapshot CUDA data to pinned host RAM."""

    @abstractmethod
    def _snapshot_to_cpu(
        self,
        module: torch.nn.Module,
        non_blocking: bool,
        freed_ptrs: set,
    ) -> int:
        """Subclass: refresh pinned buffers and collect storages.

        Must append to ``self._freed_storages`` and add ptrs to *freed_ptrs*.
        Return count of params offloaded.
        """

    @abstractmethod
    def _copy_to_cuda(
        self,
        module: torch.nn.Module,
        non_blocking: bool,
    ) -> int:
        """Subclass: copy pinned data → CUDA params.  Return count."""

    def _pre_free_hook(self) -> None:
        """Optional hook called after sync but before storage.resize_(0).

        Used by SharedPinnedParamCache for dist.barrier().
        """

    # ------------------------------------------------------------ helpers

    def _collect_storage(
        self,
        tensor: torch.Tensor,
        freed_ptrs: set,
    ) -> None:
        """Register a tensor's CUDA storage for freeing (deduped by ptr)."""
        s = tensor.untyped_storage()
        ptr = s.data_ptr()
        if ptr != 0 and ptr not in freed_ptrs:
            freed_ptrs.add(ptr)
            self._freed_storages.append((s, s.nbytes()))

    @staticmethod
    def _pin(tensor: torch.Tensor) -> torch.Tensor:
        """Best-effort pin_memory; returns the tensor unchanged on failure."""
        try:
            return tensor.pin_memory()
        except Exception:
            return tensor


# ═══════════════════════════════════════════════════════════════════════════
# PinnedParamCache — single-GPU, private pinned RAM
# ═══════════════════════════════════════════════════════════════════════════

class PinnedParamCache(BasePinnedCache):
    """Private pinned-RAM cache for standard (non-FSDP) single-GPU models."""

    def __init__(self) -> None:
        super().__init__(tag="PinnedParamCache")
        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}

    # -- build --

    def _do_build(self, module: torch.nn.Module) -> None:
        param_count = buf_count = skipped_cpu = 0

        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                skipped_cpu += 1
                continue
            self._cpu_params[name] = self._pin(
                param.data.detach().to("cpu", non_blocking=False)
            )
            param_count += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            self._cpu_buffers[name] = self._pin(
                buf.data.detach().to("cpu", non_blocking=False)
            )
            buf_count += 1

        print(
            f"[{self._tag}] Built: {param_count} params + {buf_count} buffers, "
            f"{self.pinned_ram_bytes() / 1e9:.2f} GB pinned. "
            f"(skipped_cpu={skipped_cpu})"
        )

    # -- offload --

    def _snapshot_to_cpu(self, module, non_blocking, freed_ptrs) -> int:
        offloaded = 0
        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                continue
            if name in self._cpu_params:
                self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
            else:
                self._cpu_params[name] = self._pin(
                    param.data.detach().to("cpu", non_blocking=False)
                )
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            if name in self._cpu_buffers:
                self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
            else:
                self._cpu_buffers[name] = self._pin(
                    buf.data.detach().to("cpu", non_blocking=False)
                )
            self._collect_storage(buf.data, freed_ptrs)

        return offloaded

    # -- reload --

    def _copy_to_cuda(self, module, non_blocking) -> int:
        reloaded = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data.copy_(cpu_buf, non_blocking=non_blocking)
                reloaded += 1

        for name, buf in module.named_buffers():
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                buf.data.copy_(cpu_copy, non_blocking=non_blocking)

        return reloaded

    # -- info --

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return (sum(t.nbytes for t in self._cpu_params.values())
                + sum(t.nbytes for t in self._cpu_buffers.values()))


# ═══════════════════════════════════════════════════════════════════════════
# SharedPinnedParamCache — cross-process /dev/shm + cudaHostRegister
# ═══════════════════════════════════════════════════════════════════════════

class SharedPinnedParamCache(BasePinnedCache):
    """Cross-process shared pinned-RAM cache for data-parallel workers.

    Rank 0 is the **writer** (creates ``/dev/shm`` segment, copies data).
    Other ranks are **readers** (attach to existing segment).
    All ranks call ``cudaHostRegister`` for async DMA.
    """

    def __init__(self, cache_id: str, is_writer: bool = False) -> None:
        super().__init__(tag="SharedPinnedParamCache")
        self._cache_id = cache_id
        self._is_writer = is_writer

        self._shm_name = f"raylight_pc_{cache_id}"
        self._shm: Optional[SharedMemory] = None
        self._registered_ptr: Optional[int] = None
        self._total_bytes: int = 0

        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}

    # -- build --

    def _do_build(self, module: torch.nn.Module) -> None:
        param_layout, buffer_layout, total_bytes = _compute_layout(module)
        self._total_bytes = total_bytes

        if total_bytes == 0:
            print(f"[{self._tag}] No CUDA params found — nothing to cache.")
            return

        if self._is_writer:
            self._build_writer(module, param_layout, buffer_layout, total_bytes)
        else:
            self._build_reader(param_layout, buffer_layout, total_bytes)

    def _build_writer(self, module, param_layout, buffer_layout, total_bytes):
        # Clean up stale segment
        try:
            old = SharedMemory(name=self._shm_name, create=False)
            old.close(); old.unlink()
        except FileNotFoundError:
            pass

        self._shm = SharedMemory(name=self._shm_name, create=True, size=total_bytes)
        self._register_shm(total_bytes)

        param_dict = dict(module.named_parameters())
        for name, off, nb, shape, dtype in param_layout:
            view = torch.frombuffer(
                memoryview(self._shm.buf)[off:off + nb], dtype=dtype
            ).reshape(shape)
            view.copy_(param_dict[name].data.cpu())
            self._cpu_params[name] = view

        buf_dict = dict(module.named_buffers())
        for name, off, nb, shape, dtype in buffer_layout:
            view = torch.frombuffer(
                memoryview(self._shm.buf)[off:off + nb], dtype=dtype
            ).reshape(shape)
            view.copy_(buf_dict[name].data.cpu())
            self._cpu_buffers[name] = view

        # Signal readers
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

        print(
            f"[{self._tag}] Writer built: {len(param_layout)} params + "
            f"{len(buffer_layout)} buffers, {total_bytes / 1e9:.2f} GB shared "
            f"(shm={self._shm_name})"
        )

    def _build_reader(self, param_layout, buffer_layout, total_bytes):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

        # Attach to existing segment
        for i in range(50):
            try:
                self._shm = SharedMemory(name=self._shm_name, create=False)
                break
            except FileNotFoundError:
                if i < 49:
                    time.sleep(0.2)
        else:
            raise RuntimeError(
                f"[{self._tag}] Reader timed out waiting for "
                f"shared memory '{self._shm_name}' (10s)."
            )

        if self._shm.size < total_bytes:
            raise RuntimeError(
                f"[{self._tag}] Size mismatch: expected {total_bytes}, "
                f"got {self._shm.size}."
            )

        self._register_shm(total_bytes)

        for name, off, nb, shape, dtype in param_layout:
            self._cpu_params[name] = torch.frombuffer(
                memoryview(self._shm.buf)[off:off + nb], dtype=dtype
            ).reshape(shape)

        for name, off, nb, shape, dtype in buffer_layout:
            self._cpu_buffers[name] = torch.frombuffer(
                memoryview(self._shm.buf)[off:off + nb], dtype=dtype
            ).reshape(shape)

        print(
            f"[{self._tag}] Reader attached: {len(param_layout)} params, "
            f"{total_bytes / 1e9:.2f} GB shared (shm={self._shm_name})"
        )

    def _register_shm(self, total_bytes: int) -> None:
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        ok = _cuda_host_register(ptr, total_bytes)
        self._registered_ptr = ptr
        if not ok:
            role = "Writer" if self._is_writer else "Reader"
            print(f"[{self._tag}] WARNING: cudaHostRegister failed ({role}) — "
                  "DMA will use staging buffer (slower).")

    # -- offload --

    def _snapshot_to_cpu(self, module, non_blocking, freed_ptrs) -> int:
        offloaded = 0
        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                continue
            # Writer refreshes shared buffer; readers skip the copy
            if self._is_writer and name in self._cpu_params:
                self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            if self._is_writer and name in self._cpu_buffers:
                self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
            self._collect_storage(buf.data, freed_ptrs)

        return offloaded

    def _pre_free_hook(self) -> None:
        """Barrier so readers don't free before writer finishes writing."""
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    # -- reload --

    def _copy_to_cuda(self, module, non_blocking) -> int:
        reloaded = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data.copy_(cpu_buf, non_blocking=non_blocking)
                reloaded += 1

        for name, buf in module.named_buffers():
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                buf.data.copy_(cpu_copy, non_blocking=non_blocking)

        return reloaded

    # -- info --

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return self._total_bytes

    # -- cleanup --

    def cleanup(self) -> None:
        """Unpin and release shared memory."""
        if self._registered_ptr is not None:
            _cuda_host_unregister(self._registered_ptr)
            self._registered_ptr = None
        if self._shm is not None:
            self._shm.close()
            if self._is_writer:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# FSDPShardPinnedCache — FSDP2 DTensor-aware
# ═══════════════════════════════════════════════════════════════════════════

def _is_fsdp_dtensor(param: torch.Tensor) -> bool:
    from torch.distributed._tensor import DTensor
    return isinstance(param, DTensor) or isinstance(param.data, DTensor)


def _as_dtensor(param: torch.Tensor):
    from torch.distributed._tensor import DTensor
    if isinstance(param, DTensor):
        return param
    return param.data


def _local_cuda(param: torch.Tensor) -> "torch.Tensor | None":
    """Return the rank-local CUDA backing tensor, or None if not on CUDA."""
    dt = _as_dtensor(param)
    local = dt.to_local()
    if local.device.type != "cuda" or local.numel() == 0:
        return None
    return local


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

        for name, param in module.named_parameters():
            total += 1
            if not _is_fsdp_dtensor(param):
                skipped_not_dt += 1
                continue
            local = _local_cuda(param)
            if local is None:
                skipped_cpu += 1
                continue
            self._cpu_shards[name] = self._pin(
                local.detach().to("cpu", non_blocking=False)
            )
            count += 1

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
        for name, param in module.named_parameters():
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
                self._cpu_shards[name] = self._pin(
                    local.detach().to("cpu", non_blocking=False)
                )
            self._collect_storage(local, freed_ptrs)
            offloaded += 1

        # Move small non-param buffers to CPU
        for buf in module.buffers():
            if buf.device.type == "cuda":
                buf.data = buf.data.to("cpu", non_blocking=False)

        return offloaded

    # -- reload --

    def _copy_to_cuda(self, module, non_blocking) -> int:
        reloaded = 0
        for name, param in module.named_parameters():
            if not _is_fsdp_dtensor(param):
                if param.device.type != "cuda":
                    param.data = param.data.to("cuda", non_blocking=non_blocking)
                continue

            cpu_buf = self._cpu_shards.get(name)
            if cpu_buf is None:
                print(f"[{self._tag}] WARNING: No cached shard for '{name}'.")
                continue

            dt = _as_dtensor(param)
            dt._local_tensor.copy_(cpu_buf, non_blocking=non_blocking)
            reloaded += 1

        # Restore buffers
        for buf in module.buffers():
            if buf.device.type != "cuda":
                buf.data = buf.data.to("cuda", non_blocking=False)

        return reloaded

    # -- info --

    def param_count(self) -> int:
        return len(self._cpu_shards)

    # Alias kept for backward compat
    shard_count = param_count

    def pinned_ram_bytes(self) -> int:
        return sum(t.nbytes for t in self._cpu_shards.values())
