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

import ctypes
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from raylight.ipc.backends.posix_shm import PosixShmBackend

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
        if cudart is None:
            return False
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
        if cudart is None:
            return False
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
        aligned = _align_up(offset)
        nb = param.data.nbytes
        param_layout.append((name, aligned, nb, param.data.shape, param.data.dtype))
        offset = aligned + nb

    for name, buf in module.named_buffers():
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
        # Detect actual state: if ANY param is on CUDA, treat as on-CUDA.
        self._on_cuda = any(
            p.device.type == "cuda" for p in module.parameters()
        )

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

    def offload_cuda_only(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Selective offload: snapshot only CUDA-resident params, free their VRAM.

        Used after a *partial* CUDA load where ``model.load(lowvram_model_memory=N)``
        moved some modules to CUDA and left the rest on pinned CPU.  This method:

        1. Copies CUDA-resident params back to pinned cache (refresh).
        2. Frees the corresponding CUDA storages.
        3. Leaves CPU-resident params untouched — they never left pinned RAM.

        After this call the model is fully on (pinned) CPU, ready for the
        next ``reload_to_cpu`` → ``model.load(lowvram_model_memory=…)`` cycle.
        """
        if not self._built:
            print(f"[{self._tag}] First offload — building cache now...")
            self.build(module)

        freed_ptrs: set = set()
        self._freed_storages = []

        cuda_count, cpu_count = self._snapshot_cuda_only(
            module, non_blocking, freed_ptrs
        )

        # Sync D2H copies BEFORE freeing storages
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._pre_free_hook()

        # Free CUDA storages that were on GPU
        for s, _ in self._freed_storages:
            s.resize_(0)

        torch.cuda.empty_cache()
        self._on_cuda = False

        freed_gb = sum(nb for _, nb in self._freed_storages) / 1e9
        print(
            f"[{self._tag}] Selective offload: {cuda_count} CUDA params saved + freed "
            f"({freed_gb:.2f} GB), {cpu_count} CPU params untouched."
        )
        return cuda_count

    def reload_to_cpu(
        self,
        module: torch.nn.Module,
    ) -> int:
        """Restore params from pinned cache to CPU tensors (no CUDA alloc).

        After ``offload_to_cpu`` the model's param storages are zeroed.
        This replaces each param's ``.data`` with the corresponding pinned
        CPU tensor so that a subsequent ``model.load(device, lowvram_model_memory=N)``
        can DMA only the budget-worth of modules to CUDA and leave the
        rest in pinned RAM for per-layer streaming.

        Unlike ``reload_to_cuda`` this does NOT restore CUDA storages;
        ``_freed_storages`` is cleared because ``model.load()`` will create
        fresh CUDA tensors for the modules it moves to GPU.
        """
        if not self._built:
            print(f"[{self._tag}] Not built — cannot reload to CPU.")
            return 0

        # Discard stale CUDA storage refs — model.load() will handle CUDA.
        self._freed_storages = []

        restored = self._swap_to_cpu(module)

        self._on_cuda = False
        print(f"[{self._tag}] Restored {restored} params to pinned CPU.")
        return restored

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

        t_start = time.perf_counter()

        # NOTE: intentionally no empty_cache() here — the caching allocator
        # may still hold freed blocks from a previous model being unloaded;
        # those can be reused by resize_() below without hitting cudaMalloc.
        # The offload path already called empty_cache() when it freed VRAM.

        # Re-allocate all freed CUDA storages
        restored = 0
        alloc_bytes = 0
        for s, nbytes in self._freed_storages:
            s.resize_(nbytes)
            alloc_bytes += nbytes
            restored += 1
        self._freed_storages = []

        t_alloc = time.perf_counter()

        reloaded = self._copy_to_cuda(module, non_blocking)

        t_copy = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_sync = time.perf_counter()

        self._on_cuda = True
        alloc_ms = (t_alloc - t_start) * 1000
        copy_ms = (t_copy - t_alloc) * 1000
        sync_ms = (t_sync - t_copy) * 1000
        total_ms = (t_sync - t_start) * 1000
        total_gb = alloc_bytes / 1e9
        throughput = total_gb / (t_sync - t_start) if (t_sync - t_start) > 0 else 0
        print(
            f"[{self._tag}] Reloaded {reloaded} params to CUDA "
            f"({total_gb:.2f} GB in {total_ms:.0f} ms — "
            f"alloc {alloc_ms:.0f} ms, copy+sync {copy_ms + sync_ms:.0f} ms, "
            f"{throughput:.1f} GB/s)."
        )
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

    @abstractmethod
    def _swap_to_cpu(
        self,
        module: torch.nn.Module,
    ) -> int:
        """Subclass: replace param.data with pinned CPU tensor.  Return count."""

    @abstractmethod
    def _snapshot_cuda_only(
        self,
        module: torch.nn.Module,
        non_blocking: bool,
        freed_ptrs: set,
    ) -> tuple:
        """Subclass: refresh pinned cache for CUDA params only.

        Returns ``(cuda_count, cpu_count)``.
        Must append CUDA storages to ``self._freed_storages``.
        """

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

    # ------------------------------------------------------------ cleanup

    def cleanup(self) -> None:
        """Release all pinned RAM held by this cache.

        Subclasses override to free format-specific resources (e.g. /dev/shm)
        and must call ``super().cleanup()``.
        """
        self._freed_storages = []
        self._built = False
        self._on_cuda = False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


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
        param_count = buf_count = pinned_cpu = not_pinned = 0
        needs_sync = False

        for name, param in module.named_parameters():
            if param.device.type == "cuda":
                # Pre-allocate pinned destination and async DMA — avoids the
                # double-copy of .to("cpu") + .pin_memory() and pipelines
                # all transfers on the GPU's copy engine.
                pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                pinned.copy_(param.data, non_blocking=True)
                self._cpu_params[name] = pinned
                needs_sync = True
            else:
                # Already on CPU — allocate a pinned buffer and copy in one
                # shot.  Avoids the transient unpinned clone that
                # .detach().clone().pin_memory() would create.
                pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                pinned.copy_(param.data)
                self._cpu_params[name] = pinned
                param.data = pinned          # swap — original freed by RC
                pinned_cpu += 1
            param_count += 1

        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                pinned = torch.empty_like(buf.data, device="cpu").pin_memory()
                pinned.copy_(buf.data, non_blocking=True)
                self._cpu_buffers[name] = pinned
                needs_sync = True
            else:
                pinned = torch.empty_like(buf.data, device="cpu").pin_memory()
                pinned.copy_(buf.data)
                self._cpu_buffers[name] = pinned
                buf.data = pinned
            buf_count += 1

        # Single sync point for all async D2H transfers
        if needs_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Verify pinning — unpinned buffers silently fall back to staged
        # (pageable) DMA at roughly half the bandwidth.
        for t in self._cpu_params.values():
            if not t.is_pinned():
                not_pinned += 1
        for t in self._cpu_buffers.values():
            if not t.is_pinned():
                not_pinned += 1

        pin_status = "all pinned ✓" if not_pinned == 0 else f"WARNING: {not_pinned} tensors NOT pinned (DMA will use slow staging path)"
        print(
            f"[{self._tag}] Built: {param_count} params + {buf_count} buffers, "
            f"{self.pinned_ram_bytes() / 1e9:.2f} GB pinned. "
            f"(pinned_in_place={pinned_cpu}, {pin_status})"
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
                # Cache miss: pre-allocate pinned + async DMA (no double-copy)
                pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                pinned.copy_(param.data, non_blocking=True)
                self._cpu_params[name] = pinned
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            if name in self._cpu_buffers:
                self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
            else:
                pinned = torch.empty_like(buf.data, device="cpu").pin_memory()
                pinned.copy_(buf.data, non_blocking=True)
                self._cpu_buffers[name] = pinned
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

    def _swap_to_cpu(self, module) -> int:
        """Point param.data directly at pinned CPU tensors."""
        restored = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data = cpu_buf
                restored += 1

        for name, buf in module.named_buffers():
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                buf.data = cpu_copy

        return restored

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        cuda_count = cpu_count = 0
        for name, param in module.named_parameters():
            if param.device.type == "cuda":
                if name in self._cpu_params:
                    self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
                else:
                    # Cache miss: pre-allocate pinned + async DMA (no double-copy)
                    pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                    pinned.copy_(param.data, non_blocking=True)
                    self._cpu_params[name] = pinned
                self._collect_storage(param.data, freed_ptrs)
                cuda_count += 1
            else:
                cpu_count += 1

        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                if name in self._cpu_buffers:
                    self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
                else:
                    pinned = torch.empty_like(buf.data, device="cpu").pin_memory()
                    pinned.copy_(buf.data, non_blocking=True)
                    self._cpu_buffers[name] = pinned
                self._collect_storage(buf.data, freed_ptrs)

        return cuda_count, cpu_count

    # -- info --

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return (sum(t.nbytes for t in self._cpu_params.values())
                + sum(t.nbytes for t in self._cpu_buffers.values()))

    def cleanup(self) -> None:
        """Release all pinned tensors and reset state."""
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_params else 0
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: freed {gb:.2f} GB pinned RAM.")


# ═══════════════════════════════════════════════════════════════════════════
# SharedPinnedParamCache — cross-process /dev/shm + cudaHostRegister
# ═══════════════════════════════════════════════════════════════════════════

class SharedPinnedParamCache(BasePinnedCache):
    """Cross-process shared pinned-RAM cache for data-parallel actors.

    All ranks cooperate during build:

    1. Rank 0 creates the ``/dev/shm`` segment (allocation only).
    2. Barrier — all ranks attach and ``cudaHostRegister``.
    3. Each rank copies its assigned **slice** of params (round-robin by
       index, non-overlapping — no synchronisation needed within the copy
       phase).  Buffers are copied by rank 0 only (typically tiny).
    4. Barrier — all data is in place.
    5. All ranks create views and swap ``param.data``.

    The result is identical to the old serial writer pattern but the
    memcpy work is distributed across *N* actors.
    """

    def __init__(
        self,
        cache_id: str,
        is_writer: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        *,
        shm_backend: Optional["PosixShmBackend"] = None,
    ) -> None:
        super().__init__(tag="SharedPinnedParamCache")
        self._cache_id = cache_id
        self._is_writer = is_writer
        self._local_rank = local_rank
        self._world_size = world_size

        self._shm_name = f"raylight_pc_{cache_id}"
        self._shm_backend = shm_backend
        self._shm_metadata: Optional[object] = None  # HostIpcArtifactMetadata
        self._registered_ptr: Optional[int] = None
        self._total_bytes: int = 0

        # Deterministic round-robin assignment: param at sorted-index *i*
        # is owned by rank ``i % world_size``.
        self._owned_params: set = set()   # param names this rank copies
        self._owned_buffers: set = set()  # buffer names this rank copies

        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}

    # -- build --

    def _do_build(self, module: torch.nn.Module) -> None:
        param_layout, buffer_layout, total_bytes = _compute_layout(module)
        self._total_bytes = total_bytes

        if total_bytes == 0:
            print(f"[{self._tag}] No params found — nothing to cache.")
            return

        self._build_parallel(module, param_layout, buffer_layout, total_bytes)

    def _build_parallel(self, module, param_layout, buffer_layout, total_bytes):
        """Cooperative build: all ranks create views, each copies its slice."""
        import torch.distributed as dist
        t_start = time.perf_counter()

        backend = self._shm_backend
        if backend is None:
            raise RuntimeError(
                f"[{self._tag}] shm_backend is required for SharedPinnedParamCache"
            )

        # ── Step 1: Rank 0 creates shared segment ──────────────────────
        if self._is_writer:
            from raylight.ipc.types import HostIpcBufferSpec
            spec = HostIpcBufferSpec(
                prefix="raylight_pc",
                logical_name=self._cache_id,
                owner_scope=f"rank_{self._local_rank}",
                size_bytes=total_bytes,
            )
            self._shm_metadata = backend.create_artifact(spec)

        # ── Step 2: Barrier — wait for creation, then all attach ───────
        if dist.is_initialized():
            dist.barrier()

        if not self._is_writer:
            self._shm_metadata = backend.attach_with_retry(self._shm_name)

            shm = backend.get_handle(self._shm_name)
            if shm is not None and shm.size < total_bytes:
                raise RuntimeError(
                    f"[{self._tag}] Size mismatch: expected {total_bytes}, "
                    f"got {shm.size}."
                )

        shm = backend.get_handle(self._shm_name)
        assert shm is not None, f"SharedMemory handle not found for {self._shm_name}"

        # All ranks: pin the pages for async CUDA DMA
        self._register_shm_buf(shm.buf, total_bytes)

        # ── Step 3: Build ownership sets + create views ────────────────
        #   Round-robin: param at sorted-index *i* is owned by rank
        #   i % world_size.  All ranks create ALL views (needed for
        #   reload_to_cuda), but each rank only .copy_() its own slice.
        param_dict = dict(module.named_parameters())
        buf_dict = dict(module.named_buffers())

        for i, (name, off, nb, shape, dtype) in enumerate(param_layout):
            view = torch.frombuffer(
                memoryview(shm.buf)[off:off + nb], dtype=dtype,
            ).reshape(shape)
            self._cpu_params[name] = view
            if i % self._world_size == self._local_rank:
                self._owned_params.add(name)
                # DMA directly from CUDA into host-registered shm —
                # avoids the temporary pageable CPU tensor that
                # .data.cpu() would allocate.
                src = param_dict[name].data
                if src.device.type == "cuda":
                    view.copy_(src)          # CUDA→registered-host DMA
                else:
                    view.copy_(src)          # CPU memcpy

        # Buffers are tiny; rank 0 copies them all.
        for i, (name, off, nb, shape, dtype) in enumerate(buffer_layout):
            view = torch.frombuffer(
                memoryview(shm.buf)[off:off + nb], dtype=dtype,
            ).reshape(shape)
            self._cpu_buffers[name] = view
            if i % self._world_size == self._local_rank:
                self._owned_buffers.add(name)
                src = buf_dict[name].data
                if src.device.type == "cuda":
                    view.copy_(src)
                else:
                    view.copy_(src)

        # ── Step 4: Barrier — all copies complete ──────────────────────
        if dist.is_initialized():
            dist.barrier()

        # ── Step 5: Swap param.data → shm views on ALL ranks ──────────
        for name in self._cpu_params:
            if name in param_dict:
                param_dict[name].data = self._cpu_params[name]
        for name in self._cpu_buffers:
            if name in buf_dict:
                buf_dict[name].data = self._cpu_buffers[name]

        dt_ms = (time.perf_counter() - t_start) * 1000
        my_params = len(self._owned_params)
        my_bufs = len(self._owned_buffers)
        print(
            f"[{self._tag}] Parallel build (rank {self._local_rank}): "
            f"{len(param_layout)} params + {len(buffer_layout)} buffers, "
            f"{total_bytes / 1e9:.2f} GB shared — copied {my_params} params "
            f"+ {my_bufs} buffers ({dt_ms:.0f} ms, shm={self._shm_name})"
        )

    def _register_shm_buf(self, buf: memoryview, total_bytes: int) -> None:
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
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
            # Each rank refreshes only the params it owns
            if name in self._owned_params and name in self._cpu_params:
                self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            if name in self._owned_buffers and name in self._cpu_buffers:
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

    def _swap_to_cpu(self, module) -> int:
        """Point param.data directly at pinned CPU tensors."""
        restored = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data = cpu_buf
                restored += 1

        for name, buf in module.named_buffers():
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                buf.data = cpu_copy

        return restored

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        cuda_count = cpu_count = 0
        for name, param in module.named_parameters():
            if param.device.type == "cuda":
                if name in self._cpu_params:
                    # Each rank refreshes only the params it owns
                    if name in self._owned_params:
                        self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
                    self._collect_storage(param.data, freed_ptrs)
                    cuda_count += 1
                else:
                    # Param was on CPU at build time but ended up on CUDA
                    # (VRAM budget changed). Shared layout is fixed — we
                    # cannot add new entries. Skip freeing to avoid data loss;
                    # model.load() will move it to CPU next cycle.
                    logging.warning(
                        f"[{self._tag}] CUDA param '{name}' not in shared cache "
                        f"— skipping (cannot grow shared layout)."
                    )
            else:
                cpu_count += 1

        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                if name in self._cpu_buffers:
                    if name in self._owned_buffers:
                        self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
                    self._collect_storage(buf.data, freed_ptrs)

        return cuda_count, cpu_count

    # -- info --

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return self._total_bytes

    # -- cleanup --

    def cleanup(self) -> None:
        """Unpin, release shared memory, and clear all cached tensors."""
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_params else 0

        # 1. Drop all tensor views into shm FIRST — they hold references
        #    to the shm buffer that prevent close().
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._freed_storages = []
        self._total_bytes = 0

        # 1b. Force GC so tensor objects (and their memoryview refs into
        #     shm.buf) are destroyed BEFORE we close the SharedMemory
        #     handle.  Without this, lingering buffer exports cause
        #     "BufferError: cannot close exported pointers exist".
        import gc; gc.collect()

        # 2. Unregister CUDA host pinning.
        if self._registered_ptr is not None:
            _cuda_host_unregister(self._registered_ptr)
            self._registered_ptr = None

        # 3. Release via backend (handles close + unlink for writer).
        if self._shm_backend is not None and self._shm_metadata is not None:
            try:
                self._shm_backend.release_artifact(
                    self._shm_metadata, unlink=self._is_writer,
                )
            except Exception:
                pass
            self._shm_metadata = None

        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: freed {gb:.2f} GB shared pinned RAM.")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# FSDPShardPinnedCache — FSDP2 DTensor-aware
# ═══════════════════════════════════════════════════════════════════════════

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
    local = dt.to_local() # type: ignore
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
            dt._local_tensor.copy_(cpu_buf, non_blocking=non_blocking) # type: ignore
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
