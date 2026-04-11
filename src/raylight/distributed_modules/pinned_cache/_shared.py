"""SharedPinnedParamCache — cross-process /dev/shm + cudaHostRegister."""

from __future__ import annotations

import ctypes
import logging
import time
from typing import TYPE_CHECKING, Dict, Optional

import torch

from ._base import BasePinnedCache
from ._helpers import _compute_layout, _cuda_host_register, _cuda_host_unregister

if TYPE_CHECKING:
    from raylight.ipc.backends.posix_shm import PosixShmBackend
    from raylight.ipc.types import HostIpcArtifactMetadata


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
        self._shm_metadata: Optional["HostIpcArtifactMetadata"] = None
        self._registered_ptr: Optional[int] = None
        self._total_bytes: int = 0

        # Deterministic round-robin assignment
        self._owned_params: set = set()
        self._owned_buffers: set = set()

    # ── Hook: restrict D2H refresh to owned params ────────────────────

    def _should_refresh(self, name: str, is_param: bool) -> bool:
        return name in (self._owned_params if is_param else self._owned_buffers)

    # ── Build ─────────────────────────────────────────────────────────

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

        self._register_shm_buf(shm.buf, total_bytes)

        # ── Step 3: Build ownership sets + create views ────────────────
        param_dict = dict(module.named_parameters())
        buf_dict = dict(module.named_buffers())

        for i, (name, off, nb, shape, dtype) in enumerate(param_layout):
            view = torch.frombuffer(
                memoryview(shm.buf)[off:off + nb], dtype=dtype,
            ).reshape(shape)
            self._cpu_params[name] = view
            if i % self._world_size == self._local_rank:
                self._owned_params.add(name)
                src = param_dict[name].data
                if src.device.type == "cuda":
                    view.copy_(src)
                else:
                    view.copy_(src)

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

    # ── Hooks ─────────────────────────────────────────────────────────

    def _pre_free_hook(self) -> None:
        """Barrier so readers don't free before writer finishes writing."""
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    # ── Snapshot override (safety: skip unmapped CUDA params) ─────────

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        cuda_count = cpu_count = 0
        for name, param in self._param_list:
            if param.device.type == "cuda":
                if name in self._cpu_params:
                    if name in self._owned_params:
                        self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
                    self._collect_storage(param.data, freed_ptrs)
                    cuda_count += 1
                else:
                    logging.warning(
                        f"[{self._tag}] CUDA param '{name}' not in shared cache "
                        f"— skipping (cannot grow shared layout)."
                    )
            else:
                cpu_count += 1

        for name, buf in self._buffer_list:
            if buf.device.type == "cuda":
                if name in self._cpu_buffers:
                    if name in self._owned_buffers:
                        self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
                    self._collect_storage(buf.data, freed_ptrs)

        return cuda_count, cpu_count

    # ── Info ──────────────────────────────────────────────────────────

    def pinned_ram_bytes(self) -> int:
        return self._total_bytes

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Unpin, release shared memory, and clear all cached tensors."""
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_params else 0

        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._freed_storages = []
        self._total_bytes = 0

        import gc; gc.collect()

        if self._registered_ptr is not None:
            _cuda_host_unregister(self._registered_ptr)
            self._registered_ptr = None

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
