"""
SharedPinnedParamCache — cross-process shared pinned-RAM cache for data-parallel.

For data-parallel setups where N workers hold identical model weights, this
maintains a SINGLE shared-memory copy of the parameter data in pinned host RAM
instead of N private copies.  Cuts pinned RAM from N×model_size → 1×model_size.

Architecture
------------
- Uses ``multiprocessing.shared_memory.SharedMemory`` for cross-process access.
- One contiguous buffer in ``/dev/shm/`` holds all param + buffer data.
- ``cudaHostRegister`` (portable flag) pins the shared pages for async DMA.
- Rank 0 is the **writer** (creates segment, refreshes on offload).
- All other ranks are **readers** (attach to existing segment).
- ``reload_to_cuda()`` from multiple GPUs is safe — concurrent DMA reads from
  the same pinned pages do not conflict.

Lifecycle
---------
1. ``offload_to_cpu(module)`` — first call triggers lazy ``build()``:
       - **Writer**: computes layout, allocates ``SharedMemory``, copies CUDA →
         shared buffer, ``cudaHostRegister``, frees CUDA storages.
       - **Readers**: compute layout, wait for segment (retry loop), attach,
         ``cudaHostRegister``, free own CUDA storages.
   On subsequent calls the writer refreshes (LoRA may have been un-patched),
   readers just free CUDA.

2. ``reload_to_cuda(module)`` — all ranks:
       - ``storage.resize_(nbytes)`` to re-allocate CUDA.
       - ``param.data.copy_(shared_view, non_blocking=True)`` (DMA from pinned).
       - One ``synchronize()`` at the end.

3. ``cleanup()`` — ``cudaHostUnregister``, writer ``shm.unlink()``.

Coordination
------------
Since ``torch.distributed`` is initialised for all Ray workers (even DP),
``dist.barrier()`` is used after the writer finishes ``build()`` so readers
don't read partially-written data.
"""

from __future__ import annotations

import ctypes
import hashlib
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# CUDA host-register helpers
# ---------------------------------------------------------------------------

_ALIGNMENT = 512  # byte alignment per param slot (matches GPU DMA granularity)


def _align_up(offset: int, alignment: int = _ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def _cuda_host_register(ptr: int, size: int) -> bool:
    """Pin existing host pages for CUDA DMA via ``cudaHostRegister``."""
    try:
        # Ensure CUDA context exists (cudaHostRegister requires one).
        if not torch.cuda.is_available():
            return False
        torch.cuda.current_device()  # lazily initializes CUDA

        cudart = torch.cuda.cudart()
        # cudaHostRegisterPortable = 1  (visible to all CUDA contexts)
        # pybind11 binding expects plain Python ints, NOT ctypes wrappers.
        err = cudart.cudaHostRegister(int(ptr), int(size), int(1))
        if isinstance(err, tuple):
            err = err[0]
        # err is a cudaError enum — compare by value, not identity.
        ok = int(err) == 0
        if not ok:
            print(f"[SharedPinnedParamCache] cudaHostRegister returned error {err}")
        return ok
    except Exception as e:
        print(f"[SharedPinnedParamCache] cudaHostRegister exception: {e}")
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
# Layout: deterministic param→offset mapping
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

    total = _align_up(offset)
    return param_layout, buffer_layout, total


# ---------------------------------------------------------------------------
# SharedPinnedParamCache
# ---------------------------------------------------------------------------

class SharedPinnedParamCache:
    """Cross-process shared pinned-RAM cache for data-parallel workers.

    Same public interface as ``PinnedParamCache`` so callers are agnostic.
    """

    def __init__(self, cache_id: str, is_writer: bool = False) -> None:
        self._cache_id = cache_id
        self._is_writer = is_writer

        self._shm_name = f"raylight_pc_{cache_id}"
        self._shm: Optional[SharedMemory] = None
        self._registered_ptr: Optional[int] = None
        self._total_bytes: int = 0

        # Tensor views into the shared buffer (name → tensor)
        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}

        # Per-rank CUDA storage tracking
        self._freed_storages: List[Tuple] = []   # (UntypedStorage, original_nbytes)
        self.params_on_cuda: bool = False
        self._built: bool = False

    # ----------------------------------------------------------------- build

    def build(self, module: torch.nn.Module) -> None:
        """Lazy build: writer allocates, readers attach."""
        if self._built:
            return

        param_layout, buffer_layout, total_bytes = _compute_layout(module)
        self._total_bytes = total_bytes

        if total_bytes == 0:
            print("[SharedPinnedParamCache] No CUDA params found — nothing to cache.")
            self._built = True
            return

        if self._is_writer:
            self._build_writer(module, param_layout, buffer_layout, total_bytes)
        else:
            self._build_reader(module, param_layout, buffer_layout, total_bytes)

        self._built = True
        self.params_on_cuda = True

    # -- writer --

    def _build_writer(self, module, param_layout, buffer_layout, total_bytes):
        # Clean up stale segment with same name (e.g. previous run crashed)
        try:
            old = SharedMemory(name=self._shm_name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self._shm = SharedMemory(
            name=self._shm_name, create=True, size=total_bytes
        )

        # Pin for DMA
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        ok = _cuda_host_register(ptr, total_bytes)
        self._registered_ptr = ptr
        if not ok:
            print("[SharedPinnedParamCache] WARNING: cudaHostRegister failed — "
                  "DMA will use staging buffer (slower).")

        # Create views and copy CUDA → shared buffer
        param_dict = dict(module.named_parameters())
        for name, off, nb, shape, dtype in param_layout:
            byte_view = memoryview(self._shm.buf)[off : off + nb]
            typed_view = torch.frombuffer(byte_view, dtype=dtype).reshape(shape)
            typed_view.copy_(param_dict[name].data.cpu())
            self._cpu_params[name] = typed_view

        buf_dict = dict(module.named_buffers())
        for name, off, nb, shape, dtype in buffer_layout:
            byte_view = memoryview(self._shm.buf)[off : off + nb]
            typed_view = torch.frombuffer(byte_view, dtype=dtype).reshape(shape)
            typed_view.copy_(buf_dict[name].data.cpu())
            self._cpu_buffers[name] = typed_view

        # Signal readers via barrier
        if dist.is_initialized():
            dist.barrier()

        print(
            f"[SharedPinnedParamCache] Writer built: {len(param_layout)} params + "
            f"{len(buffer_layout)} buffers, {total_bytes / 1e9:.2f} GB shared pinned RAM "
            f"(shm={self._shm_name})"
        )

    # -- reader --

    def _build_reader(self, module, param_layout, buffer_layout, total_bytes):
        # Wait for writer via barrier
        if dist.is_initialized():
            dist.barrier()

        # Attach to existing segment
        retries = 50
        for i in range(retries):
            try:
                self._shm = SharedMemory(name=self._shm_name, create=False)
                break
            except FileNotFoundError:
                if i < retries - 1:
                    time.sleep(0.2)
        else:
            raise RuntimeError(
                f"[SharedPinnedParamCache] Reader timed out waiting for "
                f"shared memory '{self._shm_name}' (10s)."
            )

        if self._shm.size < total_bytes:
            raise RuntimeError(
                f"[SharedPinnedParamCache] Size mismatch: expected {total_bytes}, "
                f"got {self._shm.size}."
            )

        # Pin for this process
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        ok = _cuda_host_register(ptr, total_bytes)
        self._registered_ptr = ptr
        if not ok:
            print("[SharedPinnedParamCache] WARNING: cudaHostRegister failed on reader.")

        # Create views (data already present from writer)
        for name, off, nb, shape, dtype in param_layout:
            byte_view = memoryview(self._shm.buf)[off : off + nb]
            typed_view = torch.frombuffer(byte_view, dtype=dtype).reshape(shape)
            self._cpu_params[name] = typed_view

        for name, off, nb, shape, dtype in buffer_layout:
            byte_view = memoryview(self._shm.buf)[off : off + nb]
            typed_view = torch.frombuffer(byte_view, dtype=dtype).reshape(shape)
            self._cpu_buffers[name] = typed_view

        print(
            f"[SharedPinnedParamCache] Reader attached: {len(param_layout)} params, "
            f"{total_bytes / 1e9:.2f} GB shared (shm={self._shm_name})"
        )

    # --------------------------------------------------------------- offload

    def offload_to_cpu(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Offload CUDA → shared pinned RAM, free VRAM via storage resize.

        Writer refreshes the shared buffer (base weights may have changed).
        Readers only free their own CUDA storages.
        """
        if not self._built:
            self.build(module)

        if not self.params_on_cuda:
            print("[SharedPinnedParamCache] Already offloaded — skipping.")
            return 0

        offloaded = 0
        freed_ptrs: set = set()
        self._freed_storages = []

        # --- Parameters ---
        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                continue

            # Writer refreshes shared buffer; readers skip the copy
            if self._is_writer and name in self._cpu_params:
                self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)

            # Collect unique CUDA storages (all ranks)
            s = param.data.untyped_storage()
            ptr = s.data_ptr()
            if ptr != 0 and ptr not in freed_ptrs:
                freed_ptrs.add(ptr)
                self._freed_storages.append((s, s.nbytes()))
            offloaded += 1

        # --- Buffers ---
        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            if self._is_writer and name in self._cpu_buffers:
                self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)

            s = buf.data.untyped_storage()
            ptr = s.data_ptr()
            if ptr != 0 and ptr not in freed_ptrs:
                freed_ptrs.add(ptr)
                self._freed_storages.append((s, s.nbytes()))

        # Sync D2H copies (writer), then barrier so readers don't free before
        # writer finishes writing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()

        # Free all unique CUDA storages
        for s, _ in self._freed_storages:
            s.resize_(0)

        torch.cuda.empty_cache()

        self.params_on_cuda = False
        freed_gb = sum(nb for _, nb in self._freed_storages) / 1e9
        tag = "Writer" if self._is_writer else "Reader"
        print(
            f"[SharedPinnedParamCache] {tag} offloaded {offloaded} params, "
            f"freed {len(self._freed_storages)} CUDA storages ({freed_gb:.2f} GB)."
        )
        return offloaded

    # --------------------------------------------------------------- reload

    def reload_to_cuda(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Restore model from shared pinned RAM → CUDA."""
        if not self._built:
            print("[SharedPinnedParamCache] Not built — cannot reload.")
            return 0
        if self.params_on_cuda:
            print("[SharedPinnedParamCache] Already on CUDA — skipping.")
            return 0

        torch.cuda.empty_cache()

        # (1) Re-allocate CUDA storages
        restored = 0
        for s, nbytes in self._freed_storages:
            s.resize_(nbytes)
            restored += 1
        self._freed_storages = []

        # (2) Copy shared pinned → CUDA (all ranks read concurrently — safe)
        reloaded = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is None:
                continue
            param.data.copy_(cpu_buf, non_blocking=non_blocking)
            reloaded += 1

        for name, buf in module.named_buffers():
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is None:
                continue
            buf.data.copy_(cpu_copy, non_blocking=non_blocking)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.params_on_cuda = True
        print(f"[SharedPinnedParamCache] Reloaded {reloaded} params to CUDA.")
        return reloaded

    # --------------------------------------------------------------- helpers

    @property
    def built(self) -> bool:
        return self._built

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return self._total_bytes

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


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_cache_id(model_path: str) -> str:
    """Deterministic short ID from model path — all ranks compute the same value."""
    return hashlib.md5(model_path.encode()).hexdigest()[:16]
