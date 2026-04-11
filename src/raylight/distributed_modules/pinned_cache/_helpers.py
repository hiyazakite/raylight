"""Pure helper functions for pinned-cache layout, alignment, and CUDA host registration."""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

_ALIGNMENT = 512  # byte alignment per param slot (matches GPU DMA granularity)


def _align_up(offset: int, alignment: int = _ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


# ---------------------------------------------------------------------------
# CUDA host-register helpers (shared by SharedPinnedParamCache & GGUF pinning)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------

def make_cache_id(model_path: str) -> str:
    """Deterministic short hash from model path — identical across ranks."""
    return hashlib.md5(model_path.encode()).hexdigest()[:16]
