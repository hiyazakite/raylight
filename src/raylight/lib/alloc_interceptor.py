"""CUDA allocator interceptor — on-demand pressure relief.

Loads ``raylight_alloc.so`` (pre-built, ships with Raylight) and registers
a Python callback that fires when any ``cudaMalloc`` or ``cuMemCreate``
(VMM / expandable_segments) fails.  The callback triggers
``offload_by_pressure()`` to free model weights, then the interceptor
retries the allocation.

This gives us aimdo-style on-demand eviction without the full VBAR
virtual memory system — just the "intercept → free → retry" pattern.

**Graceful fallback**: if the ``.so`` is missing, fails to load, or the
platform isn't Linux, everything still works — the Python-level pressure
checks in ``MemoryPolicy.before_inference()`` remain as the safety net.

Usage (called once per actor process, after model load)::

    from raylight.lib.alloc_interceptor import install_interceptor

    install_interceptor(memory_policy)
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from raylight.utils.memory import MemoryPolicy

# ctypes callback: uint64_t (*)(uint64_t)
PRESSURE_CB = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint64)

_lib: Optional[ctypes.CDLL] = None
_callback_ref: Any = None  # ctypes callback ref — prevent GC
_installed: bool = False


def _find_so() -> Optional[str]:
    """Locate raylight_alloc.so next to this file."""
    here = Path(__file__).parent
    candidates = [
        here / "raylight_alloc.so",
        here / "lib" / "raylight_alloc.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _load_lib() -> Optional[ctypes.CDLL]:
    """Load the .so with RTLD_GLOBAL so our cudaMalloc overrides the default."""
    global _lib
    if _lib is not None:
        return _lib

    if platform.system() != "Linux":
        print("[alloc_interceptor] Not Linux — skipping CUDA interceptor.")
        return None

    so_path = _find_so()
    if so_path is None:
        print("[alloc_interceptor] raylight_alloc.so not found — "
              "falling back to Python-level pressure checks.")
        return None

    try:
        # RTLD_NOW (0x2) | RTLD_GLOBAL (0x100) = 0x102 = 258
        # RTLD_GLOBAL makes our cudaMalloc/cudaFree symbols override the
        # default ones from libcudart.so for all subsequently loaded libs.
        _lib = ctypes.CDLL(so_path, mode=258)

        # Set up function signatures
        _lib.raylight_alloc_set_callback.argtypes = [PRESSURE_CB]
        _lib.raylight_alloc_set_callback.restype = None

        _lib.raylight_alloc_set_log_level.argtypes = [ctypes.c_int]
        _lib.raylight_alloc_set_log_level.restype = None

        _lib.raylight_alloc_is_active.argtypes = []
        _lib.raylight_alloc_is_active.restype = ctypes.c_bool

        print(f"[alloc_interceptor] Loaded {so_path}")
        return _lib
    except Exception as e:
        print(f"[alloc_interceptor] Failed to load {so_path}: {e}")
        return None


def install_interceptor(
    memory: "MemoryPolicy",
    log_level: int = 1,
) -> bool:
    """Install the CUDA allocator interceptor with pressure callback.

    Args:
        memory: The MemoryPolicy with a registered pinned cache
                (via ``set_offload_target``).
        log_level: 0=silent, 1=warnings, 2=info, 3=debug.

    Returns:
        True if interceptor is active, False if fallback.
    """
    global _callback_ref, _installed

    if _installed:
        return True

    lib = _load_lib()
    if lib is None:
        return False

    lib.raylight_alloc_set_log_level(log_level)

    def _pressure_callback(needed_bytes: int) -> int:
        """Called from C when cudaMalloc/cudaMallocAsync/cuMemCreate fails.

        Must free at least `needed_bytes` of VRAM and return bytes freed.
        Re-entrancy is handled in C (thread-local guard).
        """
        try:
            freed = memory.relieve_pressure(needed_bytes=needed_bytes)
            return freed
        except Exception as e:
            print(f"[alloc_interceptor] Callback error: {e}")
            return 0

    # Wrap in ctypes callback and prevent GC
    _callback_ref = PRESSURE_CB(_pressure_callback)
    lib.raylight_alloc_set_callback(_callback_ref)

    _installed = True
    preloaded = os.environ.get('LD_PRELOAD', '')
    if 'raylight_alloc' in preloaded:
        print("[alloc_interceptor] CUDA allocator interceptor active "
              "(LD_PRELOAD) — intercepting cudaMalloc + cuMemCreate (VMM/expandable_segments). "
              "OOM will trigger automatic weight eviction.")
    else:
        print("[alloc_interceptor] CUDA allocator interceptor active "
              "(ctypes fallback, no LD_PRELOAD) — interception may not "
              "override PyTorch's resolved cudaMalloc/cuMemCreate symbols.")
    return True


def uninstall_interceptor() -> None:
    """Remove the pressure callback (allocations pass through unchanged)."""
    global _callback_ref, _installed
    if _lib is not None:
        _lib.raylight_alloc_set_callback(PRESSURE_CB(0))
    _callback_ref = None
    _installed = False


def is_active() -> bool:
    """Check if the interceptor is currently active."""
    return _installed and _lib is not None and _lib.raylight_alloc_is_active()
