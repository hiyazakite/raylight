import os
import sys
import glob
import time
import ctypes
from dataclasses import dataclass
from typing import Optional

import torch
import psutil
import gc


# ---------------------------------------------------------------------------
# MemoryPolicy — centralised memory lifecycle decisions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory view.  Read-only, safe to log / serialise."""
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    rss_gb: float = 0.0
    shm_gb: float = 0.0
    mmap_gb: float = 0.0


class MemoryPolicy:
    """Centralised memory cleanup / query controller.

    **Coupling contract** — this class depends *only* on PyTorch, the OS,
    and stdlib.  It knows nothing about model formats, contexts, workers, or
    Ray.  Callers declare **intent** (``after_offload``, ``after_reload``,
    ``before_inference``); the policy decides *what* to do and *when*.

    Designed to be instantiated once per ``RayWorker`` (or per-process)
    and passed as a plain parameter wherever cleanup is needed.  Every
    public method is safe to call with ``policy=None`` via the module-level
    ``get_policy()`` / ``NullPolicy`` fallback — no sentinel checks needed
    at call sites.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        debounce_sec: float = 0.5,
    ):
        self._device = device
        self._debounce_sec = debounce_sec
        self._last_cleanup: float = 0.0

    # ─── Intent-based API ────────────────────────────────────

    def after_offload(self) -> None:
        """Full cleanup after any offload path."""
        self._sync()
        self._debounced_gc()
        self._release_cuda_cache()
        self._malloc_trim()

    def after_reload(self) -> None:
        """Light sync after hot-reload — avoid gc before inference."""
        self._sync()

    def after_model_swap(self) -> None:
        """Called between tear-down of old model and load of new one."""
        self._debounced_gc()
        self._release_cuda_cache()
        self._malloc_trim()

    def before_inference(self) -> None:
        """Ensure GPU is quiesced before a forward pass."""
        self._sync()

    def teardown(self) -> None:
        """Aggressive cleanup — final release, bypass debounce."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self._malloc_trim()

    # ─── Queries ─────────────────────────────────────────────

    def vram_allocated_gb(self) -> float:
        """Current VRAM allocated (GB) on this policy's device."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.memory_allocated(self._device) / 1e9
        except Exception:
            return 0.0

    def snapshot(self) -> MemorySnapshot:
        """Point-in-time memory view for logging."""
        va, vr = get_vram_gb(self._device)
        return MemorySnapshot(
            vram_allocated_gb=va,
            vram_reserved_gb=vr,
            rss_gb=get_process_rss_gb(),
            shm_gb=get_shm_usage_gb(),
            mmap_gb=get_gguf_mmap_gb(),
        )

    def log_vram(self, tag: str) -> None:
        """One-line VRAM log — replaces ad-hoc memory_allocated prints."""
        print(f"[{tag}] VRAM: {self.vram_allocated_gb():.2f} GB")

    # ─── Private ─────────────────────────────────────────────

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def _release_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _debounced_gc(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < self._debounce_sec:
            return
        self._last_cleanup = now
        gc.collect()
        try:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    @staticmethod
    def _malloc_trim() -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


class _NullPolicy(MemoryPolicy):
    """No-op drop-in — all intent methods silently do nothing.

    Used as the default when no real policy is available, so callers
    never need ``if policy is not None:`` guards.
    """

    def after_offload(self) -> None: ...
    def after_reload(self) -> None: ...
    def after_model_swap(self) -> None: ...
    def before_inference(self) -> None: ...
    def teardown(self) -> None: ...
    def log_vram(self, tag: str) -> None: ...


#: Module-level null fallback — importers can use as default parameter.
NULL_POLICY: MemoryPolicy = _NullPolicy()


# ---------------------------------------------------------------------------
# Original helper functions (unchanged)
# ---------------------------------------------------------------------------

def get_system_ram_gb():
    """Returns (used_gb, total_gb, available_gb, cached_gb)."""
    try:
        vm = psutil.virtual_memory()
        cached = getattr(vm, "cached", 0)
        return (vm.used / 1024**3, vm.total / 1024**3, vm.available / 1024**3, cached / 1024**3)
    except Exception:
        return (0, 0, 0, 0)

def get_process_rss_gb():
    """Returns RSS of current process in GB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    except Exception:
        return 0

def get_vram_gb(device=None):
    """Returns (allocated_gb, reserved_gb) for the specified or current device."""
    if not torch.cuda.is_available():
        return (0, 0)
        
    try:
        if device is not None:
             with torch.cuda.device(device):
                 alloc = torch.cuda.memory_allocated()
                 res = torch.cuda.memory_reserved()
        else:
             alloc = torch.cuda.memory_allocated()
             res = torch.cuda.memory_reserved()
        return (alloc / 1024**3, res / 1024**3)
    except Exception:
        return (0, 0)

def get_shm_usage_gb():
    """Returns total size of Raylight cache files in /dev/shm in GB."""
    total_bytes = 0
    try:
        files = glob.glob("/dev/shm/raylight_*.pt")
        for f in files:
            try:
                total_bytes += os.path.getsize(f)
            except OSError:
                pass
    except Exception:
        pass
    return total_bytes / 1024**3

def get_gguf_mmap_gb():
    """Returns total size of .gguf files mapped in /proc/self/maps in GB."""
    total_bytes = 0
    try:
        if os.path.exists("/proc/self/maps"):
            with open("/proc/self/maps", "r") as f:
                for line in f:
                    if ".gguf" in line:
                        parts = line.split()
                        if len(parts) > 0:
                            # format: 7f7f...-7f7f... r--p ...
                            addr_range = parts[0]
                            start_hex, end_hex = addr_range.split("-")
                            start = int(start_hex, 16)
                            end = int(end_hex, 16)
                            total_bytes += (end - start)
    except Exception:
        pass
    return total_bytes / 1024**3

def log_memory_stats(tag="Memory", device=None):
    """Logs comprehensive memory statistics to stdout."""
    # Data collection
    sys_used, sys_total, sys_avail, sys_cached = get_system_ram_gb()
    rss = get_process_rss_gb()
    vram_alloc, vram_res = get_vram_gb(device)
    shm = get_shm_usage_gb()
    mmap = get_gguf_mmap_gb()
    
    # Formatting
    header = f"[{tag}] Memory Stats"
    print(f"{header:-^60}")
    
    # System Level
    print(f"System RAM : {sys_used:5.2f} GB Used / {sys_total:5.2f} GB Total")
    print(f"           : {sys_avail:5.2f} GB Available | {sys_cached:5.2f} GB Cached (Benign)")
    
    # Process Level
    print(f"Process RSS: {rss:5.2f} GB (This Process)")
    
    # VRAM
    if vram_res > 0:
        frag = vram_res - vram_alloc
        print(f"VRAM       : {vram_alloc:5.2f} GB Alloc / {vram_res:5.2f} GB Rsrvd | Frag: {frag:5.2f} GB")
        
    # Shared Resources
    if shm > 0:
        print(f"Shared Mem : {shm:5.2f} GB (Raylight /dev/shm cache)")
    
    if mmap > 0:
        print(f"GGUF Mmap  : {mmap:5.2f} GB (Mapped into this process)")
        
    print("-" * 60)


class monitor_memory:
    """Context manager to log memory stats before and after a block."""
    def __init__(self, tag="Block", device=None):
        self.tag = tag
        self.device = device
        self.peak_start = 0

    def __enter__(self):
        log_memory_stats(f"[START] {self.tag}", self.device)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        except:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "SUCCESS"
        
        peak_gb = 0.0
        try:
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated(self.device)
                peak_gb = peak_bytes / 1024**3
        except:
            pass
            
        # print(f"[{'END:' + status}] {self.tag} | Peak VRAM Delta: {peak_gb:.2f} GB")
        pass
        log_memory_stats(f"[END:{status}] {self.tag}", self.device)
