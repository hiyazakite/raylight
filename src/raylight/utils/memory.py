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
    and stdlib.  It knows nothing about model formats, contexts, actors, or
    Ray.  Callers declare **intent** (``after_offload``, ``after_reload``,
    ``before_inference``); the policy decides *what* to do and *when*.

    Optionally, a pinned-cache and module can be registered via
    ``set_offload_target()`` to enable pressure-triggered partial offload
    (aimdo-style watermark eviction).  This is a soft reference —
    everything works without it.

    Designed to be instantiated once per ``RayActor`` (or per-process)
    and passed as a plain parameter wherever cleanup is needed.  Every
    public method is safe to call with ``policy=None`` via the module-level
    ``get_policy()`` / ``NullPolicy`` fallback — no sentinel checks needed
    at call sites.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        debounce_sec: float = 0.5,
        headroom_bytes: int = 256 * 1024**2,
    ):
        self._device = device
        self._debounce_sec = debounce_sec
        self._headroom_bytes = headroom_bytes
        self._last_cleanup: float = 0.0
        # Soft references for pressure-triggered partial offload (Phase 3/4).
        # Registered via set_offload_target(); None until then.
        self._pinned_cache = None
        self._offload_module = None
        # CompactFusion cache evictor — registered via set_compact_cache_evictor().
        # Called BEFORE weight eviction (cheaper to rebuild).
        self._compact_evictor = None
        # Guard: prevents watermark eviction while model weights are
        # actively being read during a forward pass.
        self._in_forward: bool = False

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

    def before_inference(self, needed_bytes: int = 0) -> int:
        """Ensure GPU is quiesced and has enough headroom before a forward pass.

        If *needed_bytes* > 0, checks VRAM pressure and proactively frees
        memory (PyTorch cache first, then watermark partial offload if a
        pinned cache is registered).  Mirrors aimdo's stopgap
        ``vbars_free(budget_deficit(0))`` at the top of every fault.

        Returns bytes freed (0 if no pressure).
        """
        self._sync()
        if needed_bytes > 0:
            return self.relieve_pressure(needed_bytes)
        return 0

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

    def vram_free_bytes(self) -> int:
        """Actual free VRAM from CUDA driver (not PyTorch's cached view).

        Uses ``torch.cuda.mem_get_info`` which queries ``cuMemGetInfo``
        — the same source aimdo's ``cuda_budget_deficit()`` uses.
        """
        if not torch.cuda.is_available():
            return 0
        try:
            free, _ = torch.cuda.mem_get_info(self._device)
            return free
        except Exception:
            return 0

    def vram_deficit(self, needed_bytes: int = 0) -> int:
        """How many bytes we need freed to satisfy *needed_bytes* + headroom.

        Returns 0 if we have enough room.  Mirrors aimdo's
        ``budget_deficit()`` from ``plat.h``.
        """
        free = self.vram_free_bytes()
        deficit = (needed_bytes + self._headroom_bytes) - free
        return max(0, deficit)

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

    # ─── Pressure relief ─────────────────────────────────────

    def set_offload_target(self, cache, module: torch.nn.Module) -> None:
        """Register a pinned cache + module for pressure-triggered partial offload.

        Soft reference — None-guarded everywhere.  ``MemoryPolicy`` remains
        fully usable without this.
        """
        self._pinned_cache = cache
        self._offload_module = module

    def set_compact_cache_evictor(self, evictor_fn) -> None:
        """Register a callable that frees CompactFusion CUDA caches.

        The callable should free as much as possible and return bytes freed.
        It is called BEFORE weight eviction (cheaper to rebuild — comm
        buffers are automatically recreated on the next attention call).
        """
        self._compact_evictor = evictor_fn

    def relieve_pressure(self, needed_bytes: int = 0) -> int:
        """Try to free VRAM under pressure, ordered by cost.

        Three-tier cascade:
          1. ``empty_cache()`` — free PyTorch's cached (unused) blocks.
          2. CompactFusion cache eviction — free comm buffers (cheap rebuild).
          3. Watermark partial offload via pinned cache (expensive reload).

        Returns actual bytes freed.  0 = no action needed.

        **Safety**: if ``_in_forward`` is True (model is executing a forward
        pass), step 3 is skipped — evicting weights that are being read
        would crash or produce garbage.  Steps 1–2 are always safe.
        """
        deficit = self.vram_deficit(needed_bytes)
        if deficit <= 0:
            return 0

        freed_total = 0

        # Step 1: Free PyTorch caching allocator's unused blocks
        self._release_cuda_cache()
        deficit = self.vram_deficit(needed_bytes)
        if deficit <= 0:
            return freed_total

        # Step 2: Evict CompactFusion caches (cheap to rebuild)
        evictor = self._compact_evictor
        if evictor is not None:
            try:
                freed = evictor()
                freed_total += freed
                self._release_cuda_cache()  # make freed blocks visible
                deficit = self.vram_deficit(needed_bytes)
                if deficit <= 0:
                    return freed_total
            except Exception as e:
                print(f"[MemoryPolicy] Compact cache eviction failed: {e}")

        # Step 3: Watermark partial offload (aimdo-style) — last resort
        # BLOCKED during forward pass — can't evict weights being read.
        if self._in_forward:
            return freed_total

        cache = self._pinned_cache
        module = self._offload_module
        if cache is not None and module is not None:
            # Allow eviction when the cache is built (pinned path) OR when
            # an mmap fallback is available (zero-alloc eviction).
            can_evict = getattr(cache, 'built', False) or getattr(cache, 'has_mmap_fallback', False)
            if can_evict:
                try:
                    freed = cache.offload_by_pressure(module, deficit)
                    freed_total += freed
                except Exception as e:
                    print(f"[MemoryPolicy] Pressure offload failed: {e}")

        return freed_total

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
    def before_inference(self, needed_bytes: int = 0) -> int: return 0
    def teardown(self) -> None: ...
    def log_vram(self, tag: str) -> None: ...
    def vram_free_bytes(self) -> int: return 2**63
    def vram_deficit(self, needed_bytes: int = 0) -> int: return 0
    def relieve_pressure(self, needed_bytes: int = 0) -> int: return 0
    def set_offload_target(self, cache, module) -> None: ...
    def set_compact_cache_evictor(self, evictor_fn) -> None: ...


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


def get_shm_usage_stats() -> dict[str, int]:
    """Return shared-memory usage telemetry by source.

    Keys:
      - file_mmap_bytes, file_mmap_count
    - pinned_shm_bytes, pinned_shm_count
    - gguf_shm_bytes, gguf_shm_count
      - legacy_pt_bytes, legacy_pt_count
      - total_bytes, total_count
    """
    file_mmap_bytes = 0
    file_mmap_count = 0
    pinned_shm_bytes = 0
    pinned_shm_count = 0
    gguf_shm_bytes = 0
    gguf_shm_count = 0
    legacy_pt_bytes = 0
    legacy_pt_count = 0

    try:
        from raylight.ipc.memory_stats import collect_all_stats
        from raylight.ipc.resolver import build_default_host_ipc_service

        stats = collect_all_stats(build_default_host_ipc_service())
        for src in stats:
            if src.kind_name == "file_mmap":
                file_mmap_bytes = src.artifact_bytes
                file_mmap_count = src.artifact_count
            elif src.kind_name == "pinned_shm":
                pinned_shm_bytes = src.artifact_bytes
                pinned_shm_count = src.artifact_count
            elif src.kind_name == "gguf_dequant_shm":
                gguf_shm_bytes = src.artifact_bytes
                gguf_shm_count = src.artifact_count
            elif src.kind_name == "legacy_pt":
                legacy_pt_bytes = src.artifact_bytes
                legacy_pt_count = src.artifact_count
    except Exception:
        # Keep memory reporting best-effort and non-fatal.
        pass

    total_bytes = file_mmap_bytes + pinned_shm_bytes + gguf_shm_bytes + legacy_pt_bytes
    total_count = file_mmap_count + pinned_shm_count + gguf_shm_count + legacy_pt_count

    return {
        "file_mmap_bytes": file_mmap_bytes,
        "file_mmap_count": file_mmap_count,
        "pinned_shm_bytes": pinned_shm_bytes,
        "pinned_shm_count": pinned_shm_count,
        "gguf_shm_bytes": gguf_shm_bytes,
        "gguf_shm_count": gguf_shm_count,
        "legacy_pt_bytes": legacy_pt_bytes,
        "legacy_pt_count": legacy_pt_count,
        "total_bytes": total_bytes,
        "total_count": total_count,
    }


def get_shm_usage_gb():
    """Returns total size of Raylight shared-memory artifacts in GB."""

    return get_shm_usage_stats()["total_bytes"] / 1024**3

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
    shm_stats = get_shm_usage_stats()
    shm = shm_stats["total_bytes"] / 1024**3
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
        print(f"Shared Mem : {shm:5.2f} GB (Raylight Host IPC + shared memory)")
        ipc_gb = shm_stats["file_mmap_bytes"] / 1024**3
        pinned_gb = shm_stats["pinned_shm_bytes"] / 1024**3
        gguf_shm_gb = shm_stats["gguf_shm_bytes"] / 1024**3
        legacy_gb = shm_stats["legacy_pt_bytes"] / 1024**3
        print(
            "           : "
            f"ipc_files={shm_stats['file_mmap_count']} | "
            f"pinned={shm_stats['pinned_shm_count']} | "
            f"gguf_shm={shm_stats['gguf_shm_count']} | "
            f"legacy_pt={shm_stats['legacy_pt_count']}"
        )
        print(
            "           : "
            f"ipc_gb={ipc_gb:.2f} | "
            f"pinned_gb={pinned_gb:.2f} | "
            f"gguf_shm_gb={gguf_shm_gb:.2f} | "
            f"legacy_pt_gb={legacy_gb:.2f}"
        )
    
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
