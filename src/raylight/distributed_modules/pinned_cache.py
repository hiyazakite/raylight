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
    from raylight.ipc.types import HostIpcArtifactMetadata

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

        # ── Phase 2: Signature-based skip-upload ──
        # Generation counter bumped on every offload.  If storages survive
        # (nbytes still match) on reload, we skip the H2D copy entirely —
        # equivalent to aimdo's vbar_signature_compare().
        self._offload_generation: int = 0
        self._force_next_reload: bool = False

        # ── Phase 3: Watermark partial offload ──
        # Load-order param names (populated on build).  offload_by_pressure()
        # walks this in reverse to evict tail (deepest) layers first.
        self._param_order: List[str] = []
        self._watermark: int = 0  # 0 = nothing evicted, len = all evicted
        self._partial_freed: Dict[str, Tuple[torch.UntypedStorage, int]] = {}

        # ── Read-only source optimisation ──
        # When True, the CPU-side data is authoritative and never stale
        # relative to CUDA (e.g. GGUF mmap_cache).  offload_by_pressure()
        # skips the redundant CUDA→CPU snapshot before freeing VRAM.
        self._skip_d2h_refresh: bool = False

        # ── Mmap fallback for pressure relief without built pinned cache ──
        # When the pinned cache hasn't been built yet, we can still evict
        # CUDA weights: just free the storage and reload from the mmap'd
        # safetensor later.  No extra RAM allocation needed.
        self._mmap_sd: Optional[Dict[str, torch.Tensor]] = None
        self._mmap_key_map: Optional[Dict[str, str]] = None  # param_name -> mmap_key

        # ── Phase 3: Execution-distance-aware eviction ──
        # Per-block param name groups in execution order, and the index of
        # the currently-executing block.  When set, offload_by_pressure()
        # evicts blocks farthest from the cursor first (already-computed
        # blocks before far-future blocks), rather than fixed tail-first.
        self._block_param_groups: Optional[List[frozenset]] = None
        self._execution_cursor: int = -1  # -1 = not in forward

    # -------------------------------------------------------------- public

    def set_block_structure(self, groups: list) -> None:
        """Register per-block param name groups in execution order.

        Called once by ``_install_eviction_hooks()`` after discovering
        transformer blocks.  *groups* is a list of ``frozenset[str]``
        ordered by sequential execution (block 0 first).
        """
        self._block_param_groups = list(groups)

    def set_execution_cursor(self, idx: int) -> None:
        """Update the currently-executing block index.

        Called by each block's pre-forward hook.  ``-1`` resets to
        the default tail-first ordering.
        """
        self._execution_cursor = idx

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
        # Record load-order for watermark partial offload (Phase 3).
        self._param_order = [name for name, _ in module.named_parameters()]
        # Detect actual state: if ANY param is on CUDA, treat as on-CUDA.
        self._on_cuda = any(
            p.device.type == "cuda" for p in module.parameters()
        )

    def set_mmap_fallback(self, mmap_sd: Dict[str, torch.Tensor]) -> None:
        """Register an mmap-backed state dict for zero-alloc pressure eviction.

        When the pinned cache hasn't been built yet, ``offload_by_pressure``
        can still free VRAM: it just drops the CUDA storage, knowing the
        data is recoverable from the mmap source at reload time.  This
        avoids allocating a pinned RAM buffer, which is critical in
        RAM-constrained settings with multiple actors.
        """
        self._mmap_sd = mmap_sd

    def _resolve_mmap_key(self, param_name: str) -> Optional[str]:
        """Find the mmap_sd key corresponding to a module parameter name.

        Handles prefix differences: mmap keys may be raw or prefixed with
        ``diffusion_model.`` / ``model.diffusion_model.``.
        Builds and caches the key map on first call.
        """
        if self._mmap_sd is None:
            return None
        if self._mmap_key_map is None:
            self._mmap_key_map = {}
            mmap_keys = set(self._mmap_sd.keys())
            for name in (self._param_order or []):
                # Direct match
                if name in mmap_keys:
                    self._mmap_key_map[name] = name
                    continue
                # Try stripping common prefixes from mmap keys
                found = False
                for prefix in ('diffusion_model.', 'model.diffusion_model.'):
                    candidate = prefix + name
                    if candidate in mmap_keys:
                        self._mmap_key_map[name] = candidate
                        found = True
                        break
                if not found:
                    # Reverse: check if param name has a prefix the mmap key doesn't
                    for prefix in ('diffusion_model.', 'model.diffusion_model.'):
                        if name.startswith(prefix):
                            stripped = name[len(prefix):]
                            if stripped in mmap_keys:
                                self._mmap_key_map[name] = stripped
                                break
        return self._mmap_key_map.get(param_name)

    @property
    def has_mmap_fallback(self) -> bool:
        """True if mmap fallback is available for pressure eviction."""
        return self._mmap_sd is not None

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
        self._offload_generation += 1
        for s, _ in self._freed_storages:
            s.resize_(0)

        # Clear any partial-offload state (full offload supersedes)
        self._partial_freed.clear()
        self._watermark = 0

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
        self._offload_generation += 1
        for s, _ in self._freed_storages:
            s.resize_(0)

        # Clear any partial-offload state (full offload supersedes)
        self._partial_freed.clear()
        self._watermark = 0

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

        # Drain any leftover partial-offload state (safety net: if
        # reload_evicted() didn't run after offload_by_pressure()).
        self._partial_freed.clear()
        self._watermark = 0

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

        # ── Phase 2: Signature check — skip H2D if CUDA data survived ──
        # If storages weren't actually freed (nbytes still match original),
        # the GPU data is still valid.  Skip the entire resize+copy.
        # Mirrors aimdo's vbar_signature_compare().
        if (
            not self._force_next_reload
            and self._freed_storages
            and all(s.nbytes() == expected for s, expected in self._freed_storages)
        ):
            self._freed_storages = []
            self._on_cuda = True
            print(
                f"[{self._tag}] Signature match — CUDA data still valid, "
                f"skipping H2D copy (gen={self._offload_generation})."
            )
            return 0
        self._force_next_reload = False

        t_start = time.perf_counter()

        # NOTE: intentionally no empty_cache() here — the caching allocator
        # may still hold freed blocks from a previous model being unloaded;
        # those can be reused by resize_() below without hitting cudaMalloc.
        # The offload path already called empty_cache() when it freed VRAM.

        # Re-allocate all freed CUDA storages
        restored = 0
        alloc_bytes = 0

        # Include any partially-freed storages from watermark offload
        # (safety net: if reload_evicted() didn't run after
        # offload_by_pressure(), e.g. exception during sampling).
        if self._partial_freed:
            for name, (s, nbytes) in self._partial_freed.items():
                self._freed_storages.append((s, nbytes))
            self._partial_freed.clear()
            self._watermark = 0

        # ── Cold start: cache built from CPU params, no prior offload ──
        # _freed_storages is empty because no offload cycle has freed any
        # CUDA storages yet.  Params are still pointing at CPU/pinned
        # tensors.  We must allocate fresh CUDA tensors, copy pinned data
        # there, and swap param.data to the new CUDA tensors.
        if not self._freed_storages and self._offload_generation == 0:
            reloaded, alloc_bytes = self._cold_start_to_cuda(
                module, non_blocking,
            )
            t_alloc = t_copy = time.perf_counter()
        else:
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

    def _cold_start_to_cuda(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> tuple:
        """First-ever reload: allocate fresh CUDA tensors from pinned cache.

        Unlike the normal reload path (resize freed storages + copy), this
        allocates brand-new CUDA tensors and swaps ``param.data`` to point
        at them.  The previous CPU/pinned data references are released.

        Returns ``(reloaded_count, alloc_bytes)``.
        """
        reloaded = 0
        alloc_bytes = 0
        device = None

        # Determine target CUDA device from first available GPU
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if device is None:
            print(f"[{self._tag}] Cold start: no CUDA device available.")
            return 0, 0

        cpu_params = getattr(self, "_cpu_params", {})
        cpu_buffers = getattr(self, "_cpu_buffers", {})

        for name, param in module.named_parameters():
            cpu_buf = cpu_params.get(name)
            if cpu_buf is not None:
                cuda_tensor = torch.empty_like(
                    cpu_buf, device=device,
                )
                cuda_tensor.copy_(cpu_buf, non_blocking=non_blocking)
                param.data = cuda_tensor
                alloc_bytes += cuda_tensor.nbytes
                reloaded += 1

        for name, buf in module.named_buffers():
            cpu_copy = cpu_buffers.get(name)
            if cpu_copy is not None:
                cuda_tensor = torch.empty_like(
                    cpu_copy, device=device,
                )
                cuda_tensor.copy_(cpu_copy, non_blocking=non_blocking)
                buf.data = cuda_tensor
                alloc_bytes += cuda_tensor.nbytes

        print(
            f"[{self._tag}] Cold start: allocated {alloc_bytes / 1e9:.2f} GB "
            f"of fresh CUDA tensors for {reloaded} params."
        )
        return reloaded, alloc_bytes

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

    # ── Phase 2: Invalidation ─────────────────────────────────

    def invalidate(self) -> None:
        """Mark CUDA data as dirty — next reload always does a full H2D copy.

        Call after in-place weight modification (e.g. LoRA baking) that
        makes the pinned cache stale relative to CUDA.
        """
        self._force_next_reload = True

    # ── Phase 3: Watermark partial offload ────────────────────

    def _get_cpu_buf(self, name: str) -> Optional[torch.Tensor]:
        """Return the pinned CPU tensor for *name*, or None.

        Subclasses that use a dict named ``_cpu_params`` get this for free.
        Override if the storage layout is different.
        """
        cpu_params = getattr(self, "_cpu_params", {})
        return cpu_params.get(name)

    def _set_cpu_buf(self, name: str, tensor: torch.Tensor) -> None:
        """Store a pinned CPU tensor for *name*.

        Used by incremental eviction to cache individual params on-the-fly
        without building the full contiguous slab.
        """
        cpu_params = getattr(self, "_cpu_params", None)
        if cpu_params is None:
            self._cpu_params = {}
            cpu_params = self._cpu_params
        cpu_params[name] = tensor

    def offload_by_pressure(
        self,
        module: torch.nn.Module,
        target_bytes: int,
        non_blocking: bool = True,
        exclude: Optional[set] = None,
    ) -> int:
        """Free at least *target_bytes* of VRAM by offloading tail params.

        Walks parameters in **reverse load-order** (deepest layers first).
        Stops as soon as cumulative freed bytes >= target_bytes.  Mirrors
        aimdo's watermark eviction that scans from the VBAR tail.

        Parameters in *exclude* (the currently-executing block) are skipped,
        equivalent to aimdo's pinned-page guard.

        Three source modes, chosen automatically:

        - **built** — copies CUDA → existing pinned slab views before freeing.
        - **mmap** — frees CUDA storage; data recoverable from mmap source.
        - **incremental** — allocates a pinned buffer per evicted param
          on-the-fly, copies CUDA → pinned, then frees.  Only the evicted
          layers consume host RAM; the rest stay CUDA-only.  Pinned buffers
          persist in ``_cpu_params`` for fast reload on subsequent cycles.

        Does NOT set ``_on_cuda = False`` — the model is now **partially
        resident** (some params on CUDA, some freed).

        Returns actual bytes freed.
        """
        if target_bytes <= 0:
            return 0

        # Pressure-driven eviction frees each CUDA storage immediately after
        # snapshotting it. Unlike the full offload path we cannot queue async
        # D2H copies and synchronize later, because that would let resize_(0)
        # invalidate the source storage while the copy is still in flight.
        # Keep these copies synchronous so repeated callback rounds remain safe.
        copy_non_blocking = False

        use_mmap = False
        use_incremental = False
        if not self._built:
            if self._mmap_sd is not None:
                use_mmap = True
            else:
                use_incremental = True

        # Lazy-init param_order if not yet populated (mmap path, no build)
        if not self._param_order:
            self._param_order = [name for name, _ in module.named_parameters()]
            # Also build the mmap key map now that we have param names
            self._mmap_key_map = None  # force rebuild

        freed = 0
        param_dict = dict(module.named_parameters())

        # ── Build eviction walk order ─────────────────────────────────
        # Phase 3: when a block structure + cursor is active, evict by
        # execution distance instead of fixed reverse-load-order.
        #   Priority 1: already-computed blocks (cursor-1 → 0) — safe,
        #               won't be needed again this timestep.
        #   Priority 2: farthest-future blocks (N-1 → cursor+2) — not
        #               needed for a while.
        #   Fallback:   reverse _param_order (Phase 1 behaviour).
        evict_order: Optional[List[str]] = None
        groups = self._block_param_groups
        cursor = self._execution_cursor
        if groups is not None and 0 <= cursor < len(groups):
            evict_order = []
            # Already-computed blocks, most-recent first
            for bi in range(cursor - 1, -1, -1):
                evict_order.extend(groups[bi])
            # Farthest-future blocks, farthest first
            for bi in range(len(groups) - 1, cursor + 1, -1):
                evict_order.extend(groups[bi])

        walk = evict_order if evict_order else reversed(self._param_order)

        print(f"[{self._tag}] Eviction starting: target={target_bytes/1e9:.2f} GB, "
              f"already_evicted={len(self._partial_freed)}/{len(self._param_order)}")

        evict_count = 0
        for name in walk:
            if freed >= target_bytes:
                break
            if name in self._partial_freed:
                continue
            if exclude and name in exclude:
                continue  # protected — currently executing block
            param = param_dict.get(name)
            if param is None or param.device.type != "cuda":
                continue

            if use_mmap:
                # Mmap fallback: verify the mmap source exists before freeing.
                # No CPU copy needed — data is recoverable from mmap.
                mmap_key = self._resolve_mmap_key(name)
                if mmap_key is None:
                    continue  # can't recover this param, skip
            elif not self._skip_d2h_refresh:
                # Pinned cache path: snapshot to pinned cache before freeing.
                # Skipped when source is read-only (e.g. GGUF pinned dict)
                # because the host data is always authoritative.
                cpu_buf = self._get_cpu_buf(name)
                if cpu_buf is not None:
                    cpu_buf.copy_(param.data, non_blocking=copy_non_blocking)
                elif use_incremental:
                    # First eviction of this param — allocate a pinned
                    # buffer on-the-fly.  Only the evicted layers consume
                    # host RAM; the rest stay CUDA-only.
                    pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                    pinned.copy_(param.data, non_blocking=copy_non_blocking)
                    self._set_cpu_buf(name, pinned)

            # Free CUDA storage
            s = param.data.untyped_storage()
            nb = s.nbytes()
            if nb > 0:
                self._partial_freed[name] = (s, nb)
                s.resize_(0)
                freed += nb
                evict_count += 1
                print(f"[{self._tag}]   evict [{evict_count}] {name}: "
                      f"{nb/1e6:.1f} MB (cumulative {freed/1e9:.2f}/{target_bytes/1e9:.2f} GB)")

        if freed > 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._watermark = len(self._partial_freed)
            source = "mmap" if use_mmap else (
                "incremental" if use_incremental else "pinned"
            )
            print(
                f"[{self._tag}] Watermark offload ({source}): freed {freed / 1e9:.2f} GB "
                f"({self._watermark}/{len(self._param_order)} params evicted)"
            )

        return freed

    def reload_evicted(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Restore only the params freed by ``offload_by_pressure()``.

        Equivalent to aimdo's re-fault after watermark reset via
        ``vbar.prioritize()``.

        When the pinned cache is built, copies from pinned CPU buffers.
        Falls back to the mmap state dict when pinned buffers are unavailable.
        """
        if not self._partial_freed:
            return 0

        # Re-allocate CUDA storages
        for name, (storage, nbytes) in self._partial_freed.items():
            storage.resize_(nbytes)

        # Copy source → CUDA
        param_dict = dict(module.named_parameters())
        restored = 0
        mmap_restored = 0
        for name in self._partial_freed:
            param = param_dict.get(name)
            if param is None:
                continue

            # Try pinned buffer first (fastest — DMA-capable)
            cpu_buf = self._get_cpu_buf(name)
            if cpu_buf is not None:
                param.data.copy_(cpu_buf, non_blocking=non_blocking)
                restored += 1
                continue

            # Fallback: copy from mmap source
            if self._mmap_sd is not None:
                mmap_key = self._resolve_mmap_key(name)
                if mmap_key is not None:
                    mmap_tensor = self._mmap_sd[mmap_key]
                    param.data.copy_(mmap_tensor, non_blocking=non_blocking)
                    restored += 1
                    mmap_restored += 1
                    continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        freed_gb = sum(nb for _, nb in self._partial_freed.values()) / 1e9
        evicted_names = set(self._partial_freed.keys())
        self._partial_freed.clear()
        self._watermark = 0

        # Free incremental pinned buffers — CUDA is authoritative now.
        # Only for unbuilt (incremental) path; built caches hold slab views
        # that must survive for the next offload_to_cpu snapshot.
        if not self._built:
            cpu_params = getattr(self, '_cpu_params', None)
            if cpu_params is not None:
                freed_bufs = 0
                for name in evicted_names:
                    if cpu_params.pop(name, None) is not None:
                        freed_bufs += 1
                if freed_bufs > 0:
                    print(
                        f"[{self._tag}] Freed {freed_bufs} incremental pinned "
                        f"buffers after reload."
                    )

        source_info = ""
        if mmap_restored > 0:
            source_info += f", {mmap_restored} from mmap"
        print(
            f"[{self._tag}] Restored {restored} evicted params to CUDA "
            f"({freed_gb:.2f} GB){source_info}."
        )
        return restored

    # ------------------------------------------------------------ cleanup

    def cleanup(self) -> None:
        """Release all pinned RAM held by this cache.

        Subclasses override to free format-specific resources (e.g. /dev/shm)
        and must call ``super().cleanup()``.
        """
        self._freed_storages = []
        self._partial_freed.clear()
        self._watermark = 0
        self._param_order = []
        self._built = False
        self._on_cuda = False
        self._mmap_sd = None
        self._mmap_key_map = None

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
# DictPinnedCache — external pre-pinned dict (GGUF mmap_cache, etc.)
# ═══════════════════════════════════════════════════════════════════════════

class DictPinnedCache(BasePinnedCache):
    """Pinned cache backed by an externally-pinned dict of tensors.

    Used for model formats (e.g. GGUF) that build their own pinned host
    dict (``mmap_cache``) outside the ``BasePinnedCache`` machinery.
    This class bridges that dict into the standard ``BasePinnedCache``
    interface so pressure eviction, watermark offload, and full
    offload/reload all work through a single code path.

    The external dict is the **authoritative source** — CUDA data is a
    copy, so no CUDA→CPU snapshot is needed during eviction
    (``_skip_d2h_refresh = True``).

    ``build()`` maps external dict keys → ``nn.Module`` parameter names
    and populates ``_cpu_params``.  After that, all ``BasePinnedCache``
    methods (``offload_by_pressure``, ``reload_evicted``, ``offload_to_cpu``,
    ``reload_to_cuda``, etc.) work without modification.
    """

    def __init__(
        self,
        pinned_dict: Dict[str, torch.Tensor],
        tag: str = "DictPinnedCache",
    ) -> None:
        super().__init__(tag=tag)
        self._external_dict = pinned_dict
        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}
        # External dict is authoritative — never snapshot CUDA→CPU.
        self._skip_d2h_refresh = True

    # ── build ─────────────────────────────────────────────────────────

    def _do_build(self, module: torch.nn.Module) -> None:
        """Map external dict keys → module parameter names."""
        if self._external_dict is None:
            return

        param_names = set()
        for name, _ in module.named_parameters():
            param_names.add(name)

        ext_keys = set(self._external_dict.keys())
        mapped = 0

        for ext_key in ext_keys:
            target_name = self._match_param_name(ext_key, param_names)
            if target_name is not None:
                self._cpu_params[target_name] = self._external_dict[ext_key]
                mapped += 1

        print(
            f"[{self._tag}] Built: mapped {mapped}/{len(ext_keys)} external "
            f"tensors to {len(param_names)} module params "
            f"({self.pinned_ram_bytes() / 1e9:.2f} GB)."
        )

    @staticmethod
    def _match_param_name(ext_key: str, param_names: set) -> Optional[str]:
        """Resolve an external dict key to a module parameter name.

        Handles the common prefix differences between mmap dict keys and
        ComfyUI's ``named_parameters()`` names.
        """
        # Direct match
        if ext_key in param_names:
            return ext_key
        # Try adding common prefixes
        for prefix in ('diffusion_model.', 'model.diffusion_model.'):
            candidate = prefix + ext_key
            if candidate in param_names:
                return candidate
        # Try stripping prefix from key
        for prefix in ('diffusion_model.', 'model.diffusion_model.'):
            if ext_key.startswith(prefix):
                stripped = ext_key[len(prefix):]
                if stripped in param_names:
                    return stripped
        return None

    # ── offload (no-op snapshot, CUDA storages freed by base class) ──

    def _snapshot_to_cpu(self, module, non_blocking, freed_ptrs) -> int:
        offloaded = 0
        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                continue
            # Skip D2H snapshot — external dict is authoritative.
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1
        return offloaded

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        cuda_count = cpu_count = 0
        for name, param in module.named_parameters():
            if param.device.type == "cuda":
                self._collect_storage(param.data, freed_ptrs)
                cuda_count += 1
            else:
                cpu_count += 1
        return cuda_count, cpu_count

    # ── reload (pinned external → CUDA) ──────────────────────────────

    def _copy_to_cuda(self, module, non_blocking) -> int:
        reloaded = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data.copy_(cpu_buf, non_blocking=non_blocking)
                reloaded += 1
        return reloaded

    def _swap_to_cpu(self, module) -> int:
        restored = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data = cpu_buf
                restored += 1
        return restored

    # ── info ──────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return sum(t.nbytes for t in self._cpu_params.values())

    def cleanup(self) -> None:
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_params else 0
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._external_dict = None
        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: released {gb:.2f} GB external dict references.")


# ═══════════════════════════════════════════════════════════════════════════
# ContiguousPinnedCache — single slab, zero fragmentation
# ═══════════════════════════════════════════════════════════════════════════

class ContiguousPinnedCache(BasePinnedCache):
    """Contiguous pinned-RAM cache using a single ``cudaHostAlloc`` slab.

    Instead of one ``pin_memory()`` call per parameter (hundreds of small
    ``cudaHostAlloc`` calls, ~2 KB wasted per tensor on page-rounding),
    a single contiguous buffer is allocated and individual parameter views
    are carved out via ``torch.as_strided`` on the flat storage.

    Benefits:
      - One ``cudaHostAlloc`` call → no fragmentation, no per-tensor overhead.
      - ``cudaMemcpyAsync`` can transfer the entire model in one DMA op
        (when H2D is a contiguous range).
      - ~40-50 % faster build vs per-tensor ``pin_memory()`` on large models.

    The slab is a 1-D ``uint8`` pinned tensor.  Each parameter gets an
    aligned view into this slab (512-byte aligned to match GPU DMA
    granularity, same as ``SharedPinnedParamCache``).
    """

    def __init__(self) -> None:
        super().__init__(tag="ContiguousPinnedCache")
        self._slab: Optional[torch.Tensor] = None
        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}
        self._total_bytes: int = 0

    # ── build ─────────────────────────────────────────────────────────

    def _do_build(self, module: torch.nn.Module) -> None:
        # Phase 1: compute layout — total bytes needed with alignment
        entries: List[Tuple[str, bool, int, torch.Size, torch.dtype]] = []
        offset = 0
        for name, param in module.named_parameters():
            aligned = _align_up(offset)
            nb = param.data.nbytes
            entries.append((name, True, aligned, param.data.shape, param.data.dtype))
            offset = aligned + nb
        for name, buf in module.named_buffers():
            aligned = _align_up(offset)
            nb = buf.data.nbytes
            entries.append((name, False, aligned, buf.data.shape, buf.data.dtype))
            offset = aligned + nb

        total_bytes = _align_up(offset)
        self._total_bytes = total_bytes

        # Phase 2: allocate single contiguous pinned slab
        self._slab = torch.empty(total_bytes, dtype=torch.uint8, device="cpu").pin_memory()

        # Build module map for meta→pinned swaps (param.data = view fails
        # for meta tensors; need setattr with a new Parameter on parent).
        module_map: Dict[str, torch.nn.Module] = dict(module.named_modules())

        # Phase 3: create typed views into the slab and copy data
        needs_sync = False
        param_dict = dict(module.named_parameters())
        buf_dict = dict(module.named_buffers())
        pinned_cpu = 0

        for name, is_param, byte_offset, shape, dtype in entries:
            nb = 1
            for s in shape:
                nb *= s
            nbytes = nb * torch._utils._element_size(dtype)

            # Create a typed view into the slab via uint8 slice → view(dtype)
            flat_view = self._slab[byte_offset:byte_offset + nbytes].view(dtype).reshape(shape)

            src = param_dict[name].data if is_param else buf_dict[name].data

            if src.device == torch.device("meta"):
                # Meta tensor — swap param/buffer to point at pinned view.
                # .data assignment fails for meta→CPU ("incompatible tensor
                # type"), so replace the Parameter on the parent module.
                parts = name.rsplit(".", 1)
                parent_path = parts[0] if len(parts) == 2 else ""
                attr = parts[-1]
                parent = module_map[parent_path]
                if is_param:
                    setattr(parent, attr, torch.nn.Parameter(flat_view, requires_grad=False))
                else:
                    setattr(parent, attr, flat_view)
                pinned_cpu += 1
            elif src.device.type == "cuda":
                flat_view.copy_(src, non_blocking=True)
                needs_sync = True
            else:
                flat_view.copy_(src)
                # Swap CPU param to point at pinned slab view
                if is_param:
                    param_dict[name].data = flat_view
                else:
                    buf_dict[name].data = flat_view
                pinned_cpu += 1

            if is_param:
                self._cpu_params[name] = flat_view
            else:
                self._cpu_buffers[name] = flat_view

        if needs_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        param_count = sum(1 for _, is_p, *_ in entries if is_p)
        buf_count = sum(1 for _, is_p, *_ in entries if not is_p)

        # Verify the slab is pinned
        pin_ok = self._slab.is_pinned()
        pin_status = "pinned ✓" if pin_ok else "WARNING: slab NOT pinned"

        print(
            f"[{self._tag}] Built: {param_count} params + {buf_count} buffers "
            f"in single {total_bytes / 1e9:.2f} GB slab ({pin_status}). "
            f"(pinned_in_place={pinned_cpu})"
        )

    # ── offload (CUDA → pinned) ──────────────────────────────────────

    def _snapshot_to_cpu(self, module, non_blocking, freed_ptrs) -> int:
        offloaded = 0
        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                continue
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                cpu_buf.copy_(param.data, non_blocking=non_blocking)
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                cpu_copy.copy_(buf.data, non_blocking=non_blocking)
            self._collect_storage(buf.data, freed_ptrs)

        return offloaded

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        cuda_count = cpu_count = 0
        for name, param in module.named_parameters():
            if param.device.type == "cuda":
                cpu_buf = self._cpu_params.get(name)
                if cpu_buf is not None:
                    cpu_buf.copy_(param.data, non_blocking=non_blocking)
                self._collect_storage(param.data, freed_ptrs)
                cuda_count += 1
            else:
                cpu_count += 1

        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                cpu_copy = self._cpu_buffers.get(name)
                if cpu_copy is not None:
                    cpu_copy.copy_(buf.data, non_blocking=non_blocking)
                self._collect_storage(buf.data, freed_ptrs)

        return cuda_count, cpu_count

    # ── reload (pinned → CUDA) ───────────────────────────────────────

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
        """Point param.data directly at pinned slab views."""
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

    # ── info ──────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return self._total_bytes

    def cleanup(self) -> None:
        gb = self._total_bytes / 1e9 if self._slab is not None else 0
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._slab = None
        self._total_bytes = 0
        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: freed {gb:.2f} GB contiguous pinned slab.")

    def release_host_copy(self) -> None:
        """Free pinned host memory after reload to CUDA.

        After ``reload_to_cuda()``, both CUDA and pinned copies coexist.
        This releases the pinned copy to reclaim host RAM.  The next
        ``offload_to_cpu()`` will call ``build()`` to create a fresh slab
        from live CUDA weights.

        Keeps ``_param_order`` and ``_on_cuda`` intact — the model is
        still on CUDA and future eviction walks stay valid.
        """
        gb = self._total_bytes / 1e9 if self._slab is not None else 0
        buf_gb = sum(t.nbytes for t in self._cpu_params.values()) / 1e9 if self._cpu_params else 0
        self._slab = None
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._total_bytes = 0
        self._built = False
        # _freed_storages now point to live CUDA storages — clear stale refs
        self._freed_storages = []
        total = gb + buf_gb
        if total > 0:
            print(f"[{self._tag}] Released {total:.2f} GB pinned host copy.")


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
        self._shm_metadata: Optional["HostIpcArtifactMetadata"] = None
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
        for name, param in module.named_parameters():
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
