"""BasePinnedCache — lifecycle state machine with default dict-backed ops."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)


class BasePinnedCache(ABC):
    """Abstract base for all pinned-RAM offload/reload caches.

    Subclasses must implement ``_do_build``.  The four snapshot/copy
    operations have concrete defaults that iterate ``_cpu_params`` and
    ``_cpu_buffers`` dicts.  Customise behaviour via two hooks:

    ``_should_refresh(name, is_param)``
        Whether to D2H-copy this entry during snapshot.  Default: ``True``.
        ``SharedPinnedParamCache`` overrides to limit to owned params.

    ``_on_cache_miss(name, tensor)``
        Called when the CPU dict has no entry for a CUDA tensor during
        snapshot.  Return a pinned CPU copy to cache, or ``None`` to skip.
        ``PinnedParamCache`` overrides to allocate + pin on-the-fly.
    """

    def __init__(self, tag: str = "PinnedCache") -> None:
        self._tag = tag
        self._built: bool = False
        self._on_cuda: bool = False
        self._cpu_params: Dict[str, torch.Tensor] = {}
        self._cpu_buffers: Dict[str, torch.Tensor] = {}
        # Cached module walks — avoids repeated tree traversals in hot paths.
        self._param_list: List[Tuple[str, torch.nn.Parameter]] = []
        self._buffer_list: List[Tuple[str, torch.Tensor]] = []
        self._param_dict: Dict[str, torch.nn.Parameter] = {}
        # (UntypedStorage, original_nbytes) — collected during offload
        self._freed_storages: List[Tuple[torch.UntypedStorage, int]] = []

        # ── Phase 2: Signature-based skip-upload ──
        self._offload_generation: int = 0
        self._force_next_reload: bool = False

        # ── Phase 3: Watermark partial offload ──
        self._param_order: List[str] = []
        self._watermark: int = 0  # 0 = nothing evicted, len = all evicted
        self._partial_freed: Dict[str, Tuple[torch.UntypedStorage, int]] = {}

        # ── Read-only source optimisation ──
        self._skip_d2h_refresh: bool = False

        # ── Mmap fallback ──
        self._mmap_sd: Optional[Dict[str, torch.Tensor]] = None
        self._mmap_key_map: Optional[Dict[str, str]] = None

        # ── Phase 3: Execution-distance-aware eviction ──
        self._block_param_groups: Optional[List[frozenset]] = None
        self._execution_cursor: int = -1  # -1 = not in forward

    # ── Hooks (override in subclasses) ────────────────────────────────

    def _should_refresh(self, name: str, is_param: bool) -> bool:
        """Whether to D2H-copy this entry during snapshot."""
        return True

    def _on_cache_miss(
        self, name: str, tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Handle missing cache entry during snapshot.

        *tensor* is the live CUDA tensor.  Return a pinned CPU copy
        (already populated with data), or ``None`` to skip caching.
        """
        return None

    # ── Default concrete ops ──────────────────────────────────────────

    def _snapshot_to_cpu(
        self, module: torch.nn.Module, non_blocking: bool, freed_ptrs: set,
    ) -> int:
        """Refresh pinned buffers from CUDA params and collect storages."""
        offloaded = 0
        for name, param in self._param_list:
            if param.device.type != "cuda":
                continue
            if not self._skip_d2h_refresh and self._should_refresh(name, True):
                cpu_buf = self._cpu_params.get(name)
                if cpu_buf is not None:
                    cpu_buf.copy_(param.data, non_blocking=non_blocking)
                else:
                    new_buf = self._on_cache_miss(name, param.data)
                    if new_buf is not None:
                        self._cpu_params[name] = new_buf
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1

        for name, buf in self._buffer_list:
            if buf.device.type != "cuda":
                continue
            if not self._skip_d2h_refresh and self._should_refresh(name, False):
                cpu_buf = self._cpu_buffers.get(name)
                if cpu_buf is not None:
                    cpu_buf.copy_(buf.data, non_blocking=non_blocking)
                else:
                    new_buf = self._on_cache_miss(name, buf.data)
                    if new_buf is not None:
                        self._cpu_buffers[name] = new_buf
            self._collect_storage(buf.data, freed_ptrs)

        return offloaded

    def _snapshot_cuda_only(
        self, module: torch.nn.Module, non_blocking: bool, freed_ptrs: set,
    ) -> tuple:
        """Refresh pinned cache for CUDA-resident params only.

        Returns ``(cuda_count, cpu_count)``.
        """
        cuda_count = cpu_count = 0
        for name, param in self._param_list:
            if param.device.type == "cuda":
                if not self._skip_d2h_refresh and self._should_refresh(name, True):
                    cpu_buf = self._cpu_params.get(name)
                    if cpu_buf is not None:
                        cpu_buf.copy_(param.data, non_blocking=non_blocking)
                    else:
                        new_buf = self._on_cache_miss(name, param.data)
                        if new_buf is not None:
                            self._cpu_params[name] = new_buf
                self._collect_storage(param.data, freed_ptrs)
                cuda_count += 1
            else:
                cpu_count += 1

        for name, buf in self._buffer_list:
            if buf.device.type == "cuda":
                if not self._skip_d2h_refresh and self._should_refresh(name, False):
                    cpu_buf = self._cpu_buffers.get(name)
                    if cpu_buf is not None:
                        cpu_buf.copy_(buf.data, non_blocking=non_blocking)
                    else:
                        new_buf = self._on_cache_miss(name, buf.data)
                        if new_buf is not None:
                            self._cpu_buffers[name] = new_buf
                self._collect_storage(buf.data, freed_ptrs)

        return cuda_count, cpu_count

    def _copy_to_cuda(
        self, module: torch.nn.Module, non_blocking: bool,
    ) -> int:
        """Copy pinned data → CUDA params.  Return count."""
        reloaded = 0
        for name, param in self._param_list:
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data.copy_(cpu_buf, non_blocking=non_blocking)
                reloaded += 1

        for name, buf in self._buffer_list:
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                buf.data.copy_(cpu_copy, non_blocking=non_blocking)

        return reloaded

    def _swap_to_cpu(self, module: torch.nn.Module) -> int:
        """Replace param.data with pinned CPU tensor.  Return count."""
        restored = 0
        for name, param in self._param_list:
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                param.data = cpu_buf
                restored += 1

        for name, buf in self._buffer_list:
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                buf.data = cpu_copy

        return restored

    # ── Public interface ──────────────────────────────────────────────

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
        # Cache flat param/buffer lists — avoids tree walks in hot paths.
        self._param_list = list(module.named_parameters())
        self._buffer_list = list(module.named_buffers())
        self._param_dict = dict(self._param_list)
        # Record load-order for watermark partial offload (Phase 3).
        self._param_order = [name for name, _ in self._param_list]
        # Detect actual state: if ANY param is on CUDA, treat as on-CUDA.
        self._on_cuda = any(
            p.device.type == "cuda" for _, p in self._param_list
        )

    def set_mmap_fallback(self, mmap_sd: Dict[str, torch.Tensor]) -> None:
        """Register an mmap-backed state dict for zero-alloc pressure eviction."""
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
                if name in mmap_keys:
                    self._mmap_key_map[name] = name
                    continue
                found = False
                for prefix in ('diffusion_model.', 'model.diffusion_model.'):
                    candidate = prefix + name
                    if candidate in mmap_keys:
                        self._mmap_key_map[name] = candidate
                        found = True
                        break
                if not found:
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

    # ── Offload / reload ──────────────────────────────────────────────

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._pre_free_hook()

        self._offload_generation += 1
        for s, _ in self._freed_storages:
            s.resize_(0)

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._pre_free_hook()

        self._offload_generation += 1
        for s, _ in self._freed_storages:
            s.resize_(0)

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

        self._partial_freed.clear()
        self._watermark = 0
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

        restored = 0
        alloc_bytes = 0

        if self._partial_freed:
            for name, (s, nbytes) in self._partial_freed.items():
                self._freed_storages.append((s, nbytes))
            self._partial_freed.clear()
            self._watermark = 0

        # ── Cold start: cache built from CPU params, no prior offload ──
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

        Returns ``(reloaded_count, alloc_bytes)``.
        """
        reloaded = 0
        alloc_bytes = 0
        device = None

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        if device is None:
            print(f"[{self._tag}] Cold start: no CUDA device available.")
            return 0, 0

        for name, param in self._param_list:
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is not None:
                cuda_tensor = torch.empty_like(cpu_buf, device=device)
                cuda_tensor.copy_(cpu_buf, non_blocking=non_blocking)
                param.data = cuda_tensor
                alloc_bytes += cuda_tensor.nbytes
                reloaded += 1

        for name, buf in self._buffer_list:
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is not None:
                cuda_tensor = torch.empty_like(cpu_copy, device=device)
                cuda_tensor.copy_(cpu_copy, non_blocking=non_blocking)
                buf.data = cuda_tensor
                alloc_bytes += cuda_tensor.nbytes

        print(
            f"[{self._tag}] Cold start: allocated {alloc_bytes / 1e9:.2f} GB "
            f"of fresh CUDA tensors for {reloaded} params."
        )
        return reloaded, alloc_bytes

    # ── Abstract / overridable ────────────────────────────────────────

    @abstractmethod
    def _do_build(self, module: torch.nn.Module) -> None:
        """Subclass: snapshot CUDA data to pinned host RAM."""

    def _pre_free_hook(self) -> None:
        """Optional hook called after sync but before storage.resize_(0).

        Used by SharedPinnedParamCache for dist.barrier().
        """

    # ── Default info ──────────────────────────────────────────────────

    def param_count(self) -> int:
        """Number of cached parameters.  Override if storage differs."""
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        """Total pinned RAM in bytes.  Override for slab-based caches."""
        return (sum(t.nbytes for t in self._cpu_params.values())
                + sum(t.nbytes for t in self._cpu_buffers.values()))

    # ── Helpers ───────────────────────────────────────────────────────

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

    # ── Invalidation ──────────────────────────────────────────────────

    def invalidate(self) -> None:
        """Mark CUDA data as dirty — next reload always does a full H2D copy.

        Call after in-place weight modification (e.g. LoRA baking) that
        makes the pinned cache stale relative to CUDA.
        """
        self._force_next_reload = True

    # ── Watermark partial offload ─────────────────────────────────────

    def _get_cpu_buf(self, name: str) -> Optional[torch.Tensor]:
        """Return the pinned CPU tensor for *name*, or None."""
        return self._cpu_params.get(name)

    def _set_cpu_buf(self, name: str, tensor: torch.Tensor) -> None:
        """Store a pinned CPU tensor for *name*."""
        self._cpu_params[name] = tensor

    def offload_by_pressure(
        self,
        module: torch.nn.Module,
        target_bytes: int,
        non_blocking: bool = True,
        exclude: Optional[set] = None,
    ) -> int:
        """Free at least *target_bytes* of VRAM by offloading tail params.

        Walks parameters in **reverse load-order** (deepest layers first).
        Stops as soon as cumulative freed bytes >= target_bytes.

        Three source modes, chosen automatically:

        - **built** — copies CUDA → existing pinned slab views before freeing.
        - **mmap** — frees CUDA storage; data recoverable from mmap source.
        - **incremental** — allocates a pinned buffer per evicted param
          on-the-fly, copies CUDA → pinned, then frees.

        Returns actual bytes freed.
        """
        if target_bytes <= 0:
            return 0

        copy_non_blocking = False

        use_mmap = False
        use_incremental = False
        if not self._built:
            if self._mmap_sd is not None:
                use_mmap = True
            else:
                use_incremental = True

        # Lazy-init cached lists if not yet populated (mmap path, no build)
        if not self._param_list:
            self._param_list = list(module.named_parameters())
            self._buffer_list = list(module.named_buffers())
            self._param_dict = dict(self._param_list)
            self._param_order = [name for name, _ in self._param_list]
            self._mmap_key_map = None  # force rebuild

        freed = 0

        # ── Build eviction walk order ─────────────────────────────────
        evict_order: Optional[List[str]] = None
        groups = self._block_param_groups
        cursor = self._execution_cursor
        if groups is not None and 0 <= cursor < len(groups):
            evict_order = []
            for bi in range(cursor - 1, -1, -1):
                evict_order.extend(groups[bi])
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
                continue
            param = self._param_dict.get(name)
            if param is None or param.device.type != "cuda":
                continue

            if use_mmap:
                mmap_key = self._resolve_mmap_key(name)
                if mmap_key is None:
                    continue
            elif not self._skip_d2h_refresh:
                cpu_buf = self._get_cpu_buf(name)
                if cpu_buf is not None:
                    cpu_buf.copy_(param.data, non_blocking=copy_non_blocking)
                elif use_incremental:
                    pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                    pinned.copy_(param.data, non_blocking=copy_non_blocking)
                    self._set_cpu_buf(name, pinned)

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

        When the pinned cache is built, copies from pinned CPU buffers.
        Falls back to the mmap state dict when pinned buffers are unavailable.
        """
        if not self._partial_freed:
            return 0

        for name, (storage, nbytes) in self._partial_freed.items():
            storage.resize_(nbytes)

        restored = 0
        mmap_restored = 0
        for name in self._partial_freed:
            param = self._param_dict.get(name)
            if param is None:
                continue

            cpu_buf = self._get_cpu_buf(name)
            if cpu_buf is not None:
                param.data.copy_(cpu_buf, non_blocking=non_blocking)
                restored += 1
                continue

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
        if not self._built:
            freed_bufs = 0
            for name in evicted_names:
                if self._cpu_params.pop(name, None) is not None:
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

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release all pinned RAM held by this cache.

        Subclasses override to free format-specific resources (e.g. /dev/shm)
        and must call ``super().cleanup()``.
        """
        self._freed_storages = []
        self._partial_freed.clear()
        self._watermark = 0
        self._param_list = []
        self._buffer_list = []
        self._param_dict = {}
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
