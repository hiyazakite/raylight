"""
PinnedParamCache — fast pinned-RAM offload/reload for standard (non-FSDP) models.

After the first forward pass all model parameters are on CUDA.  This cache
snapshots them to pinned CPU RAM so the model can be offloaded (VRAM freed)
and hot-reloaded with fast async H2D DMA — no mmap page faults, no disk I/O,
no per-tensor bounce-buffer copies.

Mirrors the proven ``FSDPShardPinnedCache`` design but operates on plain
``nn.Parameter`` / ``torch.Tensor`` objects rather than DTensors.

Lifecycle
---------
1. First ``offload_to_cpu(module)``
       - Calls ``build()`` automatically (lazy).
       - Snapshots every CUDA param to a pinned CPU buffer.
       - Frees VRAM via ``untyped_storage().resize_(0)`` on each unique
         CUDA storage, then ``torch.cuda.empty_cache()``.

2. Subsequent ``offload_to_cpu()`` calls
       - Refreshes pinned buffers (LoRA may have modified weights).
       - Re-frees CUDA storages.

3. ``reload_to_cuda(module)``
       - ``empty_cache()`` to reclaim residual allocations.
       - ``storage.resize_(original_nbytes)`` to re-allocate CUDA memory.
       - ``param.data.copy_(pinned_buf, non_blocking=True)`` per param.
       - One ``synchronize()`` at the end.

Why storage-level resize?
-------------------------
Each ``nn.Parameter.data`` is backed by an ``UntypedStorage``.  Some params
may share storage (e.g. weight-tying).  ``storage.resize_(0)`` truly frees
the CUDA allocation; ``resize_(nbytes)`` re-allocates it; all tensor views
that reference that storage automatically follow.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class PinnedParamCache:
    """Pinned-RAM cache for standard model parameters — fast offload/reload."""

    def __init__(self) -> None:
        self._cpu_params: Dict[str, torch.Tensor] = {}          # name → pinned CPU tensor
        self._cpu_buffers: Dict[str, torch.Tensor] = {}          # name → pinned CPU buffer
        self._meta: Dict[str, Tuple[torch.Size, torch.dtype]] = {}
        self.params_on_cuda: bool = False
        self._built: bool = False
        self._freed_storages: list = []   # [(UntypedStorage, original_nbytes)]

    # ------------------------------------------------------------------ build

    def build(self, module: torch.nn.Module) -> None:
        """Snapshot every CUDA parameter and buffer to pinned CPU RAM."""
        if self._built:
            print("[PinnedParamCache] build() called more than once — skipping.")
            return

        param_count = 0
        buf_count = 0
        skipped_cpu = 0

        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                skipped_cpu += 1
                continue
            cpu_buf = param.data.detach().to("cpu", non_blocking=False)
            try:
                cpu_buf = cpu_buf.pin_memory()
            except Exception:
                pass
            self._cpu_params[name] = cpu_buf
            self._meta[name] = (param.data.shape, param.data.dtype)
            param_count += 1

        for name, buf in module.named_buffers():
            if buf.device.type != "cuda":
                continue
            cpu_copy = buf.data.detach().to("cpu", non_blocking=False)
            try:
                cpu_copy = cpu_copy.pin_memory()
            except Exception:
                pass
            self._cpu_buffers[name] = cpu_copy
            buf_count += 1

        self._built = True
        self.params_on_cuda = True
        total_bytes = sum(t.nbytes for t in self._cpu_params.values()) + \
                      sum(t.nbytes for t in self._cpu_buffers.values())
        print(
            f"[PinnedParamCache] Built: {param_count} params + {buf_count} buffers captured, "
            f"{total_bytes / 1e9:.2f} GB pinned RAM. "
            f"(skipped_cpu={skipped_cpu})"
        )

    # --------------------------------------------------------------- offload

    def offload_to_cpu(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Move model CUDA → pinned RAM and free VRAM via storage resize."""
        if not self._built:
            print("[PinnedParamCache] First offload — building cache now...")
            self.build(module)

        if not self.params_on_cuda:
            print("[PinnedParamCache] Already on CPU — skipping offload.")
            return 0

        offloaded = 0
        freed_ptrs: set = set()
        self._freed_storages = []

        # --- Parameters ---
        for name, param in module.named_parameters():
            if param.device.type != "cuda":
                continue

            # Refresh pinned buffer
            if name in self._cpu_params:
                self._cpu_params[name].copy_(param.data, non_blocking=non_blocking)
            else:
                cpu_buf = param.data.detach().to("cpu", non_blocking=False)
                try:
                    cpu_buf = cpu_buf.pin_memory()
                except Exception:
                    pass
                self._cpu_params[name] = cpu_buf
                self._meta[name] = (param.data.shape, param.data.dtype)

            # Collect unique CUDA storage
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
            if name in self._cpu_buffers:
                self._cpu_buffers[name].copy_(buf.data, non_blocking=non_blocking)
            else:
                cpu_copy = buf.data.detach().to("cpu", non_blocking=False)
                try:
                    cpu_copy = cpu_copy.pin_memory()
                except Exception:
                    pass
                self._cpu_buffers[name] = cpu_copy

            s = buf.data.untyped_storage()
            ptr = s.data_ptr()
            if ptr != 0 and ptr not in freed_ptrs:
                freed_ptrs.add(ptr)
                self._freed_storages.append((s, s.nbytes()))

        # Sync D2H copies before freeing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Free all unique CUDA storages
        for s, _ in self._freed_storages:
            s.resize_(0)

        torch.cuda.empty_cache()

        self.params_on_cuda = False
        freed_gb = sum(nb for _, nb in self._freed_storages) / 1e9
        print(
            f"[PinnedParamCache] Offloaded {offloaded} params, freed {len(self._freed_storages)} "
            f"unique CUDA storages ({freed_gb:.2f} GB). VRAM freed."
        )
        return offloaded

    # --------------------------------------------------------------- reload

    def reload_to_cuda(
        self,
        module: torch.nn.Module,
        non_blocking: bool = True,
    ) -> int:
        """Restore model from pinned RAM → CUDA via storage resize + copy."""
        if not self._built:
            print("[PinnedParamCache] reload_to_cuda() called but not built — no-op.")
            return 0
        if self.params_on_cuda:
            print("[PinnedParamCache] Already on CUDA — skipping reload.")
            return 0

        torch.cuda.empty_cache()

        # (1) Re-allocate all freed CUDA storages
        restored = 0
        for s, nbytes in self._freed_storages:
            s.resize_(nbytes)
            restored += 1
        self._freed_storages = []
        print(f"[PinnedParamCache] Restored {restored} CUDA storages.")

        # (2) Copy param data from pinned → CUDA
        reloaded = 0
        for name, param in module.named_parameters():
            cpu_buf = self._cpu_params.get(name)
            if cpu_buf is None:
                continue
            param.data.copy_(cpu_buf, non_blocking=non_blocking)
            reloaded += 1

        # (3) Copy buffer data from pinned → CUDA
        for name, buf in module.named_buffers():
            cpu_copy = self._cpu_buffers.get(name)
            if cpu_copy is None:
                continue
            buf.data.copy_(cpu_copy, non_blocking=non_blocking)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.params_on_cuda = True
        print(f"[PinnedParamCache] Reloaded {reloaded} params to CUDA.")
        return reloaded

    # ---------------------------------------------------------------- helpers

    @property
    def built(self) -> bool:
        return self._built

    def param_count(self) -> int:
        return len(self._cpu_params)

    def pinned_ram_bytes(self) -> int:
        return sum(t.nbytes for t in self._cpu_params.values()) + \
               sum(t.nbytes for t in self._cpu_buffers.values())
