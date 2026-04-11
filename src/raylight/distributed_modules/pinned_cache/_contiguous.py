"""ContiguousPinnedCache — single slab, zero fragmentation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ._base import BasePinnedCache
from ._helpers import _align_up


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
        self._total_bytes: int = 0

    # ── Build ─────────────────────────────────────────────────────────

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

        # Build module map for meta→pinned swaps
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

        pin_ok = self._slab.is_pinned()
        pin_status = "pinned ✓" if pin_ok else "WARNING: slab NOT pinned"

        print(
            f"[{self._tag}] Built: {param_count} params + {buf_count} buffers "
            f"in single {total_bytes / 1e9:.2f} GB slab ({pin_status}). "
            f"(pinned_in_place={pinned_cpu})"
        )

    # ── Info (override — slab-based) ──────────────────────────────────

    def pinned_ram_bytes(self) -> int:
        return self._total_bytes

    # ── Cleanup ───────────────────────────────────────────────────────

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
        self._slab = None
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._total_bytes = 0
        self._built = False
        self._freed_storages = []
        if gb > 0:
            print(f"[{self._tag}] Released {gb:.2f} GB pinned host copy.")
