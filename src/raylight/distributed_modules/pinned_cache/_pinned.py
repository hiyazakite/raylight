"""PinnedParamCache — single-GPU, private per-tensor pinned RAM."""

from __future__ import annotations

from typing import Optional

import torch

from ._base import BasePinnedCache


class PinnedParamCache(BasePinnedCache):
    """Private pinned-RAM cache for standard (non-FSDP) single-GPU models."""

    def __init__(self) -> None:
        super().__init__(tag="PinnedParamCache")

    # ── Hook: allocate + pin on cache miss ────────────────────────────

    def _on_cache_miss(
        self, name: str, tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pinned = torch.empty_like(tensor, device="cpu").pin_memory()
        pinned.copy_(tensor, non_blocking=True)
        return pinned

    # ── Build ─────────────────────────────────────────────────────────

    def _do_build(self, module: torch.nn.Module) -> None:
        param_count = buf_count = pinned_cpu = not_pinned = 0
        needs_sync = False

        for name, param in module.named_parameters():
            if param.device.type == "cuda":
                pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                pinned.copy_(param.data, non_blocking=True)
                self._cpu_params[name] = pinned
                needs_sync = True
            else:
                pinned = torch.empty_like(param.data, device="cpu").pin_memory()
                pinned.copy_(param.data)
                self._cpu_params[name] = pinned
                param.data = pinned  # swap — original freed by RC
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

        if needs_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Verify pinning
        for t in self._cpu_params.values():
            if not t.is_pinned():
                not_pinned += 1
        for t in self._cpu_buffers.values():
            if not t.is_pinned():
                not_pinned += 1

        pin_status = (
            "all pinned ✓" if not_pinned == 0
            else f"WARNING: {not_pinned} tensors NOT pinned "
                 "(DMA will use slow staging path)"
        )
        print(
            f"[{self._tag}] Built: {param_count} params + {buf_count} buffers, "
            f"{self.pinned_ram_bytes() / 1e9:.2f} GB pinned. "
            f"(pinned_in_place={pinned_cpu}, {pin_status})"
        )

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_params else 0
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: freed {gb:.2f} GB pinned RAM.")
