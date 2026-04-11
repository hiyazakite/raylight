"""DictPinnedCache — external pre-pinned dict (GGUF mmap_cache, etc.)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ._base import BasePinnedCache


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
    """

    def __init__(
        self,
        pinned_dict: Dict[str, torch.Tensor],
        tag: str = "DictPinnedCache",
    ) -> None:
        super().__init__(tag=tag)
        self._external_dict = pinned_dict
        # External dict is authoritative — never snapshot CUDA→CPU.
        self._skip_d2h_refresh = True

    # ── Build ─────────────────────────────────────────────────────────

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
        """Resolve an external dict key to a module parameter name."""
        if ext_key in param_names:
            return ext_key
        for prefix in ('diffusion_model.', 'model.diffusion_model.'):
            candidate = prefix + ext_key
            if candidate in param_names:
                return candidate
        for prefix in ('diffusion_model.', 'model.diffusion_model.'):
            if ext_key.startswith(prefix):
                stripped = ext_key[len(prefix):]
                if stripped in param_names:
                    return stripped
        return None

    # ── Snapshot overrides (params only — no buffer handling) ─────────

    def _snapshot_to_cpu(self, module, non_blocking, freed_ptrs) -> int:
        offloaded = 0
        for name, param in self._param_list:
            if param.device.type != "cuda":
                continue
            # Skip D2H snapshot — external dict is authoritative.
            self._collect_storage(param.data, freed_ptrs)
            offloaded += 1
        return offloaded

    def _snapshot_cuda_only(self, module, non_blocking, freed_ptrs) -> tuple:
        cuda_count = cpu_count = 0
        for name, param in self._param_list:
            if param.device.type == "cuda":
                self._collect_storage(param.data, freed_ptrs)
                cuda_count += 1
            else:
                cpu_count += 1
        return cuda_count, cpu_count

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        gb = self.pinned_ram_bytes() / 1e9 if self._cpu_params else 0
        self._cpu_params.clear()
        self._cpu_buffers.clear()
        self._external_dict = None
        super().cleanup()
        if gb > 0:
            print(f"[{self._tag}] Cleanup: released {gb:.2f} GB external dict references.")
