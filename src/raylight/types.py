"""Protocol types for Raylight duck-typed interfaces.

Centralises the structural contracts that the distributed worker layer
expects so that static type checkers (mypy / pyright) and readers can
understand the API surface without chasing through dozens of ``getattr``
calls.
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch


# ---------------------------------------------------------------------------
# ModelPatcher-like
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelPatcherLike(Protocol):
    """Structural contract satisfied by ComfyUI's ``ModelPatcher`` and
    Raylight's ``RaylightModelPatcher`` / ``FSDPModelPatcher``.
    """

    model: Any
    load_device: torch.device
    offload_device: torch.device
    current_device: torch.device
    patches: Dict[str, Any]
    patches_uuid: Any

    # -- lifecycle --
    def load(self, device: torch.device, **kwargs: Any) -> None: ...
    def unpatch_model(self, device_to: Optional[torch.device] = None,
                      unpatch_weights: bool = False) -> None: ...
    def model_size(self) -> int: ...

    # -- hooks / callbacks --
    def get_all_callbacks(self, key: Any) -> List[Callable[..., Any]]: ...
    def apply_hooks(self, hooks: Any, *, force_apply: bool = False) -> None: ...
    def unpatch_hooks(self) -> None: ...
    def inject_model(self) -> None: ...

    # -- context managers --
    def use_ejected(self) -> Any: ...

    # -- weight patching --
    def patch_weight_to_device(self, key: str, device_to: Optional[torch.device] = None) -> None: ...


# ---------------------------------------------------------------------------
# LoRA manager
# ---------------------------------------------------------------------------

@runtime_checkable
class LoraManagerLike(Protocol):
    """Structural contract for the LoRA lifecycle manager."""

    def clear_tracking(self) -> None: ...
    def clear_gpu_refs(self, model: Any, config: Any) -> None: ...


# ---------------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------------

@runtime_checkable
class StateCacheLike(Protocol):
    """Structural contract for the LRU state-dict cache."""

    def get(self, key: str) -> Any: ...
    def put(self, key: str, value: Any) -> None: ...
    def __contains__(self, key: str) -> bool: ...


# ---------------------------------------------------------------------------
# Worker config
# ---------------------------------------------------------------------------

@runtime_checkable
class WorkerConfigLike(Protocol):
    """Minimal shape expected from ``WorkerConfig``."""

    local_rank: int
    device: torch.device
    global_world_size: int
    device_mesh: Optional[Any]
    parallel_dict: Dict[str, Any]

    @property
    def is_fsdp(self) -> bool: ...

    @property
    def fsdp_cpu_offload(self) -> bool: ...

    @property
    def is_xdit(self) -> bool: ...

    @property
    def use_mmap(self) -> bool: ...

    @property
    def vram_limit_bytes(self) -> int: ...
