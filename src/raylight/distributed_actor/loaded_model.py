from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LoadedModel:
    """Single container for a loaded model and its associated metadata.

    Phase 1: patcher (the ComfyUI ModelPatcher).
    Phase 2: is_gguf, base_sd_ref, gguf_metadata — all set after model creation.
    """
    patcher: Any
    is_gguf: bool = False
    base_sd_ref: Any = None
    gguf_metadata: Dict[str, Any] = field(default_factory=dict)
