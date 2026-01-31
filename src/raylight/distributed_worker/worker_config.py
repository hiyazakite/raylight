from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class WorkerConfig:
    """
    Immutable configuration for the RayWorker.
    Pass this to Managers instead of the full worker instance to reduce coupling.
    """
    local_rank: int
    device_id: int
    device: torch.device
    parallel_dict: Dict[str, Any]
    global_world_size: int
    device_mesh: Optional[Any] = None

    @property
    def is_fsdp(self) -> bool:
        return self.parallel_dict.get("is_fsdp", False)

    @property
    def is_xdit(self) -> bool:
        return self.parallel_dict.get("is_xdit", False)
