from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
from raylight.config import RaylightConfig

@dataclass
class ActorConfig:
    """
    Immutable configuration for the RayActor.
    Pass this to Managers instead of the full worker instance to reduce coupling.
    """
    local_rank: int
    device_id: int
    device: torch.device
    parallel_dict: Dict[str, Any]
    global_world_size: int

    raylight_config: RaylightConfig
    
    device_mesh: Optional[Any] = None

    @property
    def is_fsdp(self) -> bool:
        return self.raylight_config.strategy.fsdp_enabled

    @property
    def fsdp_cpu_offload(self) -> bool:
        return self.raylight_config.strategy.fsdp_cpu_offload

    @property
    def is_tp(self) -> bool:
        return self.raylight_config.strategy.is_tp

    @property
    def is_xdit(self) -> bool:
        return self.raylight_config.meta.total_sp_degree > 1

    @property
    def use_mmap(self) -> bool:
        return self.raylight_config.device.use_mmap

    @property
    def zero_ram(self) -> bool:
        return self.raylight_config.device.zero_ram

    @property
    def vram_limit_gb(self) -> float:
        return self.raylight_config.device.vram_limit_gb

    @property
    def vram_limit_bytes(self) -> int:
        gb = self.vram_limit_gb
        return int(gb * (1024 ** 3)) if gb > 0 else 0

    @property
    def host_ipc_mode(self) -> str:
        return self.raylight_config.system.host_ipc_mode

    @property
    def host_ipc_path(self) -> Optional[str]:
        return self.raylight_config.system.host_ipc_path
