"""BitsAndBytes (4-bit) model context."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from ._base import ModelContext, ModelState

if TYPE_CHECKING:
    from raylight.types import LoraManagerLike, ActorConfigLike


class BNBContext(ModelContext):
    """Context for BitsAndBytes (4-bit) model loading."""

    def load_state_dict_standard(self, state: ModelState, config: "ActorConfigLike") -> Any:
        return {"__bnb_internal__": True}

    def load_state_dict_mmap(self, state: ModelState, config: "ActorConfigLike") -> Any:
        return self.load_state_dict_standard(state, config)

    def instantiate_model(self, sd: Dict, state: ModelState, config: "ActorConfigLike",
                          metadata: Any = None, **kwargs) -> Any:
        unet_path = state.unet_path

        if config.is_fsdp:
            import comfy.model_patcher as model_patcher
            import comfy.model_management as model_management
            from raylight.comfy_dist.model_management import cleanup_models_gc
            from raylight.comfy_dist.model_patcher import LowVramPatch

            model_patcher.LowVramPatch = LowVramPatch
            model_management.cleanup_models_gc = cleanup_models_gc

            from raylight.comfy_dist.sd import fsdp_bnb_load_diffusion_model

            model, state_dict = fsdp_bnb_load_diffusion_model(
                unet_path,
                config.local_rank,
                config.device_mesh,
                config.fsdp_cpu_offload,
            )
            model.fsdp_state_dict = state_dict
            return model
        else:
            from raylight.comfy_dist.sd import bnb_load_diffusion_model
            return bnb_load_diffusion_model(unet_path)

    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any]) -> None:
        # BNB models are difficult to hot-load due to 4-bit/CPU offload config
        pass
