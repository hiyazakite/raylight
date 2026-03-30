"""model_context package — public API re-exports.

All existing ``from raylight.distributed_actor.model_context import X``
statements continue to work unchanged.
"""
from ._base import (                  # noqa: F401
    ModelContext,
    ModelState,
    _compute_vram_budget,
    _ops_for_model_options,
)
from .gguf import GGUFContext          # noqa: F401
from .lazy_tensor import LazyTensorContext  # noqa: F401
from .fsdp import FSDPContext          # noqa: F401
from .vae import VAEContext            # noqa: F401
from .bnb import BNBContext            # noqa: F401


def get_context(
    path: str,
    config,
    model_type: str = "unet",
    model_options=None,
) -> ModelContext:
    """Factory function to select appropriate context based on config and file type.

    Args:
        path: Path to model file
        config: Worker configuration
        model_type: "unet" or "vae"
        model_options: Optional model load options (needed for BNB detection)

    Returns:
        Appropriate ModelContext subclass instance
    """
    use_mmap = config.use_mmap

    if model_type == "vae":
        return VAEContext(use_mmap=use_mmap)

    if model_options and ("bnb_4bit" in model_options or "load_in_4bit" in model_options):
        return BNBContext(use_mmap=False)

    if getattr(config, "is_fsdp", False):
        if path.lower().endswith(".gguf"):
            raise ValueError(
                "[Raylight] FSDP is not supported for GGUF models. "
                "GGUF quantization is incompatible with FSDP sharding. "
                "Please use a standard Safetensors model or disable FSDP/Context Parallel to use GGUF."
            )
        return FSDPContext(use_mmap=use_mmap)

    if path.lower().endswith(".gguf"):
        return GGUFContext(use_mmap=use_mmap)

    return LazyTensorContext(use_mmap=use_mmap)
