"""Registry for model-specific Tensor Parallelism patching functions.

Mirrors ``FSDPShardRegistry``: each supported model class registers an
``apply_tp`` handler that monkey-patches the model's ``nn.Linear`` layers
into ``TPLinear`` (and norms into ``TPRMSNormAcrossHeads`` where needed).

Called from ``_init_tensor_parallel()`` in ``actor.py`` after
``TensorParallelState.initialize()``.
"""

from typing import Optional

import torch.distributed as dist
from comfy import model_base


class TPRegistry:
    """Dispatch registry for model-specific TP patching.

    Register the **most specific (child)** model classes FIRST, followed by
    parent classes, so that ``isinstance()`` dispatch is correct.

    Callers may pass either a raw ``model_base.*`` instance or a
    ``ModelPatcher`` wrapper.  The registry automatically unwraps patchers
    via ``_unwrap()`` before doing ``isinstance`` checks.

    Example::

        @TPRegistry.register(model_base.LTXAV)
        def _tp_ltxav(comfy_model, tp_group=None):
            from ..diffusion_models.lightricks.tp import apply_tp_to_ltxav_model
            inner = comfy_model.diffusion_model
            apply_tp_to_ltxav_model(inner, tp_group=tp_group)
    """

    _REGISTRY: dict = {}

    @staticmethod
    def _unwrap(obj):
        """Unwrap a ModelPatcher to the inner ComfyUI model if needed."""
        inner = getattr(obj, "model", None)
        return inner if inner is not None else obj

    @classmethod
    def register(cls, model_class):
        """Decorator that binds *model_class* to a TP-patching function."""
        def decorator(tp_func):
            cls._REGISTRY[model_class] = tp_func
            return tp_func
        return decorator

    @classmethod
    def apply(cls, comfy_model, tp_group: Optional[dist.ProcessGroup] = None):
        """Find and call the right TP patcher based on model type."""
        inner = cls._unwrap(comfy_model)
        for registered_cls, tp_func in cls._REGISTRY.items():
            if isinstance(inner, registered_cls):
                print(f"[TPRegistry] Applying TP to {registered_cls.__name__}")
                tp_func(inner, tp_group=tp_group)
                return
        raise ValueError(f"No TP handler found for model type: {type(inner)}")

    @classmethod
    def has_handler(cls, comfy_model) -> bool:
        """Return True if a TP handler is registered for this model."""
        inner = cls._unwrap(comfy_model)
        return any(isinstance(inner, c) for c in cls._REGISTRY)


# ---------------------------------------------------------------------------
# Per-model handler stubs (no-op until Phase 2.1+ implementations exist)
# ---------------------------------------------------------------------------
# Handlers are registered with hasattr guards so that old ComfyUI versions
# that lack a model class don't break import.

# Phase 2.1 — Flux (gather_output=True, weight-only TP)
if hasattr(model_base, "Flux"):
    @TPRegistry.register(model_base.Flux)
    def _tp_flux(comfy_model, tp_group=None):
        from ..diffusion_models.flux.tp_context_parallel import apply_tp_to_flux_model
        apply_tp_to_flux_model(comfy_model.diffusion_model, tp_group=tp_group)

# Phase 2.2 — Wan (true head-sharded TP)
if hasattr(model_base, "WAN21"):
    @TPRegistry.register(model_base.WAN21)
    def _tp_wan21(comfy_model, tp_group=None):
        raise NotImplementedError(
            "Wan TP not yet implemented (Phase 2.2). "
            "Set tensor_parallel_degree=1 for Wan models."
        )

if hasattr(model_base, "WAN22"):
    @TPRegistry.register(model_base.WAN22)
    def _tp_wan22(comfy_model, tp_group=None):
        raise NotImplementedError(
            "Wan TP not yet implemented (Phase 2.2). "
            "Set tensor_parallel_degree=1 for Wan models."
        )

# Phase 2.3 — LTXAV (true head-sharded TP + distributed QK-norm + RoPE slicing)
if hasattr(model_base, "LTXAV"):
    @TPRegistry.register(model_base.LTXAV)
    def _tp_ltxav(comfy_model, tp_group=None):
        raise NotImplementedError(
            "LTXAV TP not yet implemented (Phase 2.3). "
            "Set tensor_parallel_degree=1 for LTXAV models."
        )
