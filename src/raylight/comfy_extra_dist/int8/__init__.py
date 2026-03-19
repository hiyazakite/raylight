"""
Raylight INT8 – Fast INT8 tensorwise quantization for ComfyUI.

Registers the Int8Tensorwise layout with ComfyUI's quant system and exposes
the diffusion-model loader, LoRA loader, and dynamic-LoRA nodes.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# =============================================================================
# 1.  Layout Registration (runs at import time)
# =============================================================================

def _register_layouts():
    """Register Int8Tensorwise layout + algo with ComfyUI's quant registry."""
    try:
        from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout

        class Int8TensorwiseLayout(QuantizedLayout):
            class Params:
                def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kw):
                    self.scale = scale
                    self.orig_dtype = orig_dtype
                    self.orig_shape = orig_shape

                def clone(self):
                    return Int8TensorwiseLayout.Params(
                        scale=(
                            self.scale.clone()
                            if isinstance(self.scale, torch.Tensor)
                            else self.scale
                        ),
                        orig_dtype=self.orig_dtype,
                        orig_shape=self.orig_shape,
                    )

            @classmethod
            def state_dict_tensors(cls, qdata, params):
                return {"": qdata, "weight_scale": params.scale}

            @classmethod
            def dequantize(cls, qdata, params):
                return qdata.float() * params.scale

        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

        QUANT_ALGOS.setdefault(
            "int8_tensorwise",
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "Int8TensorwiseLayout",
            },
        )
        logger.debug("[Raylight INT8] Layout registered successfully")

    except ImportError:
        logger.warning(
            "[Raylight INT8] ComfyUI quant system not found – "
            "pre-quantized checkpoint loading may not work (update ComfyUI?)."
        )
    except Exception as e:
        logger.error("[Raylight INT8] Failed to register layouts: %s", e)


_register_layouts()

# =============================================================================
# 2.  Public API: Int8TensorwiseOps (for external / programmatic use)
# =============================================================================

try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

# =============================================================================
# 3.  Node Mappings
# =============================================================================

try:
    from .nodes_int8_loader import (
        NODE_CLASS_MAPPINGS as _LOADER_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _LOADER_NAMES,
    )
    from .nodes_int8_lora import (
        NODE_CLASS_MAPPINGS as _LORA_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _LORA_NAMES,
    )

    NODE_CLASS_MAPPINGS = {**_LOADER_CLASSES, **_LORA_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**_LOADER_NAMES, **_LORA_NAMES}

except ImportError as e:
    logger.error("[Raylight INT8] Failed to import nodes: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "Int8TensorwiseOps",
    "_register_layouts",
]
