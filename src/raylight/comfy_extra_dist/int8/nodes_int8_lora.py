"""
Raylight INT8 LoRA loaders.

Provides:
- RayINT8LoraLoader        – single LoRA with stochastic INT8 patching
- RayINT8LoraLoaderStack    – apply up to 10 LoRAs merged before rounding
- RayINT8DynamicLoraLoader  – dynamic (non-sticky) LoRA via forward hook
- RayINT8DynamicLoraStack   – combine up to 10 dynamic LoRAs
"""

import logging
import torch
import folder_paths
import comfy.utils
import comfy.lora
from .int8_quant import _resolve_module, _layer_name_from_key

logger = logging.getLogger(__name__)


# ============================================================================
# 1.  Stochastic LoRA Loader (single)
# ============================================================================

class RayINT8LoraLoader:
    """
    INT8-aware LoRA loader.  Quantized layers are patched in INT8 space via
    stochastic rounding; float layers fall back to standard patching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "Raylight/INT8"
    DESCRIPTION = (
        "Load a LoRA for INT8-quantized models.  Uses high-precision "
        "stochastic rounding in INT8 space to preserve quality."
    )

    def load_lora(self, model, lora_name, strength, seed=318008):
        if strength == 0:
            return (model,)

        from .int8_quant import INT8LoRAPatchAdapter

        if INT8LoRAPatchAdapter is None:
            raise RuntimeError(
                "INT8 LoRA patching unavailable – update ComfyUI to a version "
                "that exports comfy.weight_adapter.lora.LoRAAdapter."
            )

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_patcher = model.clone()

        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)

        final_patch_dict = {}
        applied_count = 0

        for key, adapter in patch_dict.items():
            layer_name = _layer_name_from_key(key)
            try:
                target_module = _resolve_module(model_patcher, layer_name)
                if hasattr(target_module, "_is_quantized") and target_module._is_quantized:
                    w_scale = target_module.weight_scale
                    if isinstance(w_scale, torch.Tensor):
                        w_scale = w_scale.item() if w_scale.numel() == 1 else w_scale
                    new_adapter = INT8LoRAPatchAdapter(
                        adapter.loaded_keys, adapter.weights, w_scale, seed=seed,
                    )
                    final_patch_dict[key] = new_adapter
                    applied_count += 1
                else:
                    final_patch_dict[key] = adapter
            except (AttributeError, KeyError, IndexError, TypeError):
                final_patch_dict[key] = adapter

        model_patcher.add_patches(final_patch_dict, strength)

        logger.info(
            "[Raylight INT8 LoRA] '%s' strength=%.2f  – %d quantized / %d total layers",
            lora_name, strength, applied_count, len(patch_dict),
        )
        return (model_patcher,)


# ============================================================================
# 2.  Stochastic LoRA Stack (merged rounding)
# ============================================================================

class RayINT8LoraLoaderStack:
    """Apply up to 10 LoRAs with a single merged stochastic rounding step."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs: dict = {"required": {"model": ("MODEL",)}, "optional": {}}
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "Raylight/INT8"
    DESCRIPTION = (
        "Stack up to 10 LoRAs with a single merged stochastic rounding pass "
        "for INT8 models.  More precise than applying LoRAs sequentially."
    )

    def apply_stack(self, model, seed=318008, **kwargs):
        from .int8_quant import INT8MergedLoRAPatchAdapter

        if INT8MergedLoRAPatchAdapter is None:
            raise RuntimeError("INT8 merged LoRA patching unavailable.")

        all_loras = []
        for i in range(1, 11):
            name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            if name and name != "None" and strength != 0:
                path = folder_paths.get_full_path("loras", name)
                data = comfy.utils.load_torch_file(path, safe_load=True)
                all_loras.append((data, strength, name))

        if not all_loras:
            return (model,)

        model_patcher = model.clone()

        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        layered_patches: dict = {}
        for data, strength, _name in all_loras:
            patch_dict = comfy.lora.load_lora(data, key_map, log_missing=True)
            for key, adapter in patch_dict.items():
                layered_patches.setdefault(key, []).append((adapter, strength))

        final_patch_dict = {}
        applied_count = 0

        for key, patches in layered_patches.items():
            layer_name = _layer_name_from_key(key)
            try:
                target_module = _resolve_module(model_patcher, layer_name)
                w_scale = 1.0
                is_quantized = (
                    hasattr(target_module, "_is_quantized") and target_module._is_quantized
                )
                if is_quantized:
                    w_scale = target_module.weight_scale
                    if isinstance(w_scale, torch.Tensor):
                        w_scale = w_scale.item() if w_scale.numel() == 1 else w_scale
                    applied_count += 1

                final_patch_dict[key] = INT8MergedLoRAPatchAdapter(
                    patches, w_scale, seed=seed,
                )
            except Exception:
                for adapter, strength in patches:
                    model_patcher.add_patches({key: adapter}, strength)

        model_patcher.add_patches(final_patch_dict, 1.0)

        logger.info(
            "[Raylight INT8 LoRA Stack] %d LoRAs, %d quantized layers merged",
            len(all_loras), applied_count,
        )
        return (model_patcher,)


# ============================================================================
# 3.  Dynamic LoRA Loader (non-sticky, via forward hook)
# ============================================================================

class RayINT8DynamicLoraLoader:
    """
    Dynamic INT8 LoRA loader.  LoRA patches are applied at inference time
    via a forward hook – they do not modify the stored weights and are
    automatically removed when the model patcher context changes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "Raylight/INT8"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)

        from .int8_quant import DynamicLoRAHook

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_patcher = model.clone()

        key_map = {}
        if model_patcher.model.model_type.name != "ModelType.CLIP":
            key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)

        patch_dict = comfy.lora.load_lora(lora, key_map, log_missing=True)

        DynamicLoRAHook.register(model_patcher.model.diffusion_model)

        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}

        opts = model_patcher.model_options["transformer_options"]
        if "dynamic_loras" not in opts:
            opts["dynamic_loras"] = []
        else:
            opts["dynamic_loras"] = opts["dynamic_loras"].copy()

        opts["dynamic_loras"].append({
            "name": lora_name,
            "strength": strength,
            "patches": patch_dict,
        })

        return (model_patcher,)


# ============================================================================
# 4.  Dynamic LoRA Stack
# ============================================================================

class RayINT8DynamicLoraStack:
    """Convenience node: apply up to 10 dynamic LoRAs in one go."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs: dict = {"required": {"model": ("MODEL",)}, "optional": {}}
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        for i in range(1, 11):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = (
                "FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
            )
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "Raylight/INT8"

    def apply_stack(self, model, **kwargs):
        loader = RayINT8DynamicLoraLoader()
        current_model = model
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}")
            strength = kwargs.get(f"strength_{i}", 0)
            if lora_name and lora_name != "None" and strength != 0:
                (current_model,) = loader.load_lora(current_model, lora_name, strength)
        return (current_model,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "RayINT8LoraLoader": RayINT8LoraLoader,
    "RayINT8LoraLoaderStack": RayINT8LoraLoaderStack,
    "RayINT8DynamicLoraLoader": RayINT8DynamicLoraLoader,
    "RayINT8DynamicLoraStack": RayINT8DynamicLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayINT8LoraLoader": "Load LoRA INT8 (Raylight)",
    "RayINT8LoraLoaderStack": "INT8 LoRA Stack (Raylight)",
    "RayINT8DynamicLoraLoader": "Load LoRA INT8 Dynamic (Raylight)",
    "RayINT8DynamicLoraStack": "INT8 LoRA Stack Dynamic (Raylight)",
}
