"""
Raylight INT8 W8A8 Diffusion Model Loader.

Loads diffusion models with INT8 tensorwise quantization for fast inference
via torch._int_mm / Triton fused kernels on Ampere+ GPUs.

Supported input formats:
- Pre-quantized INT8 safetensors  (loaded directly, fastest)
- FP16 / BF16 / FP32 safetensors  (on-the-fly INT8 quantization)
- FP8 (e4m3fn / e5m2) safetensors (cast to BF16 then quantized to INT8)
"""

import torch
import folder_paths
import comfy.sd

from .int8_quant import Int8TensorwiseOps

# Per-architecture layer exclusion lists.
# These layers either are too small, are embeddings/projections that need
# higher precision, or show measurable quality loss when quantized to INT8.
_MODEL_EXCLUSIONS: dict[str, list[str]] = {
    "flux": [
        "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation",
    ],
    "z-image": [
        "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
        "context_refiner", "final_layer", "noise_refiner", "adaLN",
        "x_pad_token",
    ],
    "chroma": [
        "distilled_guidance_layer", "final_layer", "img_in", "txt_in",
        "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv",
        "__x0__", "nerf_final_layer_conv",
    ],
    "qwen": [
        "time_text_embed", "img_in", "norm_out", "proj_out", "txt_in",
    ],
    "wan": [
        "patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head", "img_emb",
    ],
    "ltx": [
        "adaln_single", "audio_adaln_single", "audio_caption_projection",
        "audio_patchify_proj", "audio_proj_out", "audio_scale_shift_table",
        "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single", "av_ca_video_scale_shift_adaln_single",
        "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
    ],
    "hunyuan_video": [
        "img_in", "txt_in", "time_in", "vector_in", "guidance_in",
        "final_layer",
    ],
    "sd15": [
        "time_embed", "label_emb", "input_blocks.0", "out.0", "out.1", "out.2",
    ],
    "sdxl": [
        "time_embed", "label_emb", "input_blocks.0", "out.0", "out.1", "out.2",
    ],
}


class RayUNETLoaderINT8:
    """
    Load diffusion models with INT8 tensorwise quantization.

    Accepts **any** safetensor format:
    - Pre-quantized INT8 checkpoints are loaded directly (fastest).
    - FP16 / BF16 / FP32 checkpoints are quantized to INT8 on the fly
      when *on_the_fly_quantization* is enabled.
    - FP8 (e4m3fn, e5m2) checkpoints are cast to BF16 first, then
      quantized to INT8.

    Inference uses fast ``torch._int_mm`` with an optional Triton
    fused-kernel fast-path on Ampere+ GPUs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                ),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp16", "bf16"],
                ),
                "model_type": (
                    [
                        "auto",
                        "flux",
                        "z-image",
                        "chroma",
                        "wan",
                        "ltx",
                        "qwen",
                        "hunyuan_video",
                        "sd15",
                        "sdxl",
                    ],
                ),
                "on_the_fly_quantization": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When enabled, non-INT8 weights (FP16, BF16, FP32, FP8) "
                            "are quantized to INT8 at load time.  Not needed for "
                            "pre-quantized INT8 checkpoints."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "Raylight/INT8"
    DESCRIPTION = (
        "Load a diffusion model with INT8 quantization for fast inference "
        "on Ampere+ GPUs.  Accepts pre-quantized INT8, FP16, BF16, FP32, "
        "and FP8 safetensors.  Enable on-the-fly quantization for non-INT8 "
        "checkpoints."
    )

    def load_unet(self, unet_name, weight_dtype, model_type, on_the_fly_quantization):
        if Int8TensorwiseOps is None:
            raise RuntimeError(
                "INT8 ops unavailable – update ComfyUI to a version that "
                "exports comfy.ops.manual_cast."
            )

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        # Configure the custom ops class for this load
        Int8TensorwiseOps.dynamic_quantize = on_the_fly_quantization
        Int8TensorwiseOps._is_prequantized = False

        # Set architecture-specific exclusions
        if model_type == "auto":
            # Try to infer from filename (best-effort)
            Int8TensorwiseOps.excluded_names = _guess_exclusions(unet_name)
        else:
            Int8TensorwiseOps.excluded_names = _MODEL_EXCLUSIONS.get(model_type, [])

        model_options: dict = {"custom_operations": Int8TensorwiseOps}

        # weight_dtype controls the precision for layers that are NOT
        # quantized to INT8 (excluded layers, or all layers when
        # on_the_fly_quantization is off and the checkpoint isn't INT8).
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        # "default" → let ComfyUI decide

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


def _guess_exclusions(unet_name: str) -> list[str]:
    """Best-effort architecture detection from the filename."""
    name = unet_name.lower()
    for key in _MODEL_EXCLUSIONS:
        if key.replace("-", "").replace("_", "") in name.replace("-", "").replace("_", ""):
            return _MODEL_EXCLUSIONS[key]
    return []


NODE_CLASS_MAPPINGS = {
    "RayUNETLoaderINT8": RayUNETLoaderINT8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayUNETLoaderINT8": "Load Diffusion Model INT8 (Raylight)",
}
