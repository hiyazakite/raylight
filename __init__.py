import sys
import os

# This file is the ComfyUI custom_node entry point. It is ONLY relevant when
# ComfyUI loads this directory as a plugin. Guard all ComfyUI-specific imports
# so the file is safe to import during testing or packaging.

# Detect if we're running inside ComfyUI (not pytest, not pip, not plain Python)
_COMFY_ENTRY = (
    "comfy" in sys.modules
    or os.environ.get("COMFYUI_RUNNING") == "1"
    or any("ComfyUI" in p and "custom_nodes" not in p for p in sys.path)
)

# Always ensure src/ is on the path so `import raylight.*` works
this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(this_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if not _COMFY_ENTRY:
    # Not running inside ComfyUI — skip node registration.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    # Running inside ComfyUI — register all nodes.
    comfy_dir = os.path.abspath(os.path.join(this_dir, "../../comfy"))
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)

    from raylight.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    # Extra nodes
    from raylight.comfy_extra_dist.nodes_torch_compile import NODE_CLASS_MAPPINGS as COMPILE_NODE_CLASS_MAPPINGS
    from raylight.comfy_extra_dist.nodes_torch_compile import NODE_DISPLAY_NAME_MAPPINGS as COMPILE_DISPLAY_NAME_MAPPINGS

    from raylight.comfy_extra_dist.nodes_model_advanced import NODE_CLASS_MAPPINGS as MODEL_ADV_CLASS_MAPPINGS
    from raylight.comfy_extra_dist.nodes_easycache import NODE_CLASS_MAPPINGS as EASY_CACHE_NODE_CLASS_MAPPINGS
    from raylight.comfy_extra_dist.nodes_easycache import NODE_DISPLAY_NAME_MAPPINGS as EASY_CACHE_DISPLAY_NAME_MAPPINGS

    from raylight.comfy_extra_dist.nodes_custom_sampler import NODE_CLASS_MAPPINGS as SAMPLER_CLASS_MAPPINGS
    from raylight.comfy_extra_dist.nodes_custom_sampler import NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_DISPLAY_MAPPINGS

    from raylight.comfy_extra_dist.nodes_ltx_ffn_chunker import NODE_CLASS_MAPPINGS as LTX_FFN_NODE_CLASS_MAPPINGS
    from raylight.comfy_extra_dist.nodes_ltx_ffn_chunker import NODE_DISPLAY_NAME_MAPPINGS as LTX_FFN_NODE_DISPLAY_NAME_MAPPINGS

    from raylight.comfy_extra_dist.nodes_cachedit import NODE_CLASS_MAPPINGS as CACHEDIT_CLASS_MAPPINGS
    from raylight.comfy_extra_dist.nodes_cachedit import NODE_DISPLAY_NAME_MAPPINGS as CACHEDIT_DISPLAY_MAPPINGS

    try:
        from raylight.comfy_extra_dist.int8 import NODE_CLASS_MAPPINGS as INT8_NODE_CLASS_MAPPINGS
        from raylight.comfy_extra_dist.int8 import NODE_DISPLAY_NAME_MAPPINGS as INT8_NODE_DISPLAY_NAME_MAPPINGS
    except Exception as e:
        print(f"[Raylight] INT8 nodes unavailable (triton / comfy-kitchen required): {e}")
        INT8_NODE_CLASS_MAPPINGS = {}
        INT8_NODE_DISPLAY_NAME_MAPPINGS = {}

    if os.getenv("debug_raylight") == "1":
        print("RAYLIGHT DEBUG MODE")
        from raylight.nodes_debug import NODE_CLASS_MAPPINGS as DEBUG_NODE_CLASS_NAME_MAPPINGS
        from raylight.nodes_debug import NODE_DISPLAY_NAME_MAPPINGS as DEBUG_NODE_DISPLAY_NAME_MAPPINGS
        NODE_CLASS_MAPPINGS.update(DEBUG_NODE_CLASS_NAME_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(DEBUG_NODE_DISPLAY_NAME_MAPPINGS)

    gguf_dir = os.path.join(this_dir, "..", "ComfyUI-GGUF")
    gguf_dir = os.path.abspath(gguf_dir)

    if os.path.isdir(gguf_dir):
        from raylight.expansion.comfyui_gguf.nodes import NODE_CLASS_MAPPINGS as GGUF_NODE_CLASS_MAPPINGS
        from raylight.expansion.comfyui_gguf.nodes import NODE_DISPLAY_NAME_MAPPINGS as GGUF_NODE_DISPLAY_NAME_MAPPINGS
        NODE_CLASS_MAPPINGS.update(GGUF_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(GGUF_NODE_DISPLAY_NAME_MAPPINGS)
    else:
        print("City96 GGUF not found, GGUF ray loader disable")

    # CLASS
    NODE_CLASS_MAPPINGS.update(COMPILE_NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(MODEL_ADV_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(EASY_CACHE_NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(SAMPLER_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(LTX_FFN_NODE_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(CACHEDIT_CLASS_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(INT8_NODE_CLASS_MAPPINGS)

    # DISPLAY
    NODE_DISPLAY_NAME_MAPPINGS.update(COMPILE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(EASY_CACHE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(SAMPLER_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LTX_FFN_NODE_DISPLAY_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CACHEDIT_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(INT8_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Micko Lesmana"""
__email__ = "mickolesmana@gmail.com"
__version__ = "0.16.0"
