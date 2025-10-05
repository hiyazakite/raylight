import sys
import os
from pathlib import Path
import importlib.util

# Comfy dynamic lib loader cause the module imported as a path instead of normal python path
# e.g /home/user/ComfyUI/custom_nodes/raylight.src.raylight.etc
# this code will change it into :
# raylight.wanvideo or raylight.flux
# this is done since ray cluster needs libs to run in this case raylight.
# ray.init(runtime_env={"py_modules":[raylight]})
this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(this_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

comfy_dir = os.path.abspath(os.path.join(this_dir, "../../comfy"))
if comfy_dir not in sys.path:
    sys.path.insert(0, comfy_dir)


# For ComfyUI GGUF
try:
    current_file = Path(__file__).resolve()

    comfy_root = None
    for parent in current_file.parents:
        if parent.name == "ComfyUI":
            comfy_root = parent
            break

    gguf_path = comfy_root / "custom_nodes" / "ComfyUI-GGUF"
    module_init = gguf_path / "__init__.py"

    spec = importlib.util.spec_from_file_location("comfyui_gguf", module_init)
    comfyui_gguf = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_gguf"] = comfyui_gguf
    spec.loader.exec_module(comfyui_gguf)
except Exception as e:
    print("City96-GGUF Not available")
# For ComfyUI GGUF


from raylight.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Extra nodes
from raylight.comfy_extra_dist.nodes_torch_compile import NODE_CLASS_MAPPINGS as COMPILE_NODE_CLASS_MAPPINGS
from raylight.comfy_extra_dist.nodes_torch_compile import NODE_DISPLAY_NAME_MAPPINGS as COMPILE_DISPLAY_NAME_MAPPINGS
from raylight.comfy_extra_dist.nodes_model_advanced import NODE_CLASS_MAPPINGS as MODEL_ADV_CLASS_MAPPINGS

if os.getenv("debug_raylight") == "1":
    print("RAYLIGHT DEBUG MODE")
    from raylight.nodes_debug import NODE_CLASS_MAPPINGS as DEBUG_NODE_CLASS_NAME_MAPPINGS
    from raylight.nodes_debug import NODE_DISPLAY_NAME_MAPPINGS as DEBUG_NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(DEBUG_NODE_CLASS_NAME_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(DEBUG_NODE_DISPLAY_NAME_MAPPINGS)


# CLASS
NODE_CLASS_MAPPINGS.update(COMPILE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(MODEL_ADV_CLASS_MAPPINGS)


# DISPLAY
NODE_DISPLAY_NAME_MAPPINGS.update(COMPILE_DISPLAY_NAME_MAPPINGS)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Micko Lesmana"""
__email__ = "mickolesmana@gmail.com"
__version__ = "0.0.1"
