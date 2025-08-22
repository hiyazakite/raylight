import sys
import os

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


from raylight.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Extra nodes
from raylight.extra.nodes_torch_compile import NODE_CLASS_MAPPINGS as COMPILE_NODE_CLASS_MAPPINGS
from raylight.extra.nodes_torch_compile import NODE_DISPLAY_NAME_MAPPINGS as COMPILE_DISPLAY_NAME_MAPPINGS


# CLASS
NODE_CLASS_MAPPINGS.update(COMPILE_NODE_CLASS_MAPPINGS)


# DISPLAY
NODE_DISPLAY_NAME_MAPPINGS.update(COMPILE_DISPLAY_NAME_MAPPINGS)
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Micko Lesmana"""
__email__ = "mickolesmana@gmail.com"
__version__ = "0.0.1"
