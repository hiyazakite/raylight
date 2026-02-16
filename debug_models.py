import sys
import os
import torch
import comfy.model_base as model_base

print(f"Inspecting comfy.model_base...")
if hasattr(model_base, "Flux"):
    print(f"Flux: {model_base.Flux}")
else:
    print("Flux: Missing")

if hasattr(model_base, "WAN21"):
    print(f"WAN21: {model_base.WAN21}")
    if hasattr(model_base, "Flux"):
        print(f"WAN21 inherits Flux? {issubclass(model_base.WAN21, model_base.Flux)}")
        print(f"Flux inherits WAN21? {issubclass(model_base.Flux, model_base.WAN21)}")
    else:
        print("Cannot check inheritance (Flux missing)")
else:
    print("WAN21: Missing")

if hasattr(model_base, "WAN22"):
    print(f"WAN22: {model_base.WAN22}")
else:
    print("WAN22: Missing")

try:
    print(f"Available attributes in model_base: {dir(model_base)}")
except:
    pass
