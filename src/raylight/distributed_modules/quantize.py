import torch
import torch.nn as nn
from .kitchen_patch import apply_patches
from .kitchen_linear import KitchenQuantizedLinear

# Flag to ensure we only patch once
_PATCHES_APPLIED = False

def quantize_model(model, layout="fp8", ignore_modules=None):
    """
    Quantize linear layers of the model using ComfyUI-Kitchen (FP8).
    
    Args:
        model: The model to quantize.
        layout: "fp8" (tensor core E4M3) is the primary target.
        ignore_modules: List of module names/types to skip.
    """
    global _PATCHES_APPLIED
    if not _PATCHES_APPLIED:
        apply_patches()
        _PATCHES_APPLIED = True
        
    print(f"[Raylight] Quantizing model to {layout} using ComfyUI-Kitchen...")
    
    count = 0
    
    # Recursively replace Linear layers
    # We use a stack or explicit recursion to modify in-place
    
    for name, module in model.named_modules():
        # We can't modify the dict while iterating, so we iterate and collect replacements
        # Actually named_modules is a generator.
        # Better approach: recursive function
        pass

    _replace_linear_recursive(model, ignore_modules)
    
    print(f"[Raylight] Quantization complete.")

def _replace_linear_recursive(module, ignore_modules=None):
    """
    Recursively replace nn.Linear with KitchenQuantizedLinear
    """
    for name, child in module.named_children():
        if ignore_modules and name in ignore_modules:
            continue
            
        if isinstance(child, nn.Linear):
            # Check if we should skip specific layers (e.g. output layer might need high precision?)
            # For now, quantize all.
            
            # Create quantized wrapper
            new_layer = KitchenQuantizedLinear.from_linear(child, quantize_weights=True)
            
            # Replace
            setattr(module, name, new_layer)
        else:
            # Recurse
            _replace_linear_recursive(child, ignore_modules)
