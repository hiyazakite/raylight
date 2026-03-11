import re

path = '/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/dequant.py'
with open(path, 'r') as f:
    code = f.read()

# Restore the dequantize_tensor optimization
original = r"def dequantize_tensor\(tensor, dtype=None, dequant_dtype=None\):\n    qtype = getattr\(tensor, \"tensor_type\", None\)\n    oshape = getattr\(tensor, \"tensor_shape\", tensor\.shape\)\n\n    if qtype in TORCH_COMPATIBLE_QTYPES:\n        return tensor\.to\(dtype\)\n    elif qtype in dequantize_functions:\n        dequant_dtype = dtype if dequant_dtype == \"target\" else dequant_dtype\n        return dequantize\(tensor\.data, qtype, oshape, dtype=dequant_dtype\)\.to\(dtype\)\n    else:"

replacement = """def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    
    if hasattr(tensor, "tensor_shape"):
        oshape = tensor.tensor_shape
    else:
        oshape = getattr(tensor, "shape", None)
        
    # STRIP SUBCLASS HERE to bypass all `__torch_function__` overhead!
    # PyTorch >= 2.0: tensor.data returns another subclass! We MUST use as_subclass.
    native_tensor = tensor.as_subclass(torch.Tensor) if hasattr(tensor, "as_subclass") else tensor.data

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return native_tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(native_tensor, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:"""

code = re.sub(original, replacement, code)

with open(path, 'w') as f:
    f.write(code)

print("dequant.py dequantize_tensor restored!")
