import re

ops_path = '/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/dequant.py'
with open(ops_path, 'r') as f:
    text = f.read()

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

original = r"def dequantize_tensor\(tensor, dtype=None, dequant_dtype=None\):\n    qtype = getattr\(tensor, \"tensor_type\", None\)\n    oshape = getattr\(tensor, \"tensor_shape\", tensor\.shape\)\n\n    if qtype in TORCH_COMPATIBLE_QTYPES:\n        return tensor\.to\(dtype\)\n    elif qtype in dequantize_functions:\n        dequant_dtype = dtype if dequant_dtype == \"target\" else dequant_dtype\n        return dequantize\(tensor\.data, qtype, oshape, dtype=dequant_dtype\)\.to\(dtype\)\n    else:"

text = re.sub(original, replacement, text)
with open(ops_path, 'w') as f:
    f.write(text)

ops_file = '/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/ops.py'
with open(ops_file, 'r') as f:
    text2 = f.read()

rep_cast = """    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)

        def match_device(t, dev):
             if t is None or dev is None: return False
             return str(t.device) == str(dev) or (t.device.type == dev.type and (t.device.index or 0) == (dev.index or 0))

        if s.bias is not None:
            bias_t = s.bias if match_device(s.bias, device) else s.bias.to(device)
            bias = s.get_weight(bias_t, dtype)
            bias = comfy.ops.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight_t = s.weight if match_device(s.weight, device) else s.weight.to(device)
        weight = s.get_weight(weight_t, dtype)
        weight = comfy.ops.cast_to(
            weight, dtype, device, non_blocking=non_blocking, copy=False
        )
        return weight, bias"""

orig_cast = r"    @torch_compiler_disable\(\)\n    def cast_bias_weight\(s, input=None, dtype=None, device=None, bias_dtype=None\):\n        if input is not None:\n            if dtype is None:\n                dtype = getattr\(input, \"dtype\", torch\.float32\)\n            if bias_dtype is None:\n                bias_dtype = dtype\n            if device is None:\n                device = input\.device\n\n        bias = None\n        non_blocking = comfy\.model_management\.device_supports_non_blocking\(device\)\n        if s\.bias is not None:\n            bias_t = s\.bias if s\.bias\.device == device else s\.bias\.to\(device\)\n            bias = s\.get_weight\(bias_t, dtype\)\n            bias = comfy\.ops\.cast_to\(\n                bias, bias_dtype, device, non_blocking=non_blocking, copy=False\n            \)\n\n        weight_t = s\.weight if s\.weight\.device == device else s\.weight\.to\(device\)\n        weight = s\.get_weight\(weight_t, dtype\)\n        weight = comfy\.ops\.cast_to\(\n            weight, dtype, device, non_blocking=non_blocking, copy=False\n        \)\n        return weight, bias"

text2 = re.sub(orig_cast, rep_cast, text2)
with open(ops_file, 'w') as f:
    f.write(text2)

print("Patched dequant.py and ops.py")
