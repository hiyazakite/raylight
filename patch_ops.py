import re

with open('/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/ops.py', 'r') as f:
    text = f.read()

# Fix GGMLTensor.to
to_replacement = """    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        if new is self:
            return self
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.size())
        new.patches = getattr(self, "patches", []).copy()
        return new
"""
text = re.sub(
    r"    def to\(self, \*args, \*\*kwargs\):\n        new = super\(\)\.to\(\*args, \*\*kwargs\)\n        new\.tensor_type = getattr\(self, \"tensor_type\", None\)\n        new\.tensor_shape = getattr\(self, \"tensor_shape\", new\.data\.shape\)\n        new\.patches = getattr\(self, \"patches\", \[\]\)\.copy\(\)\n        return new\n",
    to_replacement,
    text
)

# Fix subclass clearing in get_weight
strip_replacement = """        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = weight.as_subclass(torch.Tensor)"""

text = re.sub(
    r"        # prevent propagating custom tensor class\n        if isinstance\(weight, GGMLTensor\):\n            weight = torch\.Tensor\(weight\)",
    strip_replacement,
    text
)

# Fix s.weight.to(device) in cast_bias_weight
cast_replacement = """        if s.bias is not None:
            bias_t = s.bias if s.bias.device == device else s.bias.to(device)
            bias = s.get_weight(bias_t, dtype)
            bias = comfy.ops.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight_t = s.weight if s.weight.device == device else s.weight.to(device)
        weight = s.get_weight(weight_t, dtype)
        weight = comfy.ops.cast_to(
            weight, dtype, device, non_blocking=non_blocking, copy=False
        )"""

text = re.sub(
    r"        if s\.bias is not None:\n            bias = s\.get_weight\(s\.bias\.to\(device\), dtype\)\n            bias = comfy\.ops\.cast_to\(\n                bias, bias_dtype, device, non_blocking=non_blocking, copy=False\n            \)\n\n        weight = s\.get_weight\(s\.weight\.to\(device\), dtype\)\n        weight = comfy\.ops\.cast_to\(\n            weight, dtype, device, non_blocking=non_blocking, copy=False\n        \)",
    cast_replacement,
    text
)

# Also fix the one in forward_comfy_cast_weights
comfy_fw_rep = """        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = out.as_subclass(torch.Tensor)"""
text = re.sub(
    r"        # non-ggml forward might still propagate custom tensor class\n        if isinstance\(out, GGMLTensor\):\n            out = torch\.Tensor\(out\)",
    comfy_fw_rep,
    text
)

with open('/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/ops.py', 'w') as f:
    f.write(text)

print("Patched ops.py successfully!")
