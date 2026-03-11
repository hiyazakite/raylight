import re

ops_path = '/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/ops.py'
with open(ops_path, 'r') as f:
    ops_content = f.read()

# 1. ops.py -> GGMLTensor.to
to_orig = """    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new"""

to_new = """    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        if new is self:
            return self
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.size())
        new.patches = getattr(self, "patches", []).copy()
        return new"""
ops_content = ops_content.replace(to_orig, to_new)

# 2. ops.py -> get_weight
get_weight_orig = """        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)"""
get_weight_new = """        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = weight.as_subclass(torch.Tensor)"""
ops_content = ops_content.replace(get_weight_orig, get_weight_new)

# 3. ops.py -> cast_bias_weight
cast_orig = """        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight = s.get_weight(s.weight.to(device), dtype)"""
cast_new = """        def match_device(t, dev):
             if t is None or dev is None: return False
             return str(t.device) == str(dev) or (t.device.type == dev.type and (t.device.index or 0) == (dev.index or 0))

        if s.bias is not None:
            bias_t = s.bias if match_device(s.bias, device) else s.bias.to(device)
            bias = s.get_weight(bias_t, dtype)
            bias = comfy.ops.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight_t = s.weight if match_device(s.weight, device) else s.weight.to(device)
        weight = s.get_weight(weight_t, dtype)"""
ops_content = ops_content.replace(cast_orig, cast_new)

# 4. ops.py -> forward_comfy_cast_weights
fwd_orig = """        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)"""
fwd_new = """        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = out.as_subclass(torch.Tensor)"""
ops_content = ops_content.replace(fwd_orig, fwd_new)

with open(ops_path, 'w') as f:
    f.write(ops_content)

print("ops.py restored successfully!")
