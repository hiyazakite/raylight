import os
import re

# 1. Patch `ops.py:GGMLTensor.to`
ops_path = '/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/ops.py'
with open(ops_path, 'r') as f:
    ops_content = f.read()

# Current to:
#     def to(self, *args, **kwargs):
#         new = super().to(*args, **kwargs)
#         if new is self:
#             return self
#         new.tensor_type = getattr(self, "tensor_type", None)
#         new.tensor_shape = getattr(self, "tensor_shape", new.size())
#         new.patches = getattr(self, "patches", []).copy()
#         return new

new_to = """    def to(self, *args, **kwargs):
        native = self.as_subclass(torch.Tensor)
        new = native.to(*args, **kwargs)
        if new is native:
            return self
        new = new.as_subclass(GGMLTensor)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.size())
        new.patches = getattr(self, "patches", []).copy()
        return new"""

ops_content = re.sub(
    r"    def to\(self, \*args, \*\*kwargs\):\n        new = super\(\)\.to\(\*args, \*\*kwargs\)\n        if new is self:\n            return self\n        new\.tensor_type = getattr\(self, \"tensor_type\", None\)\n        new\.tensor_shape = getattr\(self, \"tensor_shape\", new\.size\(\)\)\n        new\.patches = getattr\(self, \"patches\", \[\]\)\.copy\(\)\n        return new",
    new_to,
    ops_content
)
with open(ops_path, 'w') as f:
    f.write(ops_content)

print(f"Patched {ops_path} successfully!")

# 2. Patch all torch.cuda.empty_cache() inside src/raylight
def remove_empty_cache(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Replace but keep the structure
                if 'torch.cuda.empty_cache()' in content:
                    content = content.replace('torch.cuda.empty_cache()', 'pass # torch.cuda.empty_cache() bypassed to prevent allocator fragmentation')
                    with open(filepath, 'w') as f:
                        f.write(content)
                    print(f"Patched empty_cache in {filepath}")

remove_empty_cache('/root/ComfyUI/custom_nodes/raylight/src/raylight')
print("All patches complete.")
