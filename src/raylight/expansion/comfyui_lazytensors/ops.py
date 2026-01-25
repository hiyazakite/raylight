"""Custom operations for zero-copy safetensor loading.

Implements custom layers that override _load_from_state_dict to perform
pointer assignment instead of data copying. This allows LazySafetensor
wrappers to replace module parameters without triggering materialization/copy.
"""
import torch
import comfy.ops
import comfy.lora
import comfy.model_management
from .lazy_tensor import LazySafetensor
from raylight.comfy_dist.lora import calculate_weight as ray_calculate_weight

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    return item

class SafetensorLayer(torch.nn.Module):
    """Mixin for layers that handle LazySafetensor assignment."""
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"
        weight = state_dict.get(weight_key)
        
        # CRITICAL: If weight is a lazy tensor, assign it directly!
        # Standard _load_from_state_dict performs .copy_() which forces materialization
        if isinstance(weight, LazySafetensor) or isinstance(self, torch.nn.Linear):
             return self.eager_load_params(state_dict, prefix, *args, **kwargs)
             
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def eager_load_params(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Manually assign parameters to avoid copy overhead."""
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k.startswith(prefix):
                 # Remove prefix to identify attribute
                 attr_name = k[prefix_len:]
                 if attr_name == "weight":
                     self.weight = torch.nn.Parameter(v, requires_grad=False)
                 elif attr_name == "bias" and v is not None:
                     self.bias = torch.nn.Parameter(v, requires_grad=False)
                 # Handle internal layers/submodules if necessary? 
                 # Usually state_dict is flat, so this loop mainly hits weight/bias for this layer
        
        # Handle missing keys logic similar to PyTorch
        if self.weight is None:
             if isinstance(self, torch.nn.Linear):
                  # Create dummy if needed, but usually we expect weight
                  pass
             missing_keys.append(prefix + "weight")

    def get_weight(self, tensor, dtype, device):
        """Get weight applying patches on GPU if needed."""
        # Transfer to device (triggers lazy materialization)
        weight = tensor.to(device)
        
        # Check for patches (ComfyUI attaches .patches list to tensor)
        patches = getattr(tensor, "patches", [])
        if len(patches) > 0:
            # Move patches to GPU
            patch_list = []
            last_key = None
            for p, key in patches:
                patch_list += move_patch_to_device(p, device)
                last_key = key
            
            # Apply patches on GPU
            # Using ray_calculate_weight (same as GGUF) for consistent LoRA math
            weight = ray_calculate_weight(patch_list, weight, last_key)
            
        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        
        # Handle bias
        if self.bias is not None:
            # Use get_weight to handle potential lazy + patches on bias (rare but possible)
            bias = self.get_weight(self.bias, bias_dtype, device)
            bias = comfy.ops.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        # Handle weight - this triggers LazySafetensorTensor.to() which materializes
        # AND applies patches if present
        weight = self.get_weight(self.weight, dtype, device)
        weight = comfy.ops.cast_to(
            weight, dtype, device, non_blocking=non_blocking, copy=False
        )
        return weight, bias


class SafetensorOps(comfy.ops.manual_cast):
    """Operations factory that produces zero-copy aware layers."""
    
    class Linear(SafetensorLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(SafetensorLayer, comfy.ops.manual_cast.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)
            
    class Embedding(SafetensorLayer, comfy.ops.manual_cast.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            self.weight = None

        def forward(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight is not None and (self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16):
                out_dtype = None
            
            # Use get_weight logic (via cast_bias_weight internal logic or direct)
            # Embedding doesn't use cast_bias_weight standardly in Comfy ops, so we adapt:
            weight = self.get_weight(self.weight, None, input.device)
            
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

    class LayerNorm(SafetensorLayer, comfy.ops.manual_cast.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            self.weight = None
            self.bias = None

        def forward(self, input):
            if self.weight is None:
                return super().forward(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class GroupNorm(SafetensorLayer, comfy.ops.manual_cast.GroupNorm):
        def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )
