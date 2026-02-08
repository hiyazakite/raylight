import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy_kitchen.tensor.base import QuantizedTensor

class KitchenQuantizedLinear(nn.Module):
    """
    A Linear layer that wraps weights in QuantizedTensor (FP8)
    and quantizes inputs on-the-fly during forward pass.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We initialize as standard parameters first
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, quantize_weights=True):
        """
        Create a KitchenQuantizedLinear from an existing nn.Linear.
        If quantize_weights is True, immediately converts weight to QuantizedTensor.
        """
        obj = cls(
            linear_layer.in_features, 
            linear_layer.out_features, 
            bias=linear_layer.bias is not None, 
            device=linear_layer.weight.device, 
            dtype=linear_layer.weight.dtype
        )
        
        # Copy data
        if quantize_weights:
            # Check if weight is already FP8
            if linear_layer.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # ALREADY QUANTIZED: Wrap directly without re-quantization
                # We need to manually construct the QuantizedTensor
                # Assuming "TensorCoreFP8Layout" logic: qdata is the tensor itself, and params are implicit/dummy?
                # Actually, QuantizedTensor needs layout params (like scale).
                # If loading raw FP8, we might benefit from assuming specific scale or standard handling.
                # However, Comfortable/Kitchen usually quantizes from float.
                # For now, let's try to cast to float and re-quantize to ensure correct layout/scale metadata is generated.
                # This is safer than hacking the layout construction manually.
                w_float = linear_layer.weight.float()
                q_weight = QuantizedTensor.from_float(w_float, "TensorCoreFP8Layout")
            else:
                # Create QuantizedTensor from the float/bf16 weight
                q_weight = QuantizedTensor.from_float(linear_layer.weight.data, "TensorCoreFP8Layout")
                
            obj.weight = nn.Parameter(q_weight, requires_grad=False)
        else:
            obj.weight.data.copy_(linear_layer.weight.data)
            
        if linear_layer.bias is not None:
            obj.bias.data.copy_(linear_layer.bias.data)
            
        return obj

    def forward(self, input):
        # 1. Quantize input to FP8
        if isinstance(input, QuantizedTensor):
            input_q = input
        else:
            # On-the-fly quantization of activation
            input_q = QuantizedTensor.from_float(input, "TensorCoreFP8Layout")
            
        # 2. FP8 Linear
        # The dispatch mechanism in comfy-kitchen will handle this
        # provided both input and weight are QuantizedTensors.
        out = F.linear(input_q, self.weight, self.bias)
        
        # 3. Output handling
        # comfy-kitchen linear returns a QuantizedTensor or torch.Tensor depending on impl.
        # The _handle_fp8_linear we saw returns QuantizedTensor(output_fp8).
        # We likely want to dequantize back to high precision for the rest of the network (activations).
        
        if isinstance(out, QuantizedTensor):
            out = out.dequantize()
            
        # Ensure output is in original dtype (usually BF16)
        if out.dtype != input.dtype:
            out = out.to(dtype=input.dtype)
            
        return out
