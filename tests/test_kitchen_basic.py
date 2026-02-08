import torch
import torch.nn as nn
import sys
from unittest.mock import MagicMock

# Mock comfy_kitchen
mock_kitchen = MagicMock()
mock_kitchen.tensor.base.QuantizedTensor = MagicMock
mock_kitchen.tensor.fp8.TensorCoreFP8Layout = MagicMock
sys.modules["comfy_kitchen"] = mock_kitchen
sys.modules["comfy_kitchen.tensor.base"] = mock_kitchen.tensor.base
sys.modules["comfy_kitchen.tensor.fp8"] = mock_kitchen.tensor.fp8

# Define a fake QuantizedTensor that behaves like a Tensor
class FakeQuantizedTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)

# Mock the imports in our modules
import raylight.distributed_modules.kitchen_patch as kp
import raylight.distributed_modules.kitchen_linear as kl
import raylight.distributed_modules.quantize as kq

# Mock QuantizedTensor.from_float to return a tensor
def mock_from_float(data, layout):
    return data.clone().detach() # Return a normal tensor for FSDP simplicity in mock

# Inject mock
kl.QuantizedTensor.from_float = mock_from_float
kp.QuantizedTensor = FakeQuantizedTensor

def test_quantization_replacement():
    print("Testing quantization replacement...")
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    print("Original model:", model)
    
    kq.quantize_model(model)
    
    print("Quantized model:", model)
    
    assert isinstance(model[0], kl.KitchenQuantizedLinear), "Layer 0 should be replaced"
    assert isinstance(model[2], kl.KitchenQuantizedLinear), "Layer 2 should be replaced"
    assert isinstance(model[1], nn.ReLU), "ReLU should remain"
    
    print("Replacement success!")

def test_linear_forward():
    print("\nTesting KitchenQuantizedLinear forward...")
    layer = nn.Linear(10, 20)
    q_layer = kl.KitchenQuantizedLinear.from_linear(layer)
    
    input_dummy = torch.randn(1, 10)
    
    # We expect this to run F.linear. Since we mocked QuantizedTensor to be a normal tensor 
    # (via mock_from_float returning normal tensor), F.linear should work standardly.
    # In real execution, dispatch would handle it.
    
    out = q_layer(input_dummy)
    print("Forward output shape:", out.shape)
    assert out.shape == (1, 20)
    print("Forward pass success!")

if __name__ == "__main__":
    test_quantization_replacement()
    test_linear_forward()
