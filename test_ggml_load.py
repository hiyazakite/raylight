
import torch
import sys
import os

# Mocking Comfy/Raylight environment for test
from raylight.expansion.comfyui_gguf.ops import GGMLOps, GGMLTensor

class MockStateDict(dict):
    pass

def test_ggml_layer_load():
    print("Testing GGMLOps.Linear loading behavior...")
    
    # 1. Create a "mmap" tensor (GGMLTensor)
    raw_storage = torch.randn(10, 10)
    mmap_tensor = GGMLTensor(raw_storage, tensor_type=0, tensor_shape=(10, 10))
    print(f"Mmap Tensor Ptr: {mmap_tensor.data_ptr()}")
    
    # 2. Create Layer
    layer = GGMLOps.Linear(10, 10, bias=False)
    
    # 3. Load State Dict
    sd = {"weight": mmap_tensor}
    
    # Manually trigger load like PyTorch/Comfy does
    # _load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    layer._load_from_state_dict(sd, "", {}, True, [], [], [])
    
    # 4. Check Parameter
    if layer.weight is None:
        print("ERROR: Layer weight is None!")
        return
        
    print(f"Layer Weight Ptr: {layer.weight.data_ptr()}")
    
    if layer.weight.data_ptr() == mmap_tensor.data_ptr():
        print("SUCCESS: Zero-copy preserved!")
    else:
        print("FAILURE: Memory copy detected!")

if __name__ == "__main__":
    test_ggml_layer_load()
