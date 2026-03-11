import sys
import os

# Create dummy modules to simulate the environment
class MockLongContextAttention:
    def __init__(self, *args, **kwargs):
        pass

# Setup yunchang mock
import sys
from types import ModuleType

yunchang_mock = ModuleType("yunchang")
yunchang_mock.LongContextAttention = MockLongContextAttention
sys.modules["yunchang"] = yunchang_mock

yunchang_kernels_mock = ModuleType("yunchang.kernels")
class AttnType:
    FA = "FA"
yunchang_kernels_mock.AttnType = AttnType
sys.modules["yunchang.kernels"] = yunchang_kernels_mock

yunchang_comm_mock = ModuleType("yunchang.comm")
yunchang_comm_all_to_all_mock = ModuleType("yunchang.comm.all_to_all")
class SeqAllToAll4D:
    pass
yunchang_comm_all_to_all_mock.SeqAllToAll4D = SeqAllToAll4D
sys.modules["yunchang.comm.all_to_all"] = yunchang_comm_all_to_all_mock


# Import the layer to test
sys.path.insert(0, "/root/ComfyUI/custom_nodes/raylight/src")
import raylight.distributed_modules.compact.attn_layer as al

def sim_generation_run():
    # Simulate ComfyUI model patcher creating blocks
    blocks = []
    print(f"Creating 5 layers...")
    for i in range(5):
        layer = al.xFuserLongContextAttention(attn_type=AttnType.FA)
        # Simulate forward pass setting the idx
        if layer.idx is None:
            layer.idx = al.ATTN_LAYER_IDX
            al.ATTN_LAYER_IDX += 1
        blocks.append(layer)
        
    print("Layer IDs in this generation:")
    for b in blocks:
        print(f"  - {b.idx}")
    return blocks

print(f"Initial ATTN_LAYER_IDX: {al.ATTN_LAYER_IDX}")

print("\n--- Simulation: Run 1 ---")
sim_generation_run()

print("\n--- Simulation: Run 2 (Simulating Reload) ---")
# ComfyUI destroys and recreates standard models/blocks sometimes
sim_generation_run()

print(f"\nFinal ATTN_LAYER_IDX: {al.ATTN_LAYER_IDX}")
