
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload
from comfy_kitchen.tensor.base import QuantizedTensor, register_layout_op
from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout
import os
import sys

# Mock distributed environment (single process for behavior check logic, 
# although FSDP needs dist group. We can assume we are rank 0 of 1 to check flattening)
# To check memory reduction we really need >1 rank.
# However, we can inspect the module structure after sharding using a mock process group if possible
# or just check if the parameter is replaced by a "FlatParameter" or if it stays as QuantizedTensor.
# If it stays as QuantizedTensor with full shape, sharding didn't happen as expected (storage-wise).

# Let's try to inspect what FSDP does to a QuantizedTensor param with use_orig_params=True.

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a large weight
        self.linear = nn.Linear(1024, 1024, bias=False)
        
    def convert_to_fp8(self):
        q_weight = QuantizedTensor.from_float(self.linear.weight.data, "TensorCoreFP8Layout")
        self.linear.weight = nn.Parameter(q_weight, requires_grad=False)

def check_fsdp_behavior(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Init process group
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    model = SimpleModel()
    model.convert_to_fp8()
    
    print(f"[Rank {rank}] Before Sharding: {model.linear.weight.data.shape}")
    
    # FSDP2 (fully_shard) usually doesn't need use_orig_params=True as it shards params natively.
    # However, for QuantizedTensor (subclass), we need check if it survives.
    
    try:
        model = fully_shard(model) # Removed use_orig_params=True which is invalid for FSDP2
    except Exception as e:
        print(f"[Rank {rank}] fully_shard failed: {e}")
        return

    # Check slicing
    raw_qdata = model.linear.weight._qdata
    storage_size = raw_qdata.storage().size()
    element_size = raw_qdata.element_size()
    total_bytes = storage_size * element_size
    
    # Also check shape
    sharded_shape = model.linear.weight.shape
    local_tensor = model.linear.weight.to_local() if hasattr(model.linear.weight, "to_local") else model.linear.weight

    print(f"[Rank {rank}] After Sharding: _qdata storage bytes: {total_bytes}")
    print(f"[Rank {rank}] Layer weight shape (global view): {sharded_shape}")
    print(f"[Rank {rank}] Local tensor shape: {local_tensor.shape}")
    
    if total_bytes == 1024*1024:
         print(f"[Rank {rank}] FAIL: Storage size unchanged. Sharding did NOT happen.")
    else:
         print(f"[Rank {rank}] PASS: Storage size reduced. Sharding HAPPENED.")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    world_size = 2
    mp.spawn(check_fsdp_behavior, args=(world_size,), nprocs=world_size, join=True)

