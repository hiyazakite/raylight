#!/usr/bin/env python3
import torch
from raylight.distributed_modules.attention.backends.standard import StandardAttentionBackend

def run_test():
    print("Testing StandardAttentionBackend with a mask...")
    
    # Setup dummy yunchang process group environment to avoid Assertion errors
    import torch.distributed as dist
    import yunchang
    from xfuser.core.distributed.parallel_state import init_distributed_environment, get_world_group
    
    # We must mock or setup minimal torch dist for yunchang to be happy
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    
    if not dist.is_initialized():
        dist.init_process_group("gloo")
        
    init_distributed_environment(rank=0, world_size=1)
        
    world_group = get_world_group().group if hasattr(get_world_group(), 'group') else dist.group.WORLD
    yunchang.set_seq_parallel_pg(None, world_group, None, world_group)

    backend = StandardAttentionBackend()
    
    # Init the function
    # Note: we need yunchang to avoid erroring if we don't pass a mask, 
    # but since we DO pass a mask, we expect it to hit the FlexAttention path.
    attn_fn = backend.create_attention(attn_type="FA", sync_ulysses=False)
    
    b, heads, s, dim_head = 1, 4, 128, 64
    dim = heads * dim_head
    
    q = torch.randn(b, s, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, s, dim, dtype=torch.bfloat16, device="cuda")
    
    # Mask shape [b, 1, s, s]
    mask = torch.zeros(b, 1, s, s, dtype=torch.bfloat16, device="cuda")
    mask[:, :, :, 64:] = -10000.0 # simple causal-ish mask
    
    out = attn_fn(q, k, v, heads=heads, mask=mask)
    print("Test finished successfully. Output shape:", out.shape)

if __name__ == "__main__":
    run_test()
