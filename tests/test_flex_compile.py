#!/usr/bin/env python3
import torch
import time
from raylight.distributed_modules.attention.xfuser_ring_patch import manual_block_attention
import logging

logging.basicConfig(level=logging.INFO)

def run_test():
    print("Testing manual_block_attention with mask to verify torch.compile...")
    
    b, heads, s, dim_head = 1, 4, 128, 64
    dim = heads * dim_head
    
    q = torch.randn(b, s, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, s, dim, dtype=torch.bfloat16, device="cuda")
    
    # Mask [b, 1, s, s]
    mask = torch.zeros(b, 1, s, s, dtype=torch.bfloat16, device="cuda")
    mask[:, :, :, 64:] = -10000.0
    
    # Warmup
    print("Running initial compilation step...")
    start = time.time()
    out, lse = manual_block_attention(q, k, v, mask=mask)
    torch.cuda.synchronize()
    print(f"First run took: {time.time() - start:.4f}s")
    
    # Cached run
    start = time.time()
    out, lse = manual_block_attention(q, k, v, mask=mask)
    torch.cuda.synchronize()
    print(f"Second run took: {time.time() - start:.4f}s")
    
    print("Test finished successfully.")

if __name__ == "__main__":
    run_test()
