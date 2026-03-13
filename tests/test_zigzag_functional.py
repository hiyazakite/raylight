import torch
import torch.nn.functional as F
import sys
import os

def update_out_and_lse(out, lse, block_out, block_lse, slice_=None):
    if out is None:
        if slice_ is not None:
             out = torch.zeros_like(block_out) # simplification for test
             lse = torch.full_like(block_lse, float("-inf"))
        else:
             return block_out.clone(), block_lse.clone()

    if slice_ is not None:
        out_slice = out[slice_]
        lse_slice = lse[slice_]
    else:
        out_slice = out
        lse_slice = lse

    new_lse = torch.logsumexp(torch.stack([lse_slice, block_lse], dim=0), dim=0)
    
    out_slice.copy_(
        torch.exp(lse_slice - new_lse) * out_slice + 
        torch.exp(block_lse - new_lse) * block_out
    )
    lse_slice.copy_(new_lse)
    return out, lse

def run_zigzag_test():
    print("Running Simplified Zigzag Functional Test...")
    device = "cpu" # Stay on CPU to avoid CUDA/NCCL issues
    dtype = torch.float32 # Use float32 for better numerical precision in test
    
    bs, seqlen, heads, dim_head = 1, 64, 2, 32
    world_size = 2
    
    q_full = torch.randn(bs, seqlen, heads, dim_head, device=device, dtype=dtype)
    k_full = torch.randn(bs, seqlen, heads, dim_head, device=device, dtype=dtype)
    v_full = torch.randn(bs, seqlen, heads, dim_head, device=device, dtype=dtype)
    
    # Reference Attention (Causal)
    q_ref = q_full.transpose(1, 2)
    k_ref = k_full.transpose(1, 2)
    v_ref = v_full.transpose(1, 2)
    ref_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    ref_out = ref_out.transpose(1, 2).contiguous()

    # Zigzag Partitioning
    block_seq_len = seqlen // (2 * world_size)
    q_chunks = q_full.chunk(2 * world_size, dim=1)
    k_chunks = k_full.chunk(2 * world_size, dim=1)
    v_chunks = v_full.chunk(2 * world_size, dim=1)

    def get_zigzag_local(chunks, rank):
        return torch.cat([chunks[rank], chunks[2 * world_size - rank - 1]], dim=1)

    q_locals = [get_zigzag_local(q_chunks, r) for r in range(world_size)]
    k_locals = [get_zigzag_local(k_chunks, r) for r in range(world_size)]
    v_locals = [get_zigzag_local(v_chunks, r) for r in range(world_size)]

    def iterate_attn(_q, _k, _v, _causal):
        _qt = _q.transpose(1, 2)
        _kt = _k.transpose(1, 2)
        _vt = _v.transpose(1, 2)
        _out = F.scaled_dot_product_attention(_qt, _kt, _vt, is_causal=_causal)
        
        # Compute LSE manually
        _attn = torch.matmul(_qt, _kt.transpose(-2, -1)) * (_q.shape[-1]**-0.5)
        if _causal:
            mask = torch.triu(torch.ones(_attn.shape[-2:], device=_q.device), diagonal=1).bool()
            _attn.masked_fill_(mask, float("-inf"))
        _lse = torch.logsumexp(_attn.float(), dim=-1)
        return _out.transpose(1, 2).contiguous(), _lse.unsqueeze(-1)

    final_outs = [None] * world_size
    final_lses = [None] * world_size

    for rank in range(world_size):
        q = q_locals[rank]
        out = None
        lse = None
        
        # Simulated Ring Loop
        for step in range(world_size):
            recv_rank = (rank + step) % world_size
            k_target = k_locals[recv_rank]
            v_target = v_locals[recv_rank]
            
            if step == 0:
                # This matches our patched kernel logic for step 0
                block_out, block_lse = iterate_attn(q, k_target, v_target, True)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= rank:
                # rank 1 attends to K0 (first half)
                k0 = k_target[:, :block_seq_len]
                v0 = v_target[:, :block_seq_len]
                block_out, block_lse = iterate_attn(q, k0, v0, False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                # rank 0 attends to K1 (via Q1)
                q1 = q[:, block_seq_len:]
                block_out, block_lse = iterate_attn(q1, k_target, v_target, False)
                out, lse = update_out_and_lse(
                    out, lse, block_out, block_lse, 
                    slice_=(slice(None), slice(block_seq_len, None))
                )
        
        final_outs[rank] = out
        final_lses[rank] = lse

    # Reassemble results
    # rank r has blocks [r, 2*world_size - r - 1]
    # We need to put them back in global order
    assembled_out = torch.zeros_like(q_full)
    for r in range(world_size):
        out = final_outs[r]
        assembled_out.chunk(2 * world_size, dim=1)[r].copy_(out[:, :block_seq_len])
        assembled_out.chunk(2 * world_size, dim=1)[2 * world_size - r - 1].copy_(out[:, block_seq_len:])

    diff = (assembled_out - ref_out).abs().max().item()
    print(f"Max difference: {diff}")
    if diff < 1e-5:
        print("MATCH SUCCESS!")
    else:
        print("MATCH FAILED!")

if __name__ == "__main__":
    run_zigzag_test()
