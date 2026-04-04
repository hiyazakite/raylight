"""Unit tests for LTXAV Tensor Parallelism.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

_src = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

_comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if os.path.isdir(_comfy_root) and _comfy_root not in sys.path:
    sys.path.insert(0, _comfy_root)

# Try imports, skipping if they fail (e.g. if comfy package isn't perfectly stubbed locally)
try:
    from comfy.ldm.lightricks.model import CrossAttention, FeedForward
    import comfy.ops
except ImportError:
    pytest.skip("Could not import ComfyUI LTXAV modules. Ensure ComfyUI path is correct.", allow_module_level=True)

from raylight.distributed_modules.tensor_parallel import TensorParallelState

from raylight.diffusion_models.lightricks.tp import (
    apply_tp_to_ltxav_cross_attention,
    apply_tp_to_ltxav_feedforward,
    _slice_rope_for_tp,
)

@pytest.fixture(autouse=True)
def _reset_tp_state():
    TensorParallelState.reset()
    yield
    TensorParallelState.reset()


def mock_tp(tp_size: int, tp_rank: int = 0) -> None:
    TensorParallelState._tp_size = tp_size
    TensorParallelState._tp_rank = tp_rank
    TensorParallelState._tp_group = None


class TestLTXAVTensorParallel:
    def test_slice_rope_for_tp_interleaved(self):
        # interleaved [B, T, D]
        B = 2
        T = 4
        H = 4 
        dim_head = 64
        D = H * dim_head
        
        # we have 4 heads, tp_size=2 -> local_heads=2
        local_heads = 2
        tp_size = 2
        tp_rank = 0
        
        cos = torch.randn(B, T, D)
        sin = torch.randn(B, T, D)
        
        pe = [(cos, sin, False)]
        
        sliced = _slice_rope_for_tp(pe, tp_rank, tp_size, local_heads)
        
        s_cos, s_sin, flag = sliced[0]
        assert not flag
        assert s_cos.shape == (B, T, D // 2)
        assert torch.equal(s_cos, cos[:, :, :128])
        assert torch.equal(s_sin, sin[:, :, :128])

    def test_slice_rope_for_tp_split(self):
        # split [B, H, T, R]
        B = 2
        H = 4
        T = 4
        R = 64
        
        local_heads = 2
        tp_size = 2
        tp_rank = 1
        
        cos = torch.randn(B, H, T, R)
        sin = torch.randn(B, H, T, R)
        
        pe = [(cos, sin, True)]
        
        sliced = _slice_rope_for_tp(pe, tp_rank, tp_size, local_heads)
        
        s_cos, s_sin, flag = sliced[0]
        assert flag
        assert s_cos.shape == (B, 2, T, R)
        assert torch.equal(s_cos, cos[:, 2:4, :, :])

    def test_apply_tp_to_ltxav_cross_attention(self):
        mock_tp(2)
        
        query_dim = 256
        context_dim = 128
        heads = 8
        dim_head = 64
        attn = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            apply_gated_attention=True,
            operations=comfy.ops.disable_weight_init
        )
        
        apply_tp_to_ltxav_cross_attention(attn)
        
        # Check QKV sharding (column)
        assert attn.to_q.weight.shape == (256, 256) # 8 * 64 // 2, 256
        assert attn.to_k.weight.shape == (256, 128)
        assert attn.to_v.weight.shape == (256, 128)
        
        # Check norm sharding
        assert attn.q_norm.full_hidden_size == 512
        assert attn.q_norm.local_hidden_size == 256
        assert attn.q_norm.weight.shape == (256,)
        
        # Check out sharding (row)
        assert attn.to_out[0].local_in_features == 256
        assert attn.to_out[0].weight.shape == (256, 256)
        
        # Check gated sharding
        assert attn.to_gate_logits.weight.shape == (8, 256) # In our implementation we didn't slice the gate out_features locally inside TPLinear, we do it in forward block. Wait, did we use TPLinear gather=False? If so we DO slice it!
        
        x = torch.randn(2, 10, query_dim)
        context = torch.randn(2, 10, context_dim)
        out = attn(x, context)
        # Should be correct shape!
        assert out.shape == (2, 10, query_dim)

    def test_apply_tp_to_ltxav_feedforward(self):
        mock_tp(2)
        
        dim = 256
        dim_out = 256
        mult = 2
        
        ff = FeedForward(dim=dim, dim_out=dim_out, mult=mult, operations=comfy.ops.disable_weight_init)
        
        apply_tp_to_ltxav_feedforward(ff)
        
        # inner_dim = dim * mult = 512
        # col parallel: local_inner_dim = 256
        assert ff.net[0].proj.weight.shape == (256, 256)
        # row parallel:
        assert ff.net[2].weight.shape == (256, 256)
        
        x = torch.randn(2, 10, dim)
        out = ff(x)
        assert out.shape == (2, 10, dim_out)


if __name__ == "__main__":
    pytest.main([
        __file__, "-v", "--noconftest",
        "--rootdir", os.path.dirname(__file__),
        "-c", "/dev/null",
        "--import-mode=importlib",
    ])
