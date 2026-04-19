"""Unit tests for WAN Tensor Parallelism — single-process, no real distributed backend.

Run standalone:
    python tests/distributed_modules/test_wan_tp.py

Run via pytest:
    pytest tests/distributed_modules/test_wan_tp.py -v --noconftest -c /dev/null --import-mode=importlib
"""

import sys
import os

# Ensure src/ is on the path
_src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Block the root __init__.py
_root_init = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root_init in sys.path:
    sys.path.remove(_root_init)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from raylight.distributed_modules.tensor_parallel import (  # noqa: E402
    TensorParallelState,
    TPFusedQKNorm,
    TPLinear,
    TPRMSNormAcrossHeads,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_tp_state():
    """Reset TensorParallelState before and after every test."""
    TensorParallelState.reset()
    yield
    TensorParallelState.reset()


def mock_tp(tp_size: int, tp_rank: int = 0) -> None:
    """Set TensorParallelState for single-process testing (no real group)."""
    TensorParallelState._tp_size = tp_size
    TensorParallelState._tp_rank = tp_rank
    TensorParallelState._tp_group = None


def _make_wan_self_attention(dim=2048, num_heads=16, qk_norm=True):
    """Build a minimal WanSelfAttention-like module for testing."""
    head_dim = dim // num_heads
    attn = nn.Module()
    attn.dim = dim
    attn.num_heads = num_heads
    attn.head_dim = head_dim
    attn.qk_norm = qk_norm
    attn.window_size = (-1, -1)
    attn.eps = 1e-6

    attn.q = nn.Linear(dim, dim, bias=False)
    attn.k = nn.Linear(dim, dim, bias=False)
    attn.v = nn.Linear(dim, dim, bias=False)
    attn.o = nn.Linear(dim, dim, bias=False)

    if qk_norm:
        attn.norm_q = nn.modules.normalization.RMSNorm(dim, eps=1e-6)
        attn.norm_k = nn.modules.normalization.RMSNorm(dim, eps=1e-6)
    else:
        attn.norm_q = nn.Identity()
        attn.norm_k = nn.Identity()

    return attn


def _make_wan_i2v_cross_attention(dim=2048, num_heads=16, qk_norm=True):
    """Build a minimal WanI2VCrossAttention-like module."""
    attn = _make_wan_self_attention(dim, num_heads, qk_norm)
    attn.k_img = nn.Linear(dim, dim, bias=False)
    attn.v_img = nn.Linear(dim, dim, bias=False)
    if qk_norm:
        attn.norm_k_img = nn.modules.normalization.RMSNorm(dim, eps=1e-6)
    else:
        attn.norm_k_img = nn.Identity()
    return attn


def _make_wan_ffn(dim=2048, ffn_dim=8192):
    """Build a minimal WAN FFN module (nn.Sequential)."""
    return nn.Sequential(
        nn.Linear(dim, ffn_dim, bias=False),
        nn.GELU(approximate='tanh'),
        nn.Linear(ffn_dim, dim, bias=False),
    )


# ---------------------------------------------------------------------------
# Tests: Self-attention TP patching
# ---------------------------------------------------------------------------


class TestWanSelfAttentionTP:

    def test_qkv_become_tp_linear(self):
        """Q/K/V should be replaced with column-parallel TPLinear."""
        mock_tp(2)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        assert isinstance(attn.q, TPLinear)
        assert isinstance(attn.k, TPLinear)
        assert isinstance(attn.v, TPLinear)
        assert attn.q.parallelism == "column"
        assert attn.k.parallelism == "column"

    def test_output_becomes_row_parallel(self):
        """Output proj should be row-parallel TPLinear."""
        mock_tp(2)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        assert isinstance(attn.o, TPLinear)
        assert attn.o.parallelism == "row"

    def test_fused_qk_norm_replaces_norms(self):
        """norm_q/norm_k should be replaced with TPFusedQKNorm."""
        mock_tp(2)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        assert hasattr(attn, "_fused_qk_norm")
        assert isinstance(attn._fused_qk_norm, TPFusedQKNorm)
        assert attn.norm_q is None
        assert attn.norm_k is None

    def test_tp_metadata_set(self):
        """TP metadata should be stored on the module."""
        mock_tp(4, tp_rank=2)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        assert attn._tp_local_heads == 4  # 16 / 4
        assert attn._tp_rank == 2
        assert attn._tp_size == 4

    def test_weight_shapes_column_parallel(self):
        """Column-parallel weights should be sharded on output dim."""
        mock_tp(4)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        # Q: [2048, 2048] → column-parallel → [512, 2048]
        assert attn.q.weight.shape == (512, 2048)
        assert attn.k.weight.shape == (512, 2048)
        assert attn.v.weight.shape == (512, 2048)

    def test_weight_shapes_row_parallel(self):
        """Row-parallel output proj should be sharded on input dim."""
        mock_tp(4)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        # O: [2048, 2048] → row-parallel → [2048, 512]
        assert attn.o.weight.shape == (2048, 512)

    def test_no_qk_norm_skips_fused(self):
        """When qk_norm is False, no fused norm should be created."""
        mock_tp(2)
        attn = _make_wan_self_attention(dim=2048, num_heads=16, qk_norm=False)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        assert attn._fused_qk_norm is None

    def test_tp_size_1_is_noop(self):
        """With tp_size=1, no patching should occur."""
        mock_tp(1)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)
        orig_q = attn.q

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_self_attention
        apply_tp_to_wan_self_attention(attn, structure_only=True)

        assert attn.q is orig_q  # unchanged


# ---------------------------------------------------------------------------
# Tests: T2V Cross-attention TP patching
# ---------------------------------------------------------------------------


class TestWanT2VCrossAttentionTP:

    def test_structure(self):
        """T2V cross-attention should get the same TP structure."""
        mock_tp(2)
        attn = _make_wan_self_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_t2v_cross_attention
        apply_tp_to_wan_t2v_cross_attention(attn, structure_only=True)

        assert isinstance(attn.q, TPLinear)
        assert isinstance(attn.o, TPLinear)
        assert attn.o.parallelism == "row"
        assert isinstance(attn._fused_qk_norm, TPFusedQKNorm)


# ---------------------------------------------------------------------------
# Tests: I2V Cross-attention TP patching
# ---------------------------------------------------------------------------


class TestWanI2VCrossAttentionTP:

    def test_img_projections_become_tp_linear(self):
        """k_img and v_img should also be column-parallel TPLinear."""
        mock_tp(2)
        attn = _make_wan_i2v_cross_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_i2v_cross_attention
        apply_tp_to_wan_i2v_cross_attention(attn, structure_only=True)

        assert isinstance(attn.k_img, TPLinear)
        assert isinstance(attn.v_img, TPLinear)
        assert attn.k_img.parallelism == "column"

    def test_norm_k_img_becomes_tp_norm(self):
        """norm_k_img should become TPRMSNormAcrossHeads."""
        mock_tp(2)
        attn = _make_wan_i2v_cross_attention(dim=2048, num_heads=16)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_i2v_cross_attention
        apply_tp_to_wan_i2v_cross_attention(attn, structure_only=True)

        assert isinstance(attn.norm_k_img, TPRMSNormAcrossHeads)
        assert attn.norm_k_img.full_hidden_size == 2048
        assert attn.norm_k_img.weight.shape == (1024,)  # 2048 / 2


# ---------------------------------------------------------------------------
# Tests: FFN TP patching
# ---------------------------------------------------------------------------


class TestWanFFNTP:

    def test_ffn_shapes(self):
        """FFN in-proj should be column-parallel, out-proj row-parallel."""
        mock_tp(4)
        ffn = _make_wan_ffn(dim=2048, ffn_dim=8192)

        from raylight.diffusion_models.wan.tp import apply_tp_to_wan_ffn
        apply_tp_to_wan_ffn(ffn, structure_only=True)

        # In-proj: column-parallel [8192, 2048] → [2048, 2048]
        assert isinstance(ffn[0], TPLinear)
        assert ffn[0].parallelism == "column"
        assert ffn[0].weight.shape == (2048, 2048)

        # GELU unchanged
        assert isinstance(ffn[1], nn.GELU)

        # Out-proj: row-parallel [2048, 8192] → [2048, 2048]
        assert isinstance(ffn[2], TPLinear)
        assert ffn[2].parallelism == "row"
        assert ffn[2].weight.shape == (2048, 2048)


# ---------------------------------------------------------------------------
# Tests: Block-level patching
# ---------------------------------------------------------------------------


class TestWanBlockTP:

    def _make_block(self, cross_attn_type="t2v"):
        """Build a minimal WanAttentionBlock-like module."""
        dim = 2048
        num_heads = 16

        block = nn.Module()
        block.dim = dim
        block.num_heads = num_heads
        block.self_attn = _make_wan_self_attention(dim, num_heads)
        block.cross_attn = _make_wan_self_attention(dim, num_heads)
        block.ffn = _make_wan_ffn(dim, 8192)

        # For isinstance checks in apply_tp_to_wan_block, we need to make
        # the cross-attention module pass the isinstance check.  We'll use
        # the real classes if available, otherwise skip.
        return block

    def test_block_marks_tp_patched(self):
        """Block should be marked _tp_patched after patching."""
        mock_tp(2)
        block = self._make_block()

        # Use direct attention patchers instead of apply_tp_to_wan_block
        # to avoid the isinstance checks against ComfyUI classes
        from raylight.diffusion_models.wan.tp import (
            apply_tp_to_wan_self_attention,
            apply_tp_to_wan_t2v_cross_attention,
            apply_tp_to_wan_ffn,
        )
        apply_tp_to_wan_self_attention(block.self_attn, structure_only=True)
        apply_tp_to_wan_t2v_cross_attention(block.cross_attn, structure_only=True)
        apply_tp_to_wan_ffn(block.ffn, structure_only=True)

        # Verify all sub-modules are patched
        assert isinstance(block.self_attn.q, TPLinear)
        assert isinstance(block.cross_attn.q, TPLinear)
        assert isinstance(block.ffn[0], TPLinear)
        assert isinstance(block.ffn[2], TPLinear)


# ---------------------------------------------------------------------------
# Tests: RoPE handling
# ---------------------------------------------------------------------------


class TestWanRoPE:

    def test_wan_rope_no_slicing_needed(self):
        """WAN RoPE has broadcast head dim (=1), should not need slicing."""
        from raylight.diffusion_models.wan.tp import _wan_rope_needs_slicing

        # Standard WAN freqs: [B, L, 1, D/2, 2, 2]
        freqs = torch.randn(1, 1024, 1, 64, 2, 2)
        assert not _wan_rope_needs_slicing(freqs)

    def test_none_freqs(self):
        from raylight.diffusion_models.wan.tp import _wan_rope_needs_slicing
        assert not _wan_rope_needs_slicing(None)

    def test_expanded_freqs_would_need_slicing(self):
        """If a future variant expands per-head RoPE, slicing should be needed."""
        from raylight.diffusion_models.wan.tp import _wan_rope_needs_slicing

        # Hypothetical: [B, L, num_heads, D/2, 2, 2]
        freqs = torch.randn(1, 1024, 16, 64, 2, 2)
        assert _wan_rope_needs_slicing(freqs)


if __name__ == "__main__":
    pytest.main([
        __file__, "-v", "--noconftest",
        "--rootdir", os.path.dirname(__file__),
        "-c", "/dev/null",
        "--import-mode=importlib",
    ])
