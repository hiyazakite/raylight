"""Unit tests for tensor_parallel.py — single-process, no real distributed backend.

Run standalone:
    python tests/test_tensor_parallel.py

Run via pytest (requires --noconftest --import-mode=importlib due to the
ComfyUI root __init__.py importing raylight.nodes):
    pytest tests/test_tensor_parallel.py -v --noconftest -c /dev/null --import-mode=importlib
"""

import sys
import os

# Ensure src/ is on the path so `raylight.distributed_modules` is importable
# even when running standalone outside the full ComfyUI environment.
_src = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Block the ComfyUI root __init__.py from loading (it tries to import
# raylight.nodes which is unavailable in a standalone test environment).
# We only need the src/raylight package.
_root_init = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _root_init in sys.path:
    sys.path.remove(_root_init)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from raylight.distributed_modules.tensor_parallel import (  # noqa: E402
    TensorParallelState,
    TPAttention,
    TPLinear,
    TPMLP,
    TPRMSNormAcrossHeads,
    split_tensor_along_dim,
)


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# TensorParallelState
# ---------------------------------------------------------------------------


class TestTensorParallelState:
    def test_defaults(self):
        assert TensorParallelState.get_size() == 1
        assert TensorParallelState.get_rank() == 0
        assert TensorParallelState.get_group() is None
        assert not TensorParallelState.is_initialized()

    def test_mock_and_reset(self):
        mock_tp(4, 2)
        assert TensorParallelState.get_size() == 4
        assert TensorParallelState.get_rank() == 2
        assert TensorParallelState.is_initialized()
        TensorParallelState.reset()
        assert not TensorParallelState.is_initialized()


# ---------------------------------------------------------------------------
# TPLinear
# ---------------------------------------------------------------------------


class TestTPLinear:
    def test_column_parallel_shapes(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=True, parallelism="column")
        assert layer.local_out_features == 8
        assert layer.local_in_features == 64
        assert layer.weight.shape == (8, 64)
        assert layer.bias.shape == (8,)

    def test_column_parallel_forward(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=True, parallelism="column")
        x = torch.randn(2, 10, 64)
        assert layer(x).shape == (2, 10, 8)

    def test_row_parallel_shapes(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=True, parallelism="row")
        assert layer.local_in_features == 16
        assert layer.local_out_features == 32
        assert layer.weight.shape == (32, 16)

    def test_row_parallel_forward(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=True, parallelism="row")
        x = torch.randn(2, 10, 16)
        assert layer(x).shape == (2, 10, 32)

    def test_no_bias(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=False, parallelism="column")
        assert layer.bias is None

    def test_column_weight_loader_rank0(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=True, parallelism="column")
        full_w = torch.randn(32, 64)
        layer.weight_loader(layer.weight, full_w)
        assert torch.equal(layer.weight.data, full_w[:8, :])

    def test_row_weight_loader_rank0(self):
        mock_tp(4)
        layer = TPLinear(64, 32, bias=True, parallelism="row")
        full_w = torch.randn(32, 64)
        layer.weight_loader(layer.weight, full_w)
        assert torch.equal(layer.weight.data, full_w[:, :16])

    def test_tp_size_1_is_identity(self):
        """With tp_size=1, col+row pair should behave exactly like F.linear."""
        mock_tp(1)
        col = TPLinear(64, 128, bias=False, parallelism="column")
        row = TPLinear(128, 64, bias=False, parallelism="row")
        x = torch.randn(2, 5, 64)
        out = row(col(x))
        expected = torch.nn.functional.linear(
            torch.nn.functional.linear(x, col.weight), row.weight
        )
        assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# TPAttention
# ---------------------------------------------------------------------------


class TestTPAttention:
    def test_mha(self):
        """MHA: 8 Q heads, 8 KV heads, tp_size=4 → 2 local each."""
        mock_tp(4)
        attn = TPAttention(
            hidden_size=512, head_size=64, total_num_heads=8, total_num_kv_heads=8
        )
        assert attn.num_heads == 2
        assert attn.num_kv_heads == 2
        assert attn.num_kv_head_replicas == 1
        x = torch.randn(2, 10, 512)
        assert attn(x).shape == (2, 10, 512)

    def test_gqa(self):
        """GQA: 8 Q heads, 2 KV heads, tp_size=2 → 4Q, 1KV per rank."""
        mock_tp(2)
        attn = TPAttention(
            hidden_size=512, head_size=64, total_num_heads=8, total_num_kv_heads=2
        )
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 1
        assert attn.num_kv_head_replicas == 1
        x = torch.randn(2, 10, 512)
        assert attn(x).shape == (2, 10, 512)

    def test_extreme_gqa_replicated(self):
        """Extreme GQA: tp_size=4 > kv_heads=2 → KV heads replicated."""
        mock_tp(4)
        attn = TPAttention(
            hidden_size=512, head_size=64, total_num_heads=8, total_num_kv_heads=2
        )
        assert attn.num_heads == 2
        assert attn.num_kv_heads == 1
        assert attn.num_kv_head_replicas == 2
        # k/v proj should be plain nn.Linear (not TPLinear) when replicated
        assert type(attn.k_proj).__name__ == "Linear"
        assert attn.k_proj.weight.shape[0] == 64  # 1 head * 64
        x = torch.randn(2, 10, 512)
        assert attn(x).shape == (2, 10, 512)

    def test_gqa_kv_expansion(self):
        """Verify KV heads are expanded to match Q heads in SDPA."""
        mock_tp(4)
        attn = TPAttention(
            hidden_size=256, head_size=32, total_num_heads=8, total_num_kv_heads=4
        )
        # tp_size=4, kv_heads=4 → 1 KV head per rank, 2 Q heads per rank
        assert attn.num_heads == 2
        assert attn.num_kv_heads == 1
        x = torch.randn(1, 5, 256)
        out = attn(x)
        assert out.shape == (1, 5, 256)


# ---------------------------------------------------------------------------
# TPMLP
# ---------------------------------------------------------------------------


class TestTPMLP:
    def test_shapes(self):
        mock_tp(4)
        mlp = TPMLP(hidden_size=512, intermediate_size=2048)
        assert mlp.w1.weight.shape == (512, 512)  # 2048/4, 512
        assert mlp.w3.weight.shape == (512, 512)  # 512, 2048/4

    def test_forward(self):
        mock_tp(4)
        mlp = TPMLP(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512)
        assert mlp(x).shape == (2, 10, 512)


# ---------------------------------------------------------------------------
# TPRMSNormAcrossHeads
# ---------------------------------------------------------------------------


class TestTPRMSNormAcrossHeads:
    def test_construction(self):
        mock_tp(4)
        norm = TPRMSNormAcrossHeads(full_hidden_size=512, local_hidden_size=128)
        assert norm.weight.shape == (128,)
        assert norm.full_hidden_size == 512


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


class TestSplitTensor:
    def test_split_basic(self):
        t = torch.randn(4, 8, 16)
        parts = split_tensor_along_dim(t, dim=1, num_splits=4)
        assert len(parts) == 4
        assert parts[0].shape == (4, 2, 16)

    def test_split_roundtrip(self):
        t = torch.randn(4, 8, 16)
        parts = split_tensor_along_dim(t, dim=1, num_splits=4)
        assert torch.equal(torch.cat(parts, dim=1), t)

    def test_split_1_is_noop(self):
        t = torch.randn(4, 8, 16)
        parts = split_tensor_along_dim(t, dim=1, num_splits=1)
        assert len(parts) == 1
        assert torch.equal(parts[0], t)


if __name__ == "__main__":
    pytest.main([
        __file__, "-v", "--noconftest",
        "--rootdir", os.path.dirname(__file__),
        "-c", "/dev/null",
        "--import-mode=importlib",
    ])
