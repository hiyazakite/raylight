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
_src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Block the ComfyUI root __init__.py from loading (it tries to import
# raylight.nodes which is unavailable in a standalone test environment).
# We only need the src/raylight package.
_root_init = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root_init in sys.path:
    sys.path.remove(_root_init)

import pytest  # noqa: E402
import builtins  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from raylight.distributed_modules.tensor_parallel import (  # noqa: E402
    TensorParallelState,
    TPAttention,
    TPLinear,
    TPMLP,
    TPRMSNormAcrossHeads,
    split_tensor_along_dim,
    _DeferredGather,
    _is_collective_deferred,
    defer_tp_collectives,
    gather_tensor_along_dim_async,
)

import raylight.distributed_modules.tp_linear_factory as _tplf  # noqa: E402
from raylight.distributed_modules.tp_linear_factory import (  # noqa: E402
    warmup_scratch_buffer,
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


# ---------------------------------------------------------------------------
# _DeferredGather
# ---------------------------------------------------------------------------


class TestDeferredGather:
    def test_wait_returns_tensor(self):
        t = torch.randn(2, 8)
        dg = _DeferredGather(t, work=None, full_size=None, dim=0)
        assert torch.equal(dg.wait(), t)

    def test_wait_trims_to_full_size(self):
        # Simulate padded gather output (12 cols) that should be trimmed to 10.
        t = torch.randn(2, 12)
        dg = _DeferredGather(t, work=None, full_size=10, dim=1)
        result = dg.wait()
        assert result.shape == (2, 10)
        assert torch.equal(result, t[:, :10])

    def test_wait_trims_dim0(self):
        t = torch.randn(12, 4)
        dg = _DeferredGather(t, work=None, full_size=10, dim=0)
        result = dg.wait()
        assert result.shape == (10, 4)

    def test_no_trim_when_exact(self):
        t = torch.randn(2, 10)
        dg = _DeferredGather(t, work=None, full_size=10, dim=1)
        assert torch.equal(dg.wait(), t)


# ---------------------------------------------------------------------------
# defer_tp_collectives context manager
# ---------------------------------------------------------------------------


class TestDeferTPCollectives:
    def test_default_not_deferred(self):
        assert not _is_collective_deferred()

    def test_deferred_inside_context(self):
        with defer_tp_collectives():
            assert _is_collective_deferred()
        assert not _is_collective_deferred()

    def test_nesting_restores_outer_state(self):
        assert not _is_collective_deferred()
        with defer_tp_collectives():
            assert _is_collective_deferred()
            with defer_tp_collectives():
                assert _is_collective_deferred()
            assert _is_collective_deferred()
        assert not _is_collective_deferred()

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with defer_tp_collectives():
                assert _is_collective_deferred()
                raise RuntimeError("boom")
        assert not _is_collective_deferred()


# ---------------------------------------------------------------------------
# gather_tensor_along_dim_async (single-process / tp_size=1)
# ---------------------------------------------------------------------------


class TestGatherTensorAlongDimAsync:
    def test_noop_when_tp1(self):
        """With tp_size=1 (default), async gather is a passthrough."""
        t = torch.randn(2, 8)
        handle = gather_tensor_along_dim_async(t, dim=-1, num_splits=1)
        result = handle.wait()
        assert torch.equal(result, t)

    def test_noop_when_explicit_splits1(self):
        t = torch.randn(3, 4, 6)
        handle = gather_tensor_along_dim_async(t, dim=1, num_splits=1)
        assert torch.equal(handle.wait(), t)


# ---------------------------------------------------------------------------
# TPLinear defer integration (single-process)
# ---------------------------------------------------------------------------


class TestTPLinearDefer:
    """Verify that TPLinear.forward skips the collective when deferred."""

    def test_column_forward_deferred_returns_local_shard(self):
        mock_tp(2)
        layer = TPLinear(16, 8, bias=False, parallelism="column")
        x = torch.randn(1, 4, 16)

        # Normal forward: output dim should be local (no real group → no gather)
        normal_out = layer(x)

        # Deferred forward: same shape since tp_group is None (no gather anyway)
        with defer_tp_collectives():
            deferred_out = layer(x)

        assert normal_out.shape == deferred_out.shape
        assert torch.equal(normal_out, deferred_out)

    def test_row_forward_deferred_returns_local_shard(self):
        mock_tp(2)
        layer = TPLinear(16, 8, bias=False, parallelism="row")
        x = torch.randn(1, 4, 8)  # row-parallel expects sharded input

        normal_out = layer(x)
        with defer_tp_collectives():
            deferred_out = layer(x)

        assert normal_out.shape == deferred_out.shape
        assert torch.equal(normal_out, deferred_out)


# ---------------------------------------------------------------------------
# warmup_scratch_buffer
# ---------------------------------------------------------------------------


class _FakeTPGGMLLinear(nn.Module):
    """Minimal stand-in for TPGGMLLinear — only ``shard_shape`` is needed."""

    def __init__(self, shard_shape):
        super().__init__()
        self.shard_shape = shard_shape

    # So isinstance() check works in warmup_scratch_buffer
    pass


class TestWarmupScratchBuffer:
    @pytest.fixture(autouse=True)
    def _reset_scratch(self):
        """Clear module-level scratch state before/after each test."""
        _tplf._shared_dequant_scratch = None
        _tplf._scratch_pinned_dtype = None
        yield
        _tplf._shared_dequant_scratch = None
        _tplf._scratch_pinned_dtype = None

    def _patch_isinstance(self, monkeypatch):
        """Make _FakeTPGGMLLinear pass isinstance checks in warmup."""
        real_isinstance = builtins.isinstance

        def _patched(obj, cls):
            if cls is _tplf.TPGGMLLinear and type(obj) is _FakeTPGGMLLinear:
                return True
            return real_isinstance(obj, cls)

        monkeypatch.setattr(builtins, "isinstance", _patched)

    def test_warmup_finds_max_shard(self, monkeypatch):
        self._patch_isinstance(monkeypatch)
        parent = nn.Sequential(
            _FakeTPGGMLLinear(torch.Size((1024, 512))),    # 524_288
            _FakeTPGGMLLinear(torch.Size((4096, 3072))),   # 12_582_912  ← max
            _FakeTPGGMLLinear(torch.Size((3072, 1024))),   # 3_145_728
        )
        result = warmup_scratch_buffer(parent, dtype=torch.float16, device=torch.device("cpu"))
        assert result == 4096 * 3072
        assert _tplf._shared_dequant_scratch is not None
        assert _tplf._shared_dequant_scratch.numel() == 4096 * 3072
        assert _tplf._shared_dequant_scratch.dtype == torch.float16
        assert _tplf._scratch_pinned_dtype == torch.float16

    def test_warmup_no_modules_returns_zero(self, monkeypatch):
        self._patch_isinstance(monkeypatch)
        parent = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
        result = warmup_scratch_buffer(parent, dtype=torch.float16, device=torch.device("cpu"))
        assert result == 0
        assert _tplf._shared_dequant_scratch is None

    def test_warmup_unwraps_model_patcher(self, monkeypatch):
        """Simulate ComfyUI ModelPatcher → BaseModel → diffusion_model chain."""
        self._patch_isinstance(monkeypatch)
        diff_model = nn.Sequential(_FakeTPGGMLLinear(torch.Size((512, 256))))

        class FakeBaseModel:
            diffusion_model = diff_model

        class FakePatcher:
            model = FakeBaseModel()

        result = warmup_scratch_buffer(FakePatcher(), dtype=torch.float16, device=torch.device("cpu"))
        assert result == 512 * 256
        assert _tplf._shared_dequant_scratch is not None

    def test_warmup_skips_none_shard_shape(self, monkeypatch):
        self._patch_isinstance(monkeypatch)
        m = _FakeTPGGMLLinear(None)
        parent = nn.Sequential(m)
        result = warmup_scratch_buffer(parent, dtype=torch.float16, device=torch.device("cpu"))
        assert result == 0

    def test_warmup_dtype_bf16(self, monkeypatch):
        self._patch_isinstance(monkeypatch)
        parent = nn.Sequential(_FakeTPGGMLLinear(torch.Size((256, 128))))
        result = warmup_scratch_buffer(parent, dtype=torch.bfloat16, device=torch.device("cpu"))
        assert result == 256 * 128
        assert _tplf._shared_dequant_scratch.dtype == torch.bfloat16
        assert _tplf._scratch_pinned_dtype == torch.bfloat16

    def test_warmup_default_dtype_is_bf16(self, monkeypatch):
        """Default dtype should be bfloat16 (matches modern diffusion models)."""
        self._patch_isinstance(monkeypatch)
        parent = nn.Sequential(_FakeTPGGMLLinear(torch.Size((256, 128))))
        result = warmup_scratch_buffer(parent, device=torch.device("cpu"))
        assert result == 256 * 128
        assert _tplf._shared_dequant_scratch.dtype == torch.bfloat16
        assert _tplf._scratch_pinned_dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([
        __file__, "-v", "--noconftest",
        "--rootdir", os.path.dirname(__file__),
        "-c", "/dev/null",
        "--import-mode=importlib",
    ])
