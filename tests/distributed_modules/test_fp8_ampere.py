"""Tests for FP8 Ampere weight packing and Fp8AmpereLinear.

Covers:
  - quantize_to_fp8_e4m3:  round-trip accuracy, shape/dtype contract.
  - compute_fp8_int8_scales: scale values are finite, positive, correct dtype.
  - compute_fp8_scales: per-channel BF16 scales (1.0).
  - repack_fp8_for_marlin: Marlin layout repacking with alignment padding.
  - narrow_fp8_weight_for_tp: correct slice shapes for col- and row-parallel.
  - Fp8AmpereLinear factories: from_linear, from_fp8_checkpoint, from_fp8_weight.
  - Fp8AmpereLinear.comfy_cast_weights is False.
  - Fp8AmpereLinear fallback forward: correct output shape, dtype, close to
    reference F.linear output.
  - Fp8AmpereLinear Marlin forward (Ampere-only): numerical agreement with
    reference at FP8 quantisation tolerance.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from raylight.distributed_modules.fp8_ampere.packing import (
    compute_fp8_int8_scales,
    compute_fp8_scales,
    narrow_fp8_weight_for_tp,
    quantize_to_fp8_e4m3,
    repack_fp8_for_marlin,
)
from raylight.distributed_modules.fp8_ampere.fp8_ampere_linear import (
    Fp8AmpereLinear,
    Fp8KernelMixin,
    _is_ampere_only,
)

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_ampere = pytest.mark.skipif(
    not (torch.cuda.is_available() and _is_ampere_only(torch.device("cuda"))),
    reason="Ampere GPU (SM≥8.0) required",
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


@pytest.fixture
def cuda_device():
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# packing.py — quantize_to_fp8_e4m3
# ---------------------------------------------------------------------------


class TestQuantizeToFp8E4m3:
    def test_output_shapes(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        w_fp8, ws = quantize_to_fp8_e4m3(w)
        assert w_fp8.shape == w.shape
        assert ws.shape == (64,)

    def test_output_dtypes(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        w_fp8, ws = quantize_to_fp8_e4m3(w)
        assert w_fp8.dtype == torch.uint8, f"expected uint8, got {w_fp8.dtype}"
        assert ws.dtype == torch.bfloat16, f"expected bfloat16, got {ws.dtype}"

    def test_scale_positive_finite(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        _, ws = quantize_to_fp8_e4m3(w)
        ws_f = ws.float()
        assert torch.all(ws_f > 0), "all scales must be positive"
        assert torch.all(torch.isfinite(ws_f)), "all scales must be finite"

    def test_round_trip_accuracy(self):
        """Decoded FP8 should be close to original weight (FP8 quant error only)."""
        # quantize_to_fp8_e4m3 encodes weights directly without per-row scaling, so
        # decoded_fp8 ≈ w with only FP8 rounding error (~12% relative for 3-bit mantissa).
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        w_fp8, ws = quantize_to_fp8_e4m3(w)
        # Decode: uint8 → float8_e4m3fn → float32
        decoded = w_fp8.view(torch.float8_e4m3fn).to(torch.float32)
        # Relative error: normalise per-element by per-row absmax
        scale_ref = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        rel_err = (w.float() - decoded).abs() / scale_ref
        assert rel_err.mean().item() < 0.15, (
            f"Mean relative round-trip error too large: {rel_err.mean():.4f} "
            f"(FP8 E4M3 has 3 mantissa bits → ~12% expected)"
        )

    def test_requires_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            quantize_to_fp8_e4m3(torch.randn(4, 8, 16, dtype=torch.bfloat16))

    def test_float32_input_accepted(self):
        w = torch.randn(32, 64, dtype=torch.float32)
        w_fp8, ws = quantize_to_fp8_e4m3(w)
        assert w_fp8.shape == w.shape
        assert ws.dtype == torch.bfloat16

    @requires_cuda
    def test_cuda_tensor(self, cuda_device):
        w = torch.randn(64, 128, dtype=torch.bfloat16, device=cuda_device)
        w_fp8, ws = quantize_to_fp8_e4m3(w)
        assert w_fp8.device.type == "cuda"
        assert ws.device.type == "cuda"


# ---------------------------------------------------------------------------
# packing.py — compute_fp8_int8_scales
# ---------------------------------------------------------------------------


class TestComputeFp8Int8Scales:
    def test_from_uint8(self):
        w = torch.randn(32, 64, dtype=torch.bfloat16)
        w_fp8, _ = quantize_to_fp8_e4m3(w)
        ws = compute_fp8_int8_scales(w_fp8)
        assert ws.shape == (32,)
        assert ws.dtype == torch.float16
        assert torch.all(ws.float() > 0)

    def test_from_float8_e4m3fn(self):
        w = torch.randn(32, 64, dtype=torch.bfloat16)
        w_fp8, _ = quantize_to_fp8_e4m3(w)
        w_native = w_fp8.view(torch.float8_e4m3fn)
        ws = compute_fp8_int8_scales(w_native)
        assert ws.shape == (32,)
        assert ws.dtype == torch.float16

    def test_invalid_dtype_raises(self):
        w = torch.randn(32, 64, dtype=torch.float32)
        with pytest.raises(TypeError, match="uint8 or float8_e4m3fn"):
            compute_fp8_int8_scales(w)

    def test_requires_2d_input(self):
        w = torch.zeros(4, 8, 16, dtype=torch.uint8)
        with pytest.raises(ValueError, match="2-D"):
            compute_fp8_int8_scales(w)

    def test_scale_magnitude(self):
        """Scale should be ≈ absmax(w) / 127 since decoded_fp8 ≈ w (no pre-scaling)."""
        w = torch.randn(16, 64, dtype=torch.bfloat16)
        w_fp8, _ = quantize_to_fp8_e4m3(w)
        ws = compute_fp8_int8_scales(w_fp8)
        # Decoded FP8 absmax per row should be ≤ 127 * ws (with small tolerance)
        decoded = w_fp8.view(torch.float8_e4m3fn).to(torch.float32)
        row_absmax = decoded.abs().amax(dim=1)
        ws_f = ws.float()
        assert torch.all(row_absmax <= ws_f * 127.0 * 1.01), (
            "Some rows exceed the INT8 range implied by their scale"
        )
        # ws should be on the same order as absmax(w)/127 (typical range for random weights)
        expected_ws_approx = w.float().abs().amax(dim=1) / 127.0
        ratio = ws_f / expected_ws_approx.clamp(min=1e-8)
        assert ratio.mean().item() < 2.0, (
            f"ws much larger than absmax(w)/127: mean ratio={ratio.mean():.2f}"
        )


# ---------------------------------------------------------------------------
# packing.py — compute_fp8_scales
# ---------------------------------------------------------------------------


class TestComputeFp8Scales:
    def test_returns_ones(self):
        w = torch.randn(32, 64, dtype=torch.bfloat16)
        w_fp8, _ = quantize_to_fp8_e4m3(w)
        ws = compute_fp8_scales(w_fp8)
        assert ws.shape == (32,)
        assert ws.dtype == torch.bfloat16
        assert torch.all(ws == 1.0)

    def test_requires_2d_input(self):
        w = torch.zeros(4, 8, 16, dtype=torch.uint8)
        with pytest.raises(ValueError, match="2-D"):
            compute_fp8_scales(w)


# ---------------------------------------------------------------------------
# packing.py — repack_fp8_for_marlin
# ---------------------------------------------------------------------------


class TestRepackFp8ForMarlin:
    def test_output_shape_aligned(self):
        """Repack produces flat [K_pad * N_pad] tiled-permuted layout."""
        N, K = 256, 64
        w = torch.randint(0, 256, (N, K), dtype=torch.uint8)
        packed, _, orig_n, orig_k = repack_fp8_for_marlin(w)
        # N=256 already multiple of 128, K=64 already multiple of 32
        assert packed.shape == (K * N,)
        assert orig_n == N
        assert orig_k == K

    def test_output_shape_small(self):
        """Small dimensions are padded to tile boundaries."""
        N, K = 64, 128
        w = torch.randint(0, 256, (N, K), dtype=torch.uint8)
        packed, scale, orig_n, orig_k = repack_fp8_for_marlin(w)
        # N=64 padded to 128, K=128 already aligned
        assert packed.shape == (128 * 128,)
        assert orig_n == 64
        assert orig_k == 128

    def test_scale_passthrough(self):
        """Scale vector padded to N_pad; original values preserved."""
        N, K = 64, 128
        w = torch.randint(0, 256, (N, K), dtype=torch.uint8)
        scale = torch.ones(N, dtype=torch.bfloat16)
        _, scale_out, _, _ = repack_fp8_for_marlin(w, scale)
        # N=64 padded to 128
        assert scale_out.shape == (128,)
        assert torch.all(scale_out[:N] == 1.0)
        assert torch.all(scale_out[N:] == 1.0)  # padding filled with 1.0

    @requires_cuda
    def test_cuda_tensor(self, cuda_device):
        N, K = 256, 64
        w = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=cuda_device)
        packed, _, _, _ = repack_fp8_for_marlin(w)
        assert packed.device.type == "cuda"


# ---------------------------------------------------------------------------
# packing.py — narrow_fp8_weight_for_tp
# ---------------------------------------------------------------------------


class TestNarrowFp8WeightForTp:
    @pytest.fixture
    def packed_weight(self):
        w = torch.randn(64, 128, dtype=torch.bfloat16)
        return quantize_to_fp8_e4m3(w)

    def test_column_parallel_shapes(self, packed_weight):
        w_fp8, ws = packed_weight
        for rank in range(4):
            w_s, ws_s = narrow_fp8_weight_for_tp(w_fp8, ws, dim=0, rank=rank, tp_size=4)
            assert w_s.shape == (16, 128), f"rank {rank}: {w_s.shape}"
            assert ws_s.shape == (16,), f"rank {rank}: {ws_s.shape}"

    def test_row_parallel_shapes(self, packed_weight):
        w_fp8, ws = packed_weight
        for rank in range(4):
            w_s, ws_s = narrow_fp8_weight_for_tp(w_fp8, ws, dim=1, rank=rank, tp_size=4)
            assert w_s.shape == (64, 32), f"rank {rank}: {w_s.shape}"
            assert ws_s.shape == (64,), "row-parallel: scale must be full [N]"

    def test_column_parallel_no_overlap(self, packed_weight):
        """Shards must tile the full weight without overlap."""
        w_fp8, ws = packed_weight
        shards = [
            narrow_fp8_weight_for_tp(w_fp8, ws, dim=0, rank=r, tp_size=4)[0]
            for r in range(4)
        ]
        reconstructed = torch.cat(shards, dim=0)
        assert torch.equal(reconstructed, w_fp8)

    def test_tp_size_1_is_identity(self, packed_weight):
        w_fp8, ws = packed_weight
        w_s, ws_s = narrow_fp8_weight_for_tp(w_fp8, ws, dim=0, rank=0, tp_size=1)
        assert torch.equal(w_s, w_fp8)
        assert torch.equal(ws_s, ws)

    def test_non_divisible_raises(self, packed_weight):
        w_fp8, ws = packed_weight
        with pytest.raises(ValueError, match="not divisible"):
            narrow_fp8_weight_for_tp(w_fp8, ws, dim=0, rank=0, tp_size=3)

    def test_invalid_dim_raises(self, packed_weight):
        w_fp8, ws = packed_weight
        with pytest.raises(ValueError, match="dim must be 0"):
            narrow_fp8_weight_for_tp(w_fp8, ws, dim=2, rank=0, tp_size=4)


# ---------------------------------------------------------------------------
# Fp8AmpereLinear — construction
# ---------------------------------------------------------------------------


class TestFp8AmpereLinearConstruction:
    def test_comfy_cast_weights_is_false(self):
        """ComfyUI must never upcast this module's weights."""
        lin = nn.Linear(64, 32, bias=False)
        fp8_lin = Fp8AmpereLinear.from_linear(lin)
        assert fp8_lin.comfy_cast_weights is False

    def test_from_linear_weight_dtype(self):
        lin = nn.Linear(128, 64, bias=True)
        fp8_lin = Fp8AmpereLinear.from_linear(lin)
        assert fp8_lin.weight.dtype == torch.uint8
        assert fp8_lin.weight_scale.dtype == torch.bfloat16
        assert fp8_lin.bias.dtype == torch.bfloat16

    def test_from_linear_shapes(self):
        N, K = 64, 128
        lin = nn.Linear(K, N, bias=True)
        fp8_lin = Fp8AmpereLinear.from_linear(lin)
        assert fp8_lin.weight.shape == (N, K)
        assert fp8_lin.weight_scale.shape == (N,)
        assert fp8_lin.bias.shape == (N,)

    def test_from_linear_no_bias(self):
        lin = nn.Linear(64, 32, bias=False)
        fp8_lin = Fp8AmpereLinear.from_linear(lin)
        assert fp8_lin.bias is None

    def test_from_fp8_checkpoint_uint8(self):
        w_fp8, _ = quantize_to_fp8_e4m3(torch.randn(64, 128, dtype=torch.bfloat16))
        fp8_lin = Fp8AmpereLinear.from_fp8_checkpoint(w_fp8)
        assert fp8_lin.weight.shape == (64, 128)
        assert fp8_lin.weight_scale.dtype == torch.bfloat16

    def test_from_fp8_checkpoint_float8(self):
        w_fp8, _ = quantize_to_fp8_e4m3(torch.randn(64, 128, dtype=torch.bfloat16))
        w_native = w_fp8.view(torch.float8_e4m3fn)
        fp8_lin = Fp8AmpereLinear.from_fp8_checkpoint(w_native)
        assert fp8_lin.weight.dtype == torch.uint8

    def test_from_fp8_checkpoint_bad_dtype(self):
        w = torch.randn(64, 128, dtype=torch.float32)
        with pytest.raises(TypeError, match="float8_e4m3fn or uint8"):
            Fp8AmpereLinear.from_fp8_checkpoint(w)

    def test_from_fp8_weight_direct(self):
        w_fp8, ws = quantize_to_fp8_e4m3(torch.randn(64, 128, dtype=torch.bfloat16))
        bias = torch.zeros(64, dtype=torch.bfloat16)
        fp8_lin = Fp8AmpereLinear.from_fp8_weight(w_fp8, ws, bias)
        assert fp8_lin.in_features  == 128
        assert fp8_lin.out_features == 64
        assert fp8_lin.bias is not None

    def test_extra_repr(self):
        fp8_lin = Fp8AmpereLinear.from_linear(nn.Linear(32, 16))
        r = fp8_lin.extra_repr()
        assert "in_features=32" in r
        assert "out_features=16" in r


# ---------------------------------------------------------------------------
# Fp8AmpereLinear — fallback forward (CPU or non-Ampere)
# ---------------------------------------------------------------------------


class TestFp8AmpereLinearFallbackForward:
    """Runs on CPU — exercises the _forward_fallback path."""

    def _make_pair(self, in_f=128, out_f=64, bias=True):
        torch.manual_seed(0)
        lin = nn.Linear(in_f, out_f, bias=bias)
        fp8_lin = Fp8AmpereLinear.from_linear(lin)
        return lin, fp8_lin

    def test_output_shape_2d(self):
        lin, fp8_lin = self._make_pair()
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        out = fp8_lin(x)
        assert out.shape == (8, 64)

    def test_output_shape_3d(self):
        lin, fp8_lin = self._make_pair()
        x = torch.randn(4, 8, 128, dtype=torch.bfloat16)
        out = fp8_lin(x)
        assert out.shape == (4, 8, 64)

    def test_output_dtype_bfloat16(self):
        lin, fp8_lin = self._make_pair()
        x = torch.randn(8, 128, dtype=torch.bfloat16)
        out = fp8_lin(x)
        assert out.dtype == torch.bfloat16

    def test_no_bias(self):
        lin, fp8_lin = self._make_pair(bias=False)
        x = torch.randn(4, 128, dtype=torch.bfloat16)
        out = fp8_lin(x)
        assert out.shape == (4, 64)

    def test_float32_input_accepted(self):
        """Should upcast/convert gracefully."""
        _, fp8_lin = self._make_pair()
        x = torch.randn(4, 128, dtype=torch.float32)
        out = fp8_lin(x)   # should not raise
        assert out.dtype == torch.bfloat16

    def test_accuracy_vs_reference(self):
        """FP8 round-trip + INT8 approximate — expect <3% mean rel error on CPU."""
        torch.manual_seed(1)
        in_f, out_f = 256, 128
        lin = nn.Linear(in_f, out_f, bias=True)
        fp8_lin = Fp8AmpereLinear.from_linear(lin)

        x = torch.randn(16, in_f, dtype=torch.bfloat16)
        ref = lin(x.float()).bfloat16()   # reference in BF16

        with torch.no_grad():
            out = fp8_lin(x)

        # Relative error normalised by per-sample output magnitude
        scale = ref.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        rel_err = (ref - out).abs() / scale
        assert rel_err.mean().item() < 0.05, (
            f"Fallback forward mean relative error too large: {rel_err.mean():.4f}"
        )


# ---------------------------------------------------------------------------
# Fp8AmpereLinear — IMMA forward (Ampere GPU required)
# ---------------------------------------------------------------------------


class TestFp8AmpereLinearMarlinForward:

    @requires_cuda
    @requires_ampere
    def test_marlin_output_shape(self, cuda_device):
        lin = nn.Linear(128, 64, bias=False).to(cuda_device)
        fp8_lin = Fp8AmpereLinear.from_linear(lin).to(cuda_device)

        ext = fp8_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        x = torch.randn(8, 128, dtype=torch.bfloat16, device=cuda_device)
        out = fp8_lin(x)
        assert out.shape == (8, 64)
        assert out.dtype == torch.bfloat16

    @requires_cuda
    @requires_ampere
    def test_marlin_accuracy_vs_reference(self, cuda_device):
        """Marlin output should be close to BF16 reference (FP8 quant tolerance)."""
        torch.manual_seed(7)
        in_f, out_f = 256, 128

        lin = nn.Linear(in_f, out_f, bias=False).to(cuda_device)
        fp8_lin = Fp8AmpereLinear.from_linear(lin).to(cuda_device)

        ext = fp8_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        x = torch.randn(32, in_f, dtype=torch.bfloat16, device=cuda_device)

        with torch.no_grad():
            ref = lin(x.to(torch.float32)).bfloat16()
            out = fp8_lin(x)

        assert out.shape == ref.shape
        scale = ref.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        rel_err = (ref - out).abs() / scale
        # FP8 E4M3 has 3 mantissa bits → ~6% relative error per element.
        # Allow 10% headroom for accumulated error over K dimension.
        assert rel_err.mean().item() < 0.10, (
            f"Marlin forward mean relative error: {rel_err.mean():.4f} (limit 0.10)"
        )

    @requires_cuda
    @requires_ampere
    def test_marlin_3d_input(self, cuda_device):
        lin = nn.Linear(128, 64, bias=True).to(cuda_device)
        fp8_lin = Fp8AmpereLinear.from_linear(lin).to(cuda_device)

        ext = fp8_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        x = torch.randn(4, 8, 128, dtype=torch.bfloat16, device=cuda_device)
        out = fp8_lin(x)
        assert out.shape == (4, 8, 64)

    @requires_cuda
    @requires_ampere
    def test_repack_and_compute_ops(self, cuda_device):
        """C++ repack_fp8_for_mma and compute_fp8_scales ops."""
        fp8_lin = Fp8AmpereLinear.from_linear(
            nn.Linear(128, 64, bias=False).to(cuda_device)
        ).to(cuda_device)
        ext = fp8_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        packed = ext.repack_fp8_for_mma(fp8_lin.weight)
        assert packed.dtype == torch.uint8
        # N=64 padded to 128, K=128 already aligned → flat [128*128]
        assert packed.shape == (128 * 128,)

        scales = ext.compute_fp8_scales(fp8_lin.weight)
        assert scales.shape == (64,)
        assert scales.dtype == torch.bfloat16
        assert torch.all(scales == 1.0)


# ---------------------------------------------------------------------------
# TPFp8AmpereLinear tests
# ---------------------------------------------------------------------------

from raylight.distributed_modules.fp8_ampere.tp_fp8_ampere_linear import (
    TPFp8AmpereLinear,
)


class TestTPFp8AmpereLinearConstruction:
    """Constructor and from_linear tests (no distributed runtime needed)."""

    @requires_cuda
    def test_from_linear_column(self, cuda_device):
        """Column-parallel: shard output dim N."""
        M, K, N = 32, 256, 128
        lin = nn.Linear(K, N, bias=True).to(cuda_device)
        # Simulate tp_size=2 by passing tp_group=None (falls back to get_tp_size=1 singleton)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column")
        # With tp_size=1, local_N == N
        assert tp_lin.weight_u8.shape == (N, K)
        assert tp_lin.weight_scale.shape == (N,)
        assert tp_lin.weight_u8.dtype == torch.uint8
        assert tp_lin.weight_scale.dtype == torch.bfloat16
        assert tp_lin.local_out_features == N
        assert tp_lin.local_in_features  == K

    @requires_cuda
    def test_from_linear_row(self, cuda_device):
        """Row-parallel: shard input dim K."""
        M, K, N = 32, 256, 128
        lin = nn.Linear(K, N, bias=False).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="row")
        assert tp_lin.weight_u8.shape == (N, K)
        assert tp_lin.weight_scale.shape == (N,)
        assert tp_lin.local_out_features == N
        assert tp_lin.local_in_features  == K
        assert tp_lin.bias is None

    @requires_cuda
    def test_from_fp8_ampere_linear(self, cuda_device):
        """from_linear should accept an Fp8AmpereLinear source."""
        K, N = 128, 64
        fp8_lin = Fp8AmpereLinear.from_linear(
            nn.Linear(K, N, bias=False).to(cuda_device)
        ).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(fp8_lin, parallelism="column")
        assert tp_lin.weight_u8.shape == fp8_lin.weight.shape
        assert tp_lin.weight_scale.shape == fp8_lin.weight_scale.shape

    @requires_cuda
    def test_from_fp8_checkpoint_weight(self, cuda_device):
        """Float8_e4m3fn weight accepted."""
        K, N = 64, 32
        w_fp8 = torch.randn(N, K, device=cuda_device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        lin = nn.Linear(K, N, bias=False)
        lin.weight = nn.Parameter(w_fp8.clone())
        lin.weight.data = w_fp8
        # Manually set in_features / out_features
        lin.in_features = K
        lin.out_features = N
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column")
        assert tp_lin.weight_u8.shape == (N, K)
        assert tp_lin.weight_u8.dtype == torch.uint8

    def test_comfy_cast_weights_true(self):
        """comfy_cast_weights must be True to enable LoRA patching."""
        assert TPFp8AmpereLinear.comfy_cast_weights is True

    @requires_cuda
    def test_extra_repr(self, cuda_device):
        lin = nn.Linear(64, 32, bias=True).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column")
        r = repr(tp_lin)
        assert "column" in r
        assert "bias=True" in r


class TestTPFp8AmpereLinearFallbackForward:
    """Fallback forward (CPU or no extension) tests."""

    @requires_cuda
    def test_output_shape_2d(self, cuda_device):
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        tp_lin._load_ext = lambda: None  # force fallback

        x = torch.randn(8, K, device=cuda_device, dtype=torch.bfloat16)
        out = tp_lin(x)
        assert out.shape == (8, N)
        assert out.dtype == torch.bfloat16

    @requires_cuda
    def test_output_shape_3d(self, cuda_device):
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=True).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        tp_lin._load_ext = lambda: None  # force fallback

        x = torch.randn(2, 8, K, device=cuda_device, dtype=torch.bfloat16)
        out = tp_lin(x)
        assert out.shape == (2, 8, N)

    @requires_cuda
    def test_accuracy_vs_reference(self, cuda_device):
        """Fallback output should be close to a BF16 reference (FP8 quant error only)."""
        torch.manual_seed(7)
        K, N = 256, 128
        lin = nn.Linear(K, N, bias=False).to(device=cuda_device, dtype=torch.bfloat16)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        tp_lin._load_ext = lambda: None  # fallback path

        x = torch.randn(16, K, device=cuda_device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = F.linear(x, lin.weight)
            out = tp_lin(x)

        assert out.shape == ref.shape
        # FP8 quantisation introduces ~2-3% mean relative error
        rel_err = (out - ref).abs().mean() / (ref.abs().mean() + 1e-6)
        assert rel_err < 0.035, f"fallback rel_err={rel_err:.4f}"


class TestTPFp8AmpereLinearMarlinForward:
    """Marlin kernel forward tests (Ampere only)."""

    @requires_cuda
    @requires_ampere
    def test_marlin_output_shape(self, cuda_device):
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        ext = tp_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        x = torch.randn(16, K, device=cuda_device, dtype=torch.bfloat16)
        out = tp_lin(x)
        assert out.shape == (16, N)
        assert out.dtype == torch.bfloat16

    @requires_cuda
    @requires_ampere
    def test_marlin_accuracy_vs_reference(self, cuda_device):
        """Marlin output should agree with BF16 reference within FP8 quant error."""
        torch.manual_seed(11)
        K, N = 256, 128
        lin = nn.Linear(K, N, bias=False).to(device=cuda_device, dtype=torch.bfloat16)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        ext = tp_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        x = torch.randn(32, K, device=cuda_device, dtype=torch.bfloat16)
        with torch.no_grad():
            ref = F.linear(x, lin.weight)
            out = tp_lin(x)

        assert out.shape == ref.shape
        rel_err = (out - ref).abs().mean() / (ref.abs().mean() + 1e-6)
        assert rel_err < 0.10, f"Marlin rel_err={rel_err:.4f}"

    @requires_cuda
    @requires_ampere
    def test_marlin_3d_input(self, cuda_device):
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(cuda_device)
        tp_lin = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        ext = tp_lin._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        x = torch.randn(2, 8, K, device=cuda_device, dtype=torch.bfloat16)
        out = tp_lin(x)
        assert out.shape == (2, 8, N)


# ---------------------------------------------------------------------------
# from_tp_linear and weight-freeing tests
# ---------------------------------------------------------------------------

from raylight.distributed_modules.tensor_parallel import TPLinear


class TestTPFp8AmpereLinearFromTPLinear:
    """from_tp_linear factory: builds from a TPLinear with FP8 shard weight."""

    @requires_cuda
    def test_basic_column(self, cuda_device):
        """from_tp_linear col-parallel: adopts sharded weight without re-sharding."""
        K, N = 256, 128
        w_fp8 = torch.randn(N, K, device=cuda_device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        tp_lin = TPLinear(K, N, bias=False, parallelism="column", device=cuda_device,
                          dtype=torch.bfloat16)
        tp_lin.weight.data = w_fp8  # assign fp8 shard (as streaming loader would)

        fp8_tp = TPFp8AmpereLinear.from_tp_linear(tp_lin)
        assert fp8_tp.weight_u8.shape == (N, K)
        assert fp8_tp.weight_u8.dtype == torch.uint8
        assert fp8_tp.weight_scale.shape == (N,)
        assert fp8_tp.weight_scale.dtype == torch.bfloat16
        assert fp8_tp.parallelism == "column"
        assert fp8_tp.in_features == K
        assert fp8_tp.out_features == N

    @requires_cuda
    def test_basic_row(self, cuda_device):
        """from_tp_linear row-parallel."""
        K, N = 256, 128
        w_fp8 = torch.randn(N, K, device=cuda_device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        tp_lin = TPLinear(K, N, bias=False, parallelism="row", device=cuda_device,
                          dtype=torch.bfloat16)
        tp_lin.weight.data = w_fp8

        fp8_tp = TPFp8AmpereLinear.from_tp_linear(tp_lin)
        assert fp8_tp.weight_u8.shape == (N, K)
        assert fp8_tp.parallelism == "row"

    @requires_cuda
    def test_bias_copied(self, cuda_device):
        """Bias is copied from source TPLinear."""
        K, N = 128, 64
        w_fp8 = torch.randn(N, K, device=cuda_device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        tp_lin = TPLinear(K, N, bias=True, parallelism="column", device=cuda_device,
                          dtype=torch.bfloat16)
        tp_lin.weight.data = w_fp8
        tp_lin.bias.data.fill_(1.5)

        fp8_tp = TPFp8AmpereLinear.from_tp_linear(tp_lin)
        assert fp8_tp.bias is not None
        assert fp8_tp.bias.shape == (N,)
        assert fp8_tp.bias.abs().mean().item() > 1.0

    @requires_cuda
    def test_checkpoint_scale_preserved(self, cuda_device):
        """Checkpoint weight_scale on TPLinear is used instead of all-ones fallback."""
        K, N = 128, 64
        w_fp8 = torch.randn(N, K, device=cuda_device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        tp_lin = TPLinear(K, N, bias=False, parallelism="column", device=cuda_device,
                          dtype=torch.bfloat16)
        tp_lin.weight.data = w_fp8
        # Simulate per-channel scales loaded from a checkpoint
        checkpoint_scale = torch.rand(N, device=cuda_device, dtype=torch.bfloat16) + 0.5
        tp_lin.weight_scale = checkpoint_scale

        fp8_tp = TPFp8AmpereLinear.from_tp_linear(tp_lin)
        assert torch.allclose(fp8_tp.weight_scale, checkpoint_scale.to(torch.bfloat16))

    @requires_cuda
    def test_no_checkpoint_scale_falls_back_to_ones(self, cuda_device):
        """Without a checkpoint scale, weight_scale defaults to all-ones."""
        K, N = 128, 64
        w_fp8 = torch.randn(N, K, device=cuda_device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        tp_lin = TPLinear(K, N, bias=False, parallelism="column", device=cuda_device,
                          dtype=torch.bfloat16)
        tp_lin.weight.data = w_fp8
        # No weight_scale attribute set

        fp8_tp = TPFp8AmpereLinear.from_tp_linear(tp_lin)
        assert torch.all(fp8_tp.weight_scale == 1.0)

    @requires_cuda
    def test_wrong_dtype_raises(self, cuda_device):
        """Raises TypeError when weight is not FP8."""
        K, N = 64, 32
        tp_lin = TPLinear(K, N, bias=False, parallelism="column", device=cuda_device)
        with pytest.raises(TypeError, match="float8_e4m3fn or uint8"):
            TPFp8AmpereLinear.from_tp_linear(tp_lin)

    @requires_cuda
    def test_wrong_type_raises(self, cuda_device):
        """Raises TypeError when passed something other than TPLinear."""
        lin = nn.Linear(64, 32, device=cuda_device)
        with pytest.raises(TypeError, match="TPLinear"):
            TPFp8AmpereLinear.from_tp_linear(lin)


class TestWeightFreeing:
    """Verify raw FP8 buffers are freed after _prepare_packed."""

    @requires_cuda
    @requires_ampere
    def test_fp8_ampere_linear_frees_weight(self, cuda_device):
        """Fp8AmpereLinear.weight_packed should be populated after first Marlin forward.

        The raw FP8 weight is intentionally retained (not freed) so that the
        small-batch cuBLAS fallback path remains available.  Only weight_packed
        is checked here.
        """
        K, N = 256, 128
        lin = nn.Linear(K, N, bias=False).to(device=cuda_device, dtype=torch.bfloat16)
        fp8 = Fp8AmpereLinear.from_linear(lin).to(cuda_device)
        ext = fp8._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        assert fp8.weight.numel() == N * K, "weight should be full before first forward"

        # Call _prepare_packed directly — the forward threshold may route small
        # batches to the cuBLAS fallback, which never triggers packing.
        fp8._prepare_packed(cuda_device)

        assert fp8.weight_packed is not None, "weight_packed should be populated after _prepare_packed"
        assert fp8.weight_packed.numel() > 0

    @requires_cuda
    @requires_ampere
    def test_tp_fp8_ampere_linear_frees_weight_u8(self, cuda_device):
        """TPFp8AmpereLinear.weight_u8 should be 0-element after first Marlin forward."""
        K, N = 256, 128
        lin = nn.Linear(K, N, bias=False).to(device=cuda_device, dtype=torch.bfloat16)
        tp_fp8 = TPFp8AmpereLinear.from_linear(lin, parallelism="column").to(cuda_device)
        ext = tp_fp8._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        assert tp_fp8.weight_u8.numel() == N * K

        x = torch.randn(8, K, device=cuda_device, dtype=torch.bfloat16)
        _ = tp_fp8(x)

        assert tp_fp8.weight_u8.numel() == 0, "weight_u8 should be freed after pack"
        assert tp_fp8.weight_packed is not None

    @requires_cuda
    @requires_ampere
    def test_prepare_packed_is_idempotent(self, cuda_device):
        """Calling _prepare_packed twice on the same device is a no-op."""
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(device=cuda_device, dtype=torch.bfloat16)
        fp8 = Fp8AmpereLinear.from_linear(lin).to(cuda_device)
        ext = fp8._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        fp8._prepare_packed(cuda_device)
        packed_id = id(fp8.weight_packed)
        fp8._prepare_packed(cuda_device)  # should not re-allocate
        assert id(fp8.weight_packed) == packed_id


# ---------------------------------------------------------------------------
# Fp8KernelMixin.on_device_activated
# ---------------------------------------------------------------------------


class TestFp8KernelMixinOnDeviceActivated:
    """Verify on_device_activated migrates packed buffers to the target device."""

    @requires_cuda
    @requires_ampere
    def test_migrates_weight_packed_to_cuda(self, cuda_device):
        """on_device_activated moves weight_packed from CPU to CUDA."""
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(dtype=torch.bfloat16)
        fp8 = Fp8AmpereLinear.from_linear(lin)  # CPU
        ext = fp8._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        fp8._prepare_packed(torch.device("cpu"))
        assert fp8.weight_packed.device.type == "cpu"
        assert fp8.scale_padded.device.type == "cpu"

        fp8.on_device_activated(cuda_device)

        assert fp8.weight_packed.device.type == "cuda"
        assert fp8.scale_padded.device.type == "cuda"

    @requires_cuda
    @requires_ampere
    def test_noop_when_already_on_device(self, cuda_device):
        """on_device_activated is a no-op when buffers are already on the target device."""
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(device=cuda_device, dtype=torch.bfloat16)
        fp8 = Fp8AmpereLinear.from_linear(lin).to(cuda_device)
        ext = fp8._load_ext()
        if ext is None:
            pytest.skip("_C_fp8ampere extension not built")

        fp8._prepare_packed(cuda_device)
        packed_id = id(fp8.weight_packed)

        fp8.on_device_activated(cuda_device)  # already on cuda — should not reallocate

        assert id(fp8.weight_packed) == packed_id

    def test_noop_when_not_packed(self):
        """on_device_activated before _prepare_packed is a safe no-op."""
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(dtype=torch.bfloat16)
        fp8 = Fp8AmpereLinear.from_linear(lin)

        # No _prepare_packed called — _packed is False
        fp8.on_device_activated(torch.device("cpu"))  # must not raise

    def test_isinstance_check(self):
        """Both Fp8AmpereLinear and TPFp8AmpereLinear are Fp8KernelMixin instances."""
        from raylight.distributed_modules.fp8_ampere.tp_fp8_ampere_linear import TPFp8AmpereLinear
        K, N = 128, 64
        lin = nn.Linear(K, N, bias=False).to(dtype=torch.bfloat16)
        fp8 = Fp8AmpereLinear.from_linear(lin)
        tp_fp8 = TPFp8AmpereLinear.from_linear(lin, parallelism="column")
        assert isinstance(fp8, Fp8KernelMixin)
        assert isinstance(tp_fp8, Fp8KernelMixin)


# ---------------------------------------------------------------------------
# ModelContext._apply_fp8_swap and _activate_fp8_buffers
# ---------------------------------------------------------------------------


class _FakeLoadConfig:
    """Minimal stand-in for ModelState."""
    def __init__(self, model_options):
        self.model_options = model_options


class _FakeModel:
    """Minimal stand-in for a ComfyUI ModelPatcher."""
    def __init__(self, inner):
        self.model = inner
        self.load_config = _FakeLoadConfig({})


class _FakeInner(nn.Module):
    """A tiny model with float8 linear weights for swap testing."""
    def __init__(self, K=128, N=64):
        super().__init__()
        w = torch.randn(N, K).to(torch.float8_e4m3fn)
        self.linear = nn.Linear(K, N, bias=False)
        self.linear.weight = nn.Parameter(w, requires_grad=False)


class TestApplyFp8Swap:
    """Tests for ModelContext._apply_fp8_swap."""

    @requires_cuda
    @requires_ampere
    def test_swaps_linear_when_flag_set(self, cuda_device):
        """_apply_fp8_swap replaces float8 nn.Linear with Fp8AmpereLinear."""
        from raylight.distributed_actor.model_context._base import ModelContext

        inner = _FakeInner()
        model = _FakeModel(inner)
        model.load_config.model_options = {"fp8_ampere": True}

        ModelContext._apply_fp8_swap(model, cuda_device)

        assert isinstance(inner.linear, Fp8AmpereLinear), (
            "nn.Linear with float8 weight should be swapped to Fp8AmpereLinear"
        )

    @requires_cuda
    @requires_ampere
    def test_no_swap_when_flag_absent(self, cuda_device):
        """_apply_fp8_swap is a no-op when fp8_ampere flag is not set."""
        from raylight.distributed_actor.model_context._base import ModelContext

        inner = _FakeInner()
        model = _FakeModel(inner)
        # model_options has no "fp8_ampere" key

        ModelContext._apply_fp8_swap(model, cuda_device)

        assert isinstance(inner.linear, nn.Linear) and not isinstance(inner.linear, Fp8AmpereLinear)

    @requires_cuda
    @requires_ampere
    def test_idempotent(self, cuda_device):
        """Calling _apply_fp8_swap twice does not double-swap."""
        from raylight.distributed_actor.model_context._base import ModelContext

        inner = _FakeInner()
        model = _FakeModel(inner)
        model.load_config.model_options = {"fp8_ampere": True}

        ModelContext._apply_fp8_swap(model, cuda_device)
        first_mod = inner.linear
        ModelContext._apply_fp8_swap(model, cuda_device)  # second call — no-op

        assert inner.linear is first_mod, "_fp8_swap_done guard should prevent re-swap"

    @requires_cuda
    @requires_ampere
    def test_packed_buffers_on_cpu_after_swap(self, cuda_device):
        """After _apply_fp8_swap, weight_packed is on CPU (weights not yet moved to CUDA)."""
        from raylight.distributed_actor.model_context._base import ModelContext

        inner = _FakeInner()
        model = _FakeModel(inner)
        model.load_config.model_options = {"fp8_ampere": True}

        ModelContext._apply_fp8_swap(model, cuda_device)

        fp8 = inner.linear
        # Swap packs on source device (CPU), cache.build() will capture them there.
        assert fp8._packed, "weight should be packed after swap"
        assert fp8.weight_packed.device.type == "cpu"


class TestActivateFp8Buffers:
    """Tests for ModelContext._activate_fp8_buffers."""

    @requires_cuda
    @requires_ampere
    def test_migrates_all_mixin_instances(self, cuda_device):
        """_activate_fp8_buffers calls on_device_activated on every Fp8KernelMixin."""
        from raylight.distributed_actor.model_context._base import ModelContext

        inner = _FakeInner()
        model = _FakeModel(inner)
        model.load_config.model_options = {"fp8_ampere": True}

        # Swap and pack on CPU
        ModelContext._apply_fp8_swap(model, cuda_device)
        assert inner.linear.weight_packed.device.type == "cpu"

        # Now activate — should move packed buffers to CUDA
        ModelContext._activate_fp8_buffers(model, cuda_device)

        assert inner.linear.weight_packed.device.type == "cuda"
        assert inner.linear.scale_padded.device.type == "cuda"

    def test_noop_on_model_with_no_fp8(self):
        """_activate_fp8_buffers is safe on models with no Fp8KernelMixin layers."""
        from raylight.distributed_actor.model_context._base import ModelContext

        class _PlainInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)

        inner = _PlainInner()
        model = _FakeModel(inner)

        # Must not raise
        ModelContext._activate_fp8_buffers(model, torch.device("cpu"))
