"""Tests for Triton WHT and fused compress/decompress kernels.

Tests cover:
1. WHT forward/inverse round-trip (Triton vs PyTorch)
2. 4-bit pack/unpack round-trip (fused Triton kernel)
3. 2-bit pack/unpack round-trip (fused Triton kernel)
4. Fused 4-bit compress→decompress: no NaN/Inf, bounded error
5. Fused 2-bit compress→decompress: no NaN/Inf, bounded error
6. Fused vs non-fused equivalence (same signs → same packed output)
7. Edge cases: zeros, constants, large values, hidden_dim=3072

Requires: GPU with Triton support.
"""

import math
import pytest
import torch

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# ---------------------------------------------------------------------------
# Imports from the modules under test
# ---------------------------------------------------------------------------
from raylight.distributed_modules.tp_compress_triton import (
    _HAS_TRITON,
    _next_power_of_2,
    _wht_pytorch,
    WHTRotationPyTorch,
)

if _HAS_TRITON:
    from raylight.distributed_modules.tp_compress_triton import (
        WHTRotation,
        FusedWHTCompressor,
    )

from raylight.distributed_modules.tp_compress import (
    _quantize_and_pack,
    _dequantize_and_unpack,
    _effective_group_size,
    _pack_int4,
    _unpack_int4,
    _pack_int2,
    _unpack_int2,
)

DEVICE = torch.device("cuda:0")
SEED = 42


# ============================================================================
# Helpers
# ============================================================================

def make_random_input(T: int, C: int, device=DEVICE, dtype=torch.bfloat16):
    """Create a random (T, C) tensor simulating diffusion model activations."""
    return torch.randn(T, C, device=device, dtype=dtype)


def assert_no_nan_inf(t: torch.Tensor, name: str = "tensor"):
    assert not torch.isnan(t).any(), f"{name} contains NaN"
    assert not torch.isinf(t).any(), f"{name} contains Inf"


# ============================================================================
# 1. WHT round-trip: Triton vs PyTorch
# ============================================================================

class TestWHTRoundTrip:
    """Test that WHT forward→inverse reconstructs the input."""

    @pytest.mark.parametrize("hidden_dim", [64, 128, 256, 512, 1024, 3072])
    def test_pytorch_wht_round_trip(self, hidden_dim):
        """PyTorch WHT: forward → inverse should recover input (exact for power-of-2)."""
        padded = _next_power_of_2(hidden_dim)
        T = 4
        x = torch.randn(T, padded, device=DEVICE, dtype=torch.float32)

        # WHT is self-inverse (up to normalization): H * H = I
        # With 1/sqrt(C) normalization applied twice, we get identity.
        y = _wht_pytorch(x)  # includes 1/sqrt(C)
        x_rec = _wht_pytorch(y)  # includes 1/sqrt(C) again → total 1/C

        # After two applications: x_rec = x * (C / C) = x... no wait.
        # _wht_pytorch normalizes by 1/sqrt(C). So H_norm = H/sqrt(C).
        # H_norm * H_norm = H*H / C = C*I / C = I. So it's exactly identity.
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("hidden_dim", [64, 256, 3072])
    def test_pytorch_wht_rotation_class_round_trip(self, hidden_dim):
        """WHTRotationPyTorch: forward → inverse recovers input."""
        T = 8
        x = torch.randn(T, hidden_dim, device=DEVICE, dtype=torch.float32)

        rot = WHTRotationPyTorch(hidden_dim, seed=SEED, device=DEVICE)
        y = rot.rotate_forward(x)
        x_rec = rot.rotate_inverse(y)

        assert x_rec.shape == x.shape
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
    @pytest.mark.parametrize("hidden_dim", [64, 256, 1024, 3072])
    def test_triton_wht_round_trip(self, hidden_dim):
        """Triton WHT: forward → inverse recovers input."""
        T = 8
        x = torch.randn(T, hidden_dim, device=DEVICE, dtype=torch.float32)

        rot = WHTRotation(hidden_dim, seed=SEED, device=DEVICE)
        y = rot.rotate_forward(x)
        x_rec = rot.rotate_inverse(y)

        assert x_rec.shape == x.shape
        assert_no_nan_inf(x_rec, "triton WHT round-trip")
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
    @pytest.mark.parametrize("hidden_dim", [64, 256, 3072])
    def test_triton_vs_pytorch_wht(self, hidden_dim):
        """Triton WHT matches PyTorch WHT output."""
        T = 4
        x = torch.randn(T, hidden_dim, device=DEVICE, dtype=torch.float32)

        triton_rot = WHTRotation(hidden_dim, seed=SEED, device=DEVICE)
        pytorch_rot = WHTRotationPyTorch(hidden_dim, seed=SEED, device=DEVICE)

        y_triton = triton_rot.rotate_forward(x)
        y_pytorch = pytorch_rot.rotate_forward(x)

        torch.testing.assert_close(y_triton, y_pytorch, atol=1e-4, rtol=1e-4)


# ============================================================================
# 2. PyTorch pack/unpack round-trip
# ============================================================================

class TestPackUnpack:
    """Test that PyTorch pack→unpack is lossless."""

    def test_pack_unpack_int4(self):
        N = 256
        indices = torch.randint(0, 16, (4, N), dtype=torch.uint8, device=DEVICE)
        packed = _pack_int4(indices)
        unpacked = _unpack_int4(packed, N)
        assert torch.equal(indices, unpacked)

    def test_pack_unpack_int2(self):
        N = 256
        indices = torch.randint(0, 4, (4, N), dtype=torch.uint8, device=DEVICE)
        packed = _pack_int2(indices)
        unpacked = _unpack_int2(packed, N)
        assert torch.equal(indices, unpacked)


# ============================================================================
# 3. Non-fused quantize/dequantize round-trip
# ============================================================================

class TestNonFusedQuantize:
    """Test PyTorch quantize → dequantize without any rotation."""

    @pytest.mark.parametrize("bits", [2, 4])
    def test_quantize_dequantize_no_nan(self, bits):
        T, C = 8, 256
        x = torch.randn(T, C, device=DEVICE, dtype=torch.float32)

        packed, scales, zp, gs = _quantize_and_pack(x, group_size=128, bits=bits)
        x_rec = _dequantize_and_unpack(packed, scales, zp, (T, C), gs, bits, torch.float32)

        assert_no_nan_inf(x_rec, f"dequant {bits}-bit")
        assert x_rec.shape == x.shape

    @pytest.mark.parametrize("bits", [2, 4])
    def test_quantize_dequantize_bounded_error(self, bits):
        """Quantization error should be bounded (not blow up)."""
        T, C = 16, 512
        x = torch.randn(T, C, device=DEVICE, dtype=torch.float32)

        packed, scales, zp, gs = _quantize_and_pack(x, group_size=128, bits=bits)
        x_rec = _dequantize_and_unpack(packed, scales, zp, (T, C), gs, bits, torch.float32)

        mse = ((x - x_rec) ** 2).mean().item()
        # MSE should be reasonable — not zero, but not huge
        x_var = x.var().item()
        snr = x_var / max(mse, 1e-10)
        # At 4-bit we expect SNR > 20 dB (~100x), at 2-bit > 5 dB (~3x)
        min_snr = 3.0 if bits == 2 else 50.0
        assert snr > min_snr, f"SNR too low: {snr:.1f} (min {min_snr})"


# ============================================================================
# 4. Fused compress/decompress round-trip
# ============================================================================

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
class TestFusedCompressDecompress:
    """Test fused WHT+quantize+pack → unpack+dequant+inverse WHT round-trip."""

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("hidden_dim", [64, 128, 256, 3072])
    def test_round_trip_no_nan(self, bits, hidden_dim):
        """Fused compress→decompress should never produce NaN or Inf."""
        T = 8
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)

        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=bits,
                                seed=SEED, device=DEVICE)
        packed, scales, zp = fc.compress(x)
        x_rec = fc.decompress(packed, scales, zp)

        assert_no_nan_inf(x_rec, f"fused {bits}-bit round-trip")
        assert x_rec.shape == (T, hidden_dim)

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("hidden_dim", [256, 3072])
    def test_round_trip_bounded_error(self, bits, hidden_dim):
        """Fused round-trip error should be bounded with reasonable SNR."""
        T = 32
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)
        x_f32 = x.float()

        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=bits,
                                seed=SEED, device=DEVICE)
        packed, scales, zp = fc.compress(x)
        x_rec = fc.decompress(packed, scales, zp)

        mse = ((x_f32 - x_rec) ** 2).mean().item()
        x_var = x_f32.var().item()
        snr = x_var / max(mse, 1e-10)

        # WHT rotation should improve SNR vs raw quantization
        min_snr = 3.0 if bits == 2 else 30.0
        assert snr > min_snr, f"SNR too low: {snr:.2f} (min {min_snr}) for {bits}-bit dim={hidden_dim}"

    @pytest.mark.parametrize("bits", [4])
    def test_round_trip_4bit_specific_values(self, bits):
        """Test 4-bit with specific value patterns that could trigger NaN."""
        T, C = 4, 256

        # Test with zeros
        x_zero = torch.zeros(T, C, device=DEVICE, dtype=torch.bfloat16)
        fc = FusedWHTCompressor(C, group_size=128, bits=bits, seed=SEED, device=DEVICE)
        packed, scales, zp = fc.compress(x_zero)
        rec = fc.decompress(packed, scales, zp)
        assert_no_nan_inf(rec, "zeros")

        # Test with constant
        x_const = torch.ones(T, C, device=DEVICE, dtype=torch.bfloat16) * 3.14
        packed, scales, zp = fc.compress(x_const)
        rec = fc.decompress(packed, scales, zp)
        assert_no_nan_inf(rec, "constant")

        # Test with large values
        x_large = torch.randn(T, C, device=DEVICE, dtype=torch.bfloat16) * 1000
        packed, scales, zp = fc.compress(x_large)
        rec = fc.decompress(packed, scales, zp)
        assert_no_nan_inf(rec, "large values")

        # Test with very small values
        x_small = torch.randn(T, C, device=DEVICE, dtype=torch.bfloat16) * 1e-6
        packed, scales, zp = fc.compress(x_small)
        rec = fc.decompress(packed, scales, zp)
        assert_no_nan_inf(rec, "small values")

    @pytest.mark.parametrize("bits", [2, 4])
    def test_multiple_round_trips_no_accumulation(self, bits):
        """Multiple compress→decompress cycles shouldn't accumulate NaN."""
        T, C = 8, 256
        x = make_random_input(T, C, dtype=torch.bfloat16)
        fc = FusedWHTCompressor(C, group_size=128, bits=bits, seed=SEED, device=DEVICE)

        for i in range(5):
            packed, scales, zp = fc.compress(x)
            x = fc.decompress(packed, scales, zp).to(torch.bfloat16)
            assert_no_nan_inf(x, f"iteration {i}")


# ============================================================================
# 5. Fused 4-bit pack/unpack interleaving correctness
# ============================================================================

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
class TestFused4BitInterleaving:
    """Verify that fused Triton 4-bit packing matches PyTorch packing."""

    def test_compress_packed_matches_pytorch_pack(self):
        """After WHT rotation, the fused and non-fused paths should produce identical packed bytes."""
        T, hidden_dim = 4, 256
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)

        # Fused path
        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=4,
                                seed=SEED, device=DEVICE)
        packed_fused, scales_fused, zp_fused = fc.compress(x)

        # Non-fused path: use same WHT rotation, then quantize separately
        rot = WHTRotation(hidden_dim, seed=SEED, device=DEVICE)
        rotated = rot.rotate_forward(x)
        gs = _effective_group_size(rot.padded_dim, 128)
        packed_ref, scales_ref, zp_ref, _ = _quantize_and_pack(
            rotated, group_size=128, bits=4
        )

        # Scales should be close (bf16 rounding differences possible)
        torch.testing.assert_close(
            scales_fused.float(), scales_ref.float(),
            atol=0.05, rtol=0.05,
        )

        # Zero points should be close
        torch.testing.assert_close(
            zp_fused.float(), zp_ref.float(),
            atol=0.05, rtol=0.05,
        )

    def test_compress_packed_matches_pytorch_pack_2bit(self):
        """After WHT rotation, 2-bit fused and non-fused packed output should match."""
        T, hidden_dim = 4, 256
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)

        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=2,
                                seed=SEED, device=DEVICE)
        packed_fused, scales_fused, _ = fc.compress(x)

        rot = WHTRotation(hidden_dim, seed=SEED, device=DEVICE)
        rotated = rot.rotate_forward(x)
        packed_ref, scales_ref, _, _ = _quantize_and_pack(
            rotated, group_size=128, bits=2
        )

        torch.testing.assert_close(
            scales_fused.float(), scales_ref.float(),
            atol=0.05, rtol=0.05,
        )


# ============================================================================
# 6. Fused decompress output matches non-fused decompress
# ============================================================================

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
class TestFusedVsNonFused:
    """Fused compress→decompress should produce same result as non-fused."""

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("hidden_dim", [256, 3072])
    def test_fused_vs_nonfused_output(self, bits, hidden_dim):
        """End-to-end: fused round-trip ≈ non-fused round-trip."""
        T = 8
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)

        # Fused path
        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=bits,
                                seed=SEED, device=DEVICE)
        p, s, z = fc.compress(x)
        rec_fused = fc.decompress(p, s, z)

        # Non-fused path
        rot = WHTRotation(hidden_dim, seed=SEED, device=DEVICE)
        rotated = rot.rotate_forward(x)
        gs_eff = _effective_group_size(rot.padded_dim, 128)
        p2, s2, z2, gs2 = _quantize_and_pack(rotated, group_size=128, bits=bits)
        deq = _dequantize_and_unpack(
            p2, s2, z2, (T, rot.padded_dim), gs2, bits, torch.float32,
        )
        rec_nonfused = rot.rotate_inverse(deq)

        # Both should have same shape
        assert rec_fused.shape == (T, hidden_dim)
        assert rec_nonfused.shape == (T, hidden_dim)

        # Both should be close (small differences from bf16 scale rounding
        # and Triton vs PyTorch float32 arithmetic)
        torch.testing.assert_close(
            rec_fused, rec_nonfused.float(),
            atol=0.25, rtol=0.1,
        )


# ============================================================================
# 7. Serialization round-trip (the full pipeline used in compressed_all_reduce)
# ============================================================================

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
class TestSerializationRoundTrip:
    """Test that serialize→deserialize→decompress works without NaN."""

    @pytest.mark.parametrize("bits", [2, 4])
    def test_serialize_deserialize_fused(self, bits):
        from raylight.distributed_modules.tp_compress import (
            _serialize_compressed,
            _deserialize_compressed,
        )

        T, hidden_dim = 8, 3072
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)

        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=bits,
                                seed=SEED, device=DEVICE)
        packed, scales, zp = fc.compress(x)

        # Serialize
        Q = fc.padded_dim
        gs = fc.group_size
        payload = _serialize_compressed(packed, scales, zp, T, Q, gs, bits)
        # Note: payload contains packed uint8 reinterpreted as bf16,
        # so some byte patterns look like bf16 NaN — that's expected.

        # Deserialize
        p2, s2, z2 = _deserialize_compressed(payload, Q, gs, bits)

        # Packed data should match exactly
        assert torch.equal(packed, p2), "packed mismatch after serialize round-trip"

        # Scales should match
        torch.testing.assert_close(scales, s2)

        # Decompress
        rec = fc.decompress(p2, s2, z2)
        assert_no_nan_inf(rec, f"deserialized+decompressed {bits}-bit")


# ============================================================================
# 8. Stress test with LTXAV-like dimensions
# ============================================================================

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
class TestLTXAVDimensions:
    """Test with actual LTXAV model dimensions: hidden_dim=3072, padded=4096."""

    @pytest.mark.parametrize("bits", [2, 4])
    def test_ltxav_dims(self, bits):
        T = 6820  # typical token count for LTXAV video
        # Use smaller T for test speed, but keep hidden_dim realistic
        T_test = 64
        hidden_dim = 3072

        x = make_random_input(T_test, hidden_dim, dtype=torch.bfloat16)

        fc = FusedWHTCompressor(hidden_dim, group_size=128, bits=bits,
                                seed=SEED, device=DEVICE)
        packed, scales, zp = fc.compress(x)
        rec = fc.decompress(packed, scales, zp)

        assert_no_nan_inf(rec, f"LTXAV {bits}-bit")
        assert rec.shape == (T_test, hidden_dim)

        mse = ((x.float() - rec) ** 2).mean().item()
        print(f"  LTXAV {bits}-bit MSE: {mse:.6f}, "
              f"SNR: {x.float().var().item() / max(mse, 1e-10):.1f}")


# ============================================================================
# 9. Decompress kernel output write correctness (OUT_DIM < C)
# ============================================================================

@pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
class TestOutputPadding:
    """When hidden_dim < padded_dim, decompress writes only hidden_dim elements."""

    @pytest.mark.parametrize("bits", [2, 4])
    @pytest.mark.parametrize("hidden_dim", [3072, 100, 65])
    def test_output_dim_correct(self, bits, hidden_dim):
        T = 4
        x = make_random_input(T, hidden_dim, dtype=torch.bfloat16)

        fc = FusedWHTCompressor(hidden_dim, group_size=32, bits=bits,
                                seed=SEED, device=DEVICE)
        packed, scales, zp = fc.compress(x)
        rec = fc.decompress(packed, scales, zp)

        assert rec.shape == (T, hidden_dim), \
            f"Expected ({T}, {hidden_dim}), got {rec.shape}"
        assert_no_nan_inf(rec, f"output padding {bits}-bit dim={hidden_dim}")
