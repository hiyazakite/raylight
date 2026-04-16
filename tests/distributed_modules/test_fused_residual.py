"""Standalone test for fused WHT residual (base subtract/add inside kernel).

Run with:
    export PYTHONPATH=$PYTHONPATH:/root/ComfyUI/custom_nodes/raylight/src
    export CUDA_VISIBLE_DEVICES=0
    python3 tests/test_fused_residual.py
"""

import sys
import torch
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42


def make_random_input(T, C, device=DEVICE, dtype=torch.bfloat16):
    return torch.randn(T, C, device=device, dtype=dtype)


def assert_no_nan_inf(t, name="tensor"):
    assert not torch.isnan(t).any(), f"{name} contains NaN"
    assert not torch.isinf(t).any(), f"{name} contains Inf"


@pytest.mark.parametrize("hidden_dim", [256, 3072])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_fused_compress_matches_explicit(bits, hidden_dim):
    """Fused compress(x, base) should produce equivalent output to compress(x - base).

    Note: packed bytes may differ slightly because fused subtracts in fp32
    while explicit subtracts in bf16 before padding. We verify the
    decompressed outputs are close instead.
    """
    from raylight.distributed_modules.tp_compress_triton import FusedWHTCompressor

    T = 8
    x = make_random_input(T, hidden_dim)
    base = make_random_input(T, hidden_dim) * 0.1

    fc = FusedWHTCompressor(hidden_dim, group_size=64, bits=bits, seed=SEED, device=DEVICE)

    # Fused path: kernel subtracts base in-register (fp32)
    packed_fused, scales_fused, zp_fused = fc.compress(x, base=base)
    rec_fused = fc.decompress(packed_fused, scales_fused, zp_fused)

    # Explicit path: subtract base in bf16, then compress
    delta = x - base
    packed_explicit, scales_explicit, zp_explicit = fc.compress(delta)
    rec_explicit = fc.decompress(packed_explicit, scales_explicit, zp_explicit)

    # Decompressed deltas should be very close (fused is slightly more precise
    # due to fp32 subtraction, but quantization dominates the error).
    # At 2-bit, boundary rounding differences from bf16→fp32 can cause ~0.3%
    # of elements to land in different centroids.
    atol = 0.3 if bits <= 3 else 0.15
    torch.testing.assert_close(rec_fused, rec_explicit, atol=atol, rtol=0.15)

    print(f"  PASS: compress matches explicit ({bits}-bit, dim={hidden_dim})")


@pytest.mark.parametrize("hidden_dim", [256, 3072])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_fused_decompress_matches_explicit(bits, hidden_dim):
    """Fused decompress(packed, base) should equal decompress(packed) + base."""
    from raylight.distributed_modules.tp_compress_triton import FusedWHTCompressor

    T = 8
    x = make_random_input(T, hidden_dim)
    base = make_random_input(T, hidden_dim, dtype=torch.float32) * 0.1

    fc = FusedWHTCompressor(hidden_dim, group_size=64, bits=bits, seed=SEED, device=DEVICE)
    packed, scales, zp = fc.compress(x)

    # Fused path: kernel adds base in-register
    rec_fused = fc.decompress(packed, scales, zp, base=base)

    # Explicit path: decompress then add base.
    # decompress(no_base) returns bf16, promoting to fp32 on add.
    # The bf16 truncation introduces ~0.01 error vs the fused path
    # which computes entirely in fp32.
    rec_explicit = fc.decompress(packed, scales, zp).float() + base

    torch.testing.assert_close(rec_fused.float(), rec_explicit.float(), atol=0.02, rtol=0.01)
    print(f"  PASS: decompress matches explicit ({bits}-bit, dim={hidden_dim})")


@pytest.mark.parametrize("hidden_dim", [256, 3072])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_fused_residual_round_trip(bits, hidden_dim):
    """Full fused round-trip: compress(x, base) -> decompress(packed, base)."""
    from raylight.distributed_modules.tp_compress_triton import FusedWHTCompressor

    T = 16
    x = make_random_input(T, hidden_dim)
    base = x + torch.randn_like(x) * 0.05  # close to x (simulates previous step)

    fc = FusedWHTCompressor(hidden_dim, group_size=64, bits=bits, seed=SEED, device=DEVICE)

    packed, scales, zp = fc.compress(x, base=base)
    rec = fc.decompress(packed, scales, zp, base=base)

    assert_no_nan_inf(rec, f"fused residual round-trip {bits}-bit")
    assert rec.shape == (T, hidden_dim)

    mse = ((x.float() - rec) ** 2).mean().item()
    x_var = x.float().var().item()
    snr = x_var / max(mse, 1e-10)
    min_snr = 3.0 if bits == 2 else 30.0
    assert snr > min_snr, f"SNR too low: {snr:.1f} (min {min_snr})"

    print(f"  PASS: round-trip SNR={snr:.1f} ({bits}-bit, dim={hidden_dim})")


@pytest.mark.parametrize("bits", [2, 4])
def test_no_base_unchanged(bits):
    """compress/decompress without base should still work identically."""
    from raylight.distributed_modules.tp_compress_triton import FusedWHTCompressor

    T, hidden_dim = 8, 256
    x = make_random_input(T, hidden_dim)

    fc = FusedWHTCompressor(hidden_dim, group_size=64, bits=bits, seed=SEED, device=DEVICE)

    packed, scales, zp = fc.compress(x)
    rec = fc.decompress(packed, scales, zp)

    packed2, scales2, zp2 = fc.compress(x, base=None)
    rec2 = fc.decompress(packed2, scales2, zp2, base=None)

    assert torch.equal(packed, packed2)
    torch.testing.assert_close(rec, rec2)
    print(f"  PASS: no-base unchanged ({bits}-bit)")


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    try:
        from raylight.distributed_modules.tp_compress_triton import _HAS_TRITON
        if not _HAS_TRITON:
            print("SKIP: Triton not available")
            return
    except ImportError:
        print("SKIP: tp_compress_triton not importable")
        return

    passed = 0
    failed = 0

    for bits in [2, 3, 4]:
        for hidden_dim in [256, 3072]:
            for test_fn in [
                test_fused_compress_matches_explicit,
                test_fused_decompress_matches_explicit,
                test_fused_residual_round_trip,
            ]:
                try:
                    test_fn(bits, hidden_dim)
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {test_fn.__name__}({bits}-bit, dim={hidden_dim}): {e}")
                    failed += 1

    for bits in [2, 3, 4]:
        try:
            test_no_base_unchanged(bits)
            passed += 1
        except Exception as e:
            print(f"  FAIL: test_no_base_unchanged({bits}-bit): {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All fused residual tests passed!")


if __name__ == "__main__":
    main()
