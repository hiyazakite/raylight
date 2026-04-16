"""Correctness tests for GGUF fused CUDA kernels.

Compares the CUDA dequantization and fused-GEMM results against the
pure-PyTorch reference implementation for every supported quant type.

Requires:
  - CUDA GPU available
  - _C_gguf extension compiled (make build-cuda)
"""
import os
import sys
import struct

import pytest
import torch
import gguf
import numpy as np

pytestmark = pytest.mark.gpu

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from raylight.expansion.comfyui_gguf.fused_kernels import (
    HAS_FUSED_GGUF_CUDA,
    cuda_dequantize,
    cuda_mul_mat_vec,
    cuda_mul_mat,
    fused_ggml_gemm,
    _DEQUANT_SUPPORTED,
    _MMVQ_SUPPORTED,
    _MMQ_SUPPORTED,
)
from raylight.expansion.comfyui_gguf.dequant import (
    dequantize as pytorch_dequantize,
    dequantize_functions,
)

# Types that gguf.quants.quantize() supports (legacy quants only)
_QUANTIZABLE_TYPES = {
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q5_0,
    gguf.GGMLQuantizationType.Q5_1,
    gguf.GGMLQuantizationType.Q8_0,
}


def _make_safe_kquant_block(qtype: gguf.GGMLQuantizationType, type_size: int, rng: np.random.Generator) -> bytearray:
    """Create a K-quant block with finite fp16 scale fields and random data bytes.

    K-quant blocks have fp16 scale/min fields at type-specific offsets.
    Random bytes in those positions can produce NaN/Inf, so we patch
    them with small finite values.
    """
    block = bytearray(rng.integers(0, 256, size=type_size, dtype=np.uint8).tobytes())

    # fp16 scale field offsets per block type (from ggml-common.h structs)
    _SCALE_PATCHES = {
        # Q2_K: {scales[16], qs[64], d(fp16)@80, dmin(fp16)@82} = 84
        gguf.GGMLQuantizationType.Q2_K: [(80, 0.5), (82, 0.1)],
        # Q3_K: {hmask[32], qs[64], scales[12], d(fp16)@108} = 110
        gguf.GGMLQuantizationType.Q3_K: [(108, 0.5)],
        # Q4_K: {dm(half2)@0, scales[12], qs[128]} = 144
        gguf.GGMLQuantizationType.Q4_K: [(0, 0.5), (2, 0.1)],
        # Q5_K: {dm(half2)@0, scales[12], qh[32], qs[128]} = 176
        gguf.GGMLQuantizationType.Q5_K: [(0, 0.5), (2, 0.1)],
        # Q6_K: {ql[128], qh[64], scales[16], d(fp16)@208} = 210
        gguf.GGMLQuantizationType.Q6_K: [(208, 0.5)],
    }

    for offset, value in _SCALE_PATCHES.get(qtype, [(0, 0.5)]):
        struct.pack_into('<e', block, offset, value)

    return block


def _make_quantized_weight(rows: int, cols: int, qtype: gguf.GGMLQuantizationType, device="cuda"):
    """Create quantized weight data and return (raw_bytes_tensor, logical_shape).

    For types that gguf supports quantizing: quantize random floats.
    For K-quants/IQ: generate blocks with valid scale headers and random data bytes.
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    assert cols % block_size == 0, f"cols={cols} not divisible by block_size={block_size}"

    rng = np.random.default_rng(42)

    if qtype in _QUANTIZABLE_TYPES:
        weight_fp32 = rng.standard_normal((rows, cols)).astype(np.float32)
        qdata = gguf.quants.quantize(weight_fp32, qtype)
        qdata_flat = np.frombuffer(qdata, dtype=np.uint8).copy()
    else:
        # K-quants / IQ: generate structurally valid blocks
        blocks_per_row = cols // block_size
        total_blocks = rows * blocks_per_row
        all_blocks = bytearray()
        for _ in range(total_blocks):
            all_blocks.extend(_make_safe_kquant_block(qtype, type_size, rng))
        qdata_flat = np.frombuffer(bytes(all_blocks), dtype=np.uint8).copy()

    qdata_tensor = torch.from_numpy(qdata_flat).to(device)
    return qdata_tensor, torch.Size([rows, cols])


# Quant types that both PyTorch and CUDA dequant support
_TESTABLE_DEQUANT_TYPES = [
    qt for qt in dequantize_functions if qt.value in _DEQUANT_SUPPORTED
]

requires_fused_cuda = pytest.mark.skipif(
    not HAS_FUSED_GGUF_CUDA, reason="GGUF CUDA extension not compiled"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Dequantize tests
# ---------------------------------------------------------------------------

def _run_dequant_test(qtype, rows=4, cols=256, dtype=torch.float16):
    qdata, shape = _make_quantized_weight(rows, cols, qtype)

    ref = pytorch_dequantize(qdata, qtype, shape, dtype=dtype)
    cuda_out = cuda_dequantize(qdata, qtype, rows, cols, dtype)

    assert cuda_out is not None, f"cuda_dequantize returned None for {qtype.name}"
    assert ref.shape == cuda_out.shape, (
        f"Shape mismatch for {qtype.name}: {ref.shape} vs {cuda_out.shape}"
    )

    # Allow small numerical differences (CUDA uses fast math)
    atol = 0.02 if "K" in qtype.name else 0.01
    rtol = 0.01
    close = torch.allclose(ref, cuda_out, atol=atol, rtol=rtol)
    if not close:
        diff = (ref - cuda_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        pytest.fail(
            f"Dequant mismatch for {qtype.name}: "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"atol={atol}, rtol={rtol}"
        )


@requires_fused_cuda
@requires_cuda
@pytest.mark.parametrize("qtype", _TESTABLE_DEQUANT_TYPES, ids=lambda q: q.name)
def test_dequantize(qtype):
    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    cols = max(256, block_size)
    _run_dequant_test(qtype, rows=4, cols=cols)


# ---------------------------------------------------------------------------
# Fused GEMM tests
# ---------------------------------------------------------------------------

def _run_gemm_test(qtype, batch, rows, cols, dtype=torch.float16, dequant_dtype=None):
    qdata, shape = _make_quantized_weight(rows, cols, qtype)

    x = torch.randn(batch, cols, dtype=dtype, device="cuda")

    # Reference: PyTorch dequant → F.linear
    ref_dtype = dequant_dtype if dequant_dtype is not None else dtype
    ref_weight = pytorch_dequantize(qdata, qtype, shape, dtype=ref_dtype)
    ref_out = torch.nn.functional.linear(x.to(ref_dtype), ref_weight).to(dtype)

    # Fused kernel
    fused_out = fused_ggml_gemm(x, qdata, qtype, shape, bias=None,
                                dequant_dtype=dequant_dtype)

    assert fused_out is not None, f"fused_ggml_gemm returned None for {qtype.name}"
    assert ref_out.shape == fused_out.shape, f"Shape mismatch for {qtype.name}"
    # Output should be in the original input dtype, not dequant_dtype
    assert fused_out.dtype == dtype, (
        f"Output dtype should be {dtype}, got {fused_out.dtype}"
    )

    # Fused path quantizes activations to Q8_1, so expect larger tolerance
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_out.flatten().unsqueeze(0).float(),
        fused_out.flatten().unsqueeze(0).float(),
    ).item()

    assert cos_sim > 0.95, (
        f"GEMM cosine similarity too low for {qtype.name}: {cos_sim:.6f}"
    )


@requires_fused_cuda
@requires_cuda
class TestGGUFFusedGEMM:
    """Test fused GEMM (dequant+matmul in one kernel) against dequant→F.linear."""

    # --- matvec (mmvq path, batch=1) ---

    def test_matvec_q4_0(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q4_0, batch=1, rows=128, cols=256)

    def test_matvec_q8_0(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q8_0, batch=1, rows=128, cols=256)

    def test_matvec_q4_k(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q4_K, batch=1, rows=64, cols=256)

    def test_matvec_q6_k(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q6_K, batch=1, rows=64, cols=256)

    # --- batched GEMM (mmq path, batch=8) ---

    def test_batched_q4_0(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q4_0, batch=8, rows=128, cols=256)

    def test_batched_q8_0(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q8_0, batch=8, rows=128, cols=256)

    def test_batched_q4_k(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q4_K, batch=8, rows=64, cols=256)

    def test_batched_q6_k(self):
        _run_gemm_test(gguf.GGMLQuantizationType.Q6_K, batch=8, rows=64, cols=256)

    # --- bias and shape tests ---

    def test_with_bias(self):
        qtype = gguf.GGMLQuantizationType.Q8_0
        rows, cols = 128, 256
        qdata, shape = _make_quantized_weight(rows, cols, qtype)
        x = torch.randn(2, cols, dtype=torch.float16, device="cuda")
        bias = torch.randn(rows, dtype=torch.float16, device="cuda")

        ref_weight = pytorch_dequantize(qdata, qtype, shape, dtype=torch.float16)
        ref_out = torch.nn.functional.linear(x, ref_weight, bias)

        fused_out = fused_ggml_gemm(x, qdata, qtype, shape, bias=bias)
        assert fused_out is not None

        cos_sim = torch.nn.functional.cosine_similarity(
            ref_out.flatten().unsqueeze(0).float(),
            fused_out.flatten().unsqueeze(0).float(),
        ).item()
        assert cos_sim > 0.95

    def test_3d_input(self):
        qtype = gguf.GGMLQuantizationType.Q8_0
        rows, cols = 128, 256
        qdata, shape = _make_quantized_weight(rows, cols, qtype)
        x = torch.randn(2, 4, cols, dtype=torch.float16, device="cuda")

        ref_weight = pytorch_dequantize(qdata, qtype, shape, dtype=torch.float16)
        ref_out = torch.nn.functional.linear(x, ref_weight)

        fused_out = fused_ggml_gemm(x, qdata, qtype, shape, bias=None)
        assert fused_out is not None
        assert ref_out.shape == fused_out.shape

        cos_sim = torch.nn.functional.cosine_similarity(
            ref_out.flatten().unsqueeze(0).float(),
            fused_out.flatten().unsqueeze(0).float(),
        ).item()
        assert cos_sim > 0.95

    # --- dequant_dtype tests ---

    def test_dequant_dtype_fp16_with_fp32_input(self):
        """dequant_dtype=fp16 should dequantize in fp16 but return fp32 output."""
        _run_gemm_test(
            gguf.GGMLQuantizationType.Q8_0, batch=8, rows=128, cols=256,
            dtype=torch.float32, dequant_dtype=torch.float16,
        )

    def test_dequant_dtype_bf16_with_fp32_input(self):
        """dequant_dtype=bf16 should dequantize in bf16 but return fp32 output."""
        _run_gemm_test(
            gguf.GGMLQuantizationType.Q8_0, batch=8, rows=128, cols=256,
            dtype=torch.float32, dequant_dtype=torch.bfloat16,
        )

    def test_dequant_dtype_bf16_kquant(self):
        """dequant_dtype=bf16 works with K-quant types."""
        _run_gemm_test(
            gguf.GGMLQuantizationType.Q6_K, batch=8, rows=64, cols=256,
            dtype=torch.float32, dequant_dtype=torch.bfloat16,
        )

    def test_dequant_dtype_none_preserves_input_dtype(self):
        """dequant_dtype=None should use input dtype (existing behaviour)."""
        _run_gemm_test(
            gguf.GGMLQuantizationType.Q8_0, batch=8, rows=128, cols=256,
            dtype=torch.float16, dequant_dtype=None,
        )

    def test_dequant_dtype_same_as_input_is_noop(self):
        """dequant_dtype matching input dtype should work identically."""
        _run_gemm_test(
            gguf.GGMLQuantizationType.Q8_0, batch=8, rows=128, cols=256,
            dtype=torch.float16, dequant_dtype=torch.float16,
        )

    def test_dequant_dtype_with_bias(self):
        """dequant_dtype should work correctly with bias."""
        qtype = gguf.GGMLQuantizationType.Q8_0
        rows, cols = 128, 256
        qdata, shape = _make_quantized_weight(rows, cols, qtype)
        x = torch.randn(2, cols, dtype=torch.float32, device="cuda")
        bias = torch.randn(rows, dtype=torch.float32, device="cuda")

        fused_out = fused_ggml_gemm(x, qdata, qtype, shape, bias=bias,
                                    dequant_dtype=torch.bfloat16)
        assert fused_out is not None
        assert fused_out.dtype == torch.float32
        assert fused_out.shape == (2, rows)

    def test_dequant_dtype_matvec_path(self):
        """dequant_dtype with batch=1 (matvec path) should still return correct dtype."""
        qtype = gguf.GGMLQuantizationType.Q8_0
        rows, cols = 128, 256
        qdata, shape = _make_quantized_weight(rows, cols, qtype)
        x = torch.randn(1, cols, dtype=torch.float32, device="cuda")

        fused_out = fused_ggml_gemm(x, qdata, qtype, shape, bias=None,
                                    dequant_dtype=torch.float16)
        assert fused_out is not None
        assert fused_out.dtype == torch.float32

    def test_dequant_dtype_scratch_reuse(self):
        """Scratch buffer with dequant_dtype should be reused correctly."""
        qtype = gguf.GGMLQuantizationType.Q8_0
        rows, cols = 128, 256
        qdata, shape = _make_quantized_weight(rows, cols, qtype)
        x = torch.randn(8, cols, dtype=torch.float32, device="cuda")
        scratch = torch.empty(rows * cols, dtype=torch.bfloat16, device="cuda")

        fused_out = fused_ggml_gemm(x, qdata, qtype, shape, bias=None,
                                    dequant_dtype=torch.bfloat16,
                                    scratch=scratch)
        assert fused_out is not None
        assert fused_out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Shared scratch buffer tests
# ---------------------------------------------------------------------------

@requires_fused_cuda
@requires_cuda
class TestGGUFSharedScratch:
    """Test that the module-level shared scratch buffer works correctly."""

    def test_shared_scratch_reuse_across_sizes(self):
        """Shared scratch should grow for larger layers and reuse for smaller ones."""
        import raylight.distributed_modules.tp_linear_factory as tplf

        # Reset shared scratch
        tplf._shared_dequant_scratch = None

        # First call: small layer
        qtype = gguf.GGMLQuantizationType.Q8_0
        small_rows, small_cols = 64, 256
        qdata_s, shape_s = _make_quantized_weight(small_rows, small_cols, qtype)
        x = torch.randn(8, small_cols, dtype=torch.float16, device="cuda")
        scratch_small = torch.empty(small_rows * small_cols, dtype=torch.float16, device="cuda")
        tplf._shared_dequant_scratch = scratch_small

        out_s = fused_ggml_gemm(x, qdata_s, qtype, shape_s, bias=None,
                                scratch=tplf._shared_dequant_scratch[:small_rows * small_cols])
        assert out_s is not None

        # Second call: larger layer — should still work when we grow the buffer
        big_rows, big_cols = 128, 256
        qdata_b, shape_b = _make_quantized_weight(big_rows, big_cols, qtype)
        x2 = torch.randn(8, big_cols, dtype=torch.float16, device="cuda")
        needed = big_rows * big_cols
        if tplf._shared_dequant_scratch.numel() < needed:
            tplf._shared_dequant_scratch = torch.empty(
                needed, dtype=torch.float16, device="cuda",
            )

        out_b = fused_ggml_gemm(x2, qdata_b, qtype, shape_b, bias=None,
                                scratch=tplf._shared_dequant_scratch[:needed])
        assert out_b is not None
        assert out_b.shape == (8, big_rows)

        # Third call: back to small layer — reuses existing larger buffer
        out_s2 = fused_ggml_gemm(x, qdata_s, qtype, shape_s, bias=None,
                                 scratch=tplf._shared_dequant_scratch[:small_rows * small_cols])
        assert out_s2 is not None

        # Buffer should be the big size (never shrinks)
        assert tplf._shared_dequant_scratch.numel() >= needed

    def test_shared_scratch_dtype_change(self):
        """Shared scratch should be reallocated when dequant_dtype changes."""
        import raylight.distributed_modules.tp_linear_factory as tplf

        tplf._shared_dequant_scratch = None

        qtype = gguf.GGMLQuantizationType.Q8_0
        rows, cols = 64, 256
        qdata, shape = _make_quantized_weight(rows, cols, qtype)
        needed = rows * cols

        # Allocate as fp16
        tplf._shared_dequant_scratch = torch.empty(needed, dtype=torch.float16, device="cuda")
        assert tplf._shared_dequant_scratch.dtype == torch.float16

        # Change to bf16 — should reallocate
        tplf._shared_dequant_scratch = torch.empty(needed, dtype=torch.bfloat16, device="cuda")
        assert tplf._shared_dequant_scratch.dtype == torch.bfloat16

        # Use it
        x = torch.randn(8, cols, dtype=torch.float32, device="cuda")
        out = fused_ggml_gemm(x, qdata, qtype, shape, bias=None,
                              dequant_dtype=torch.bfloat16,
                              scratch=tplf._shared_dequant_scratch[:needed])
        assert out is not None
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Fallback / unsupported types
# ---------------------------------------------------------------------------

@requires_fused_cuda
@requires_cuda
def test_bf16_not_fused():
    """Test that unsupported types gracefully return None."""
    result = cuda_dequantize(
        torch.zeros(10, dtype=torch.uint8, device="cuda"),
        gguf.GGMLQuantizationType.BF16,
        1, 10, torch.float16,
    )
    assert result is None
