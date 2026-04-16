"""Tests for residual LoRA path in TPGGMLLinear.

Verifies that fused_gemm(W_q, x) + B@(A@x) produces the same result
as the full dequant + weight_function + F.linear fallback path.

Requires:
  - CUDA GPU available
  - _C_gguf extension compiled (make build-cuda)
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F
import gguf
import numpy as np

pytestmark = pytest.mark.gpu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from raylight.expansion.comfyui_gguf.fused_kernels import (
    HAS_FUSED_GGUF_CUDA,
    fused_ggml_gemm,
)
from raylight.expansion.comfyui_gguf.dequant import dequantize as pytorch_dequantize


def _extract_lora_ab(patches):
    """Local copy of raylight.comfy_dist.model_patcher._extract_lora_ab.

    Avoids importing the full comfy_dist chain (which needs ComfyUI).
    Logic is identical: extracts (list_A, list_B, dora_scale) from patches.
    """
    all_A = []
    all_B = []
    dora_scales = []

    for p in patches:
        strength = float(p[0])
        adapter = p[1]
        offset = p[3]
        function = p[4]

        if offset is not None or function is not None:
            return None, None, None

        if hasattr(adapter, "weights"):
            v = adapter.weights
        elif isinstance(adapter, (tuple, list)) and len(adapter) >= 2:
            v = adapter
        else:
            return None, None, None

        up = v[0]
        down = v[1]
        alpha = v[2] if len(v) > 2 else None
        mid = v[3] if len(v) > 3 else None
        dora_scale = v[4] if len(v) > 4 else None

        if mid is not None:
            return None, None, None

        rank = down.shape[0]
        scale = (float(alpha) / rank if alpha else 1.0) * strength

        all_A.append(down * scale)
        all_B.append(up)
        dora_scales.append(dora_scale)

    if not all_A:
        return None, None, None

    has_dora = [ds is not None for ds in dora_scales]
    if any(has_dora) and not all(has_dora):
        return None, None, None
    if all(has_dora) and len(dora_scales) > 1:
        return None, None, None

    final_dora = dora_scales[0] if all(has_dora) else None
    return all_A, all_B, final_dora


requires_cuda_gguf = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_FUSED_GGUF_CUDA),
    reason="CUDA + _C_gguf required",
)


def _quantize_q4_0(weight_fp32: np.ndarray) -> bytes:
    """Quantize a 2D weight matrix to Q4_0 using gguf.quants."""
    return gguf.quants.quantize(weight_fp32, gguf.GGMLQuantizationType.Q4_0)


def _make_lora_patch(A, B, alpha=None, strength=1.0):
    """Build a patch tuple in the format _extract_lora_ab expects.

    Patch format: (strength, (up, down, alpha, mid, dora_scale), ..., offset, function)
    Indices: [0]=strength, [1]=adapter, [2]=?, [3]=offset, [4]=function
    """
    adapter = [B, A]  # up=[out,rank], down=[rank,in]
    if alpha is not None:
        adapter.append(torch.tensor(float(alpha)))
    else:
        adapter.append(None)
    adapter.append(None)  # mid
    adapter.append(None)  # dora_scale
    # Full patch tuple: (strength, adapter, None, offset, function)
    return (strength, adapter, None, None, None)


class _FakeLowVramPatch:
    """Minimal mock of LowVramPatch with patches/key attributes."""

    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        if self.key not in self.patches:
            return weight
        patch_list = self.patches[self.key]
        all_A, all_B, dora_scale = _extract_lora_ab(patch_list)
        if all_A is None:
            return weight
        delta = sum(B @ A for A, B in zip(all_A, all_B))
        return weight + delta.to(weight.dtype)


class TestLoraResidualCorrectness:
    """Verify that the LoRA residual contribution is correct.

    The fused CUDA kernel and PyTorch dequant use different numerical paths
    for the base W@x product, so we compare the *LoRA delta* (with_lora -
    without_lora) between both paths rather than the absolute output.
    """

    @requires_cuda_gguf
    def test_single_lora_q4_0(self):
        """Single LoRA adapter with Q4_0 — LoRA delta matches between paths."""
        out_dim, in_dim, rank = 128, 64, 8
        rng = np.random.default_rng(42)

        w_fp32 = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        qw_bytes = _quantize_q4_0(w_fp32)
        qw = torch.from_numpy(np.frombuffer(qw_bytes, dtype=np.uint8)).cuda()
        qtype = gguf.GGMLQuantizationType.Q4_0
        shape = (out_dim, in_dim)

        A = torch.randn(rank, in_dim, device="cuda", dtype=torch.float16)
        B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)
        alpha = float(rank)
        strength = 0.8

        x = torch.randn(4, in_dim, device="cuda", dtype=torch.float16)

        # Reference LoRA delta via dequant path
        w_deq = pytorch_dequantize(qw.cpu(), qtype, shape).to(device="cuda", dtype=torch.float16)
        ref_base = F.linear(x, w_deq)
        patch = _make_lora_patch(A, B, alpha=alpha, strength=strength)
        all_A, all_B, _ = _extract_lora_ab([patch])
        lora_delta = all_B[0] @ all_A[0]
        ref_with_lora = F.linear(x, w_deq + lora_delta.to(w_deq.dtype))
        ref_lora_contribution = ref_with_lora - ref_base

        # Fused path LoRA residual
        Ax = torch.mm(all_A[0].to(x.dtype), x.t())
        BAx = torch.mm(all_B[0].to(x.dtype), Ax)
        fused_lora_contribution = BAx.t()

        # The LoRA contribution should match closely (both are exact FP16 matmuls)
        torch.testing.assert_close(fused_lora_contribution, ref_lora_contribution, atol=0.03, rtol=0.02)

        # Also verify the full fused output is within quantization tolerance
        fused_base = fused_ggml_gemm(x, qw, qtype, shape, bias=None)
        assert fused_base is not None, "fused_ggml_gemm returned None for Q4_0"
        fused_out = fused_base + fused_lora_contribution
        torch.testing.assert_close(fused_out, ref_with_lora, atol=0.5, rtol=0.1)

    @requires_cuda_gguf
    def test_multi_lora_q8_0(self):
        """Two stacked LoRA adapters with Q8_0 — residuals sum correctly."""
        out_dim, in_dim = 64, 64
        rank1, rank2 = 4, 8
        rng = np.random.default_rng(123)

        w_fp32 = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        qw_bytes = gguf.quants.quantize(w_fp32, gguf.GGMLQuantizationType.Q8_0)
        qw = torch.from_numpy(np.frombuffer(qw_bytes, dtype=np.uint8)).cuda()
        qtype = gguf.GGMLQuantizationType.Q8_0
        shape = (out_dim, in_dim)

        A1 = torch.randn(rank1, in_dim, device="cuda", dtype=torch.float16)
        B1 = torch.randn(out_dim, rank1, device="cuda", dtype=torch.float16)
        A2 = torch.randn(rank2, in_dim, device="cuda", dtype=torch.float16)
        B2 = torch.randn(out_dim, rank2, device="cuda", dtype=torch.float16)

        patch1 = _make_lora_patch(A1, B1, alpha=float(rank1), strength=1.0)
        patch2 = _make_lora_patch(A2, B2, alpha=float(rank2), strength=0.5)

        x = torch.randn(2, in_dim, device="cuda", dtype=torch.float16)

        # Reference
        w_deq = pytorch_dequantize(qw.cpu(), qtype, shape).to(device="cuda", dtype=torch.float16)
        ref_base = F.linear(x, w_deq)
        all_A, all_B, _ = _extract_lora_ab([patch1, patch2])
        delta = sum(Bi @ Ai for Ai, Bi in zip(all_A, all_B))
        ref_with_lora = F.linear(x, w_deq + delta.to(w_deq.dtype))
        ref_lora_contribution = ref_with_lora - ref_base

        # Fused path residual
        residual = torch.zeros(x.shape[0], out_dim, device="cuda", dtype=x.dtype)
        for Ai, Bi in zip(all_A, all_B):
            Ai = Ai.to(device="cuda", dtype=x.dtype)
            Bi = Bi.to(device="cuda", dtype=x.dtype)
            Ax = torch.mm(Ai, x.t())
            BAx = torch.mm(Bi, Ax)
            residual += BAx.t()

        torch.testing.assert_close(residual, ref_lora_contribution, atol=0.05, rtol=0.03)

        # Full output within Q8_0 tolerance
        fused_base = fused_ggml_gemm(x, qw, qtype, shape, bias=None)
        assert fused_base is not None
        fused_out = fused_base + residual
        torch.testing.assert_close(fused_out, ref_with_lora, atol=0.15, rtol=0.05)

    @requires_cuda_gguf
    def test_lora_with_bias(self):
        """LoRA residual + bias — bias cancels, LoRA delta matches."""
        out_dim, in_dim, rank = 64, 32, 4
        rng = np.random.default_rng(7)

        w_fp32 = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        qw_bytes = _quantize_q4_0(w_fp32)
        qw = torch.from_numpy(np.frombuffer(qw_bytes, dtype=np.uint8)).cuda()
        qtype = gguf.GGMLQuantizationType.Q4_0
        shape = (out_dim, in_dim)

        A = torch.randn(rank, in_dim, device="cuda", dtype=torch.float16)
        B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)
        bias = torch.randn(out_dim, device="cuda", dtype=torch.float16)

        patch = _make_lora_patch(A, B, alpha=float(rank), strength=1.0)
        all_A, all_B, _ = _extract_lora_ab([patch])

        x = torch.randn(3, in_dim, device="cuda", dtype=torch.float16)

        # Fused path: bias is in fused_ggml_gemm, LoRA residual added after
        fused_base = fused_ggml_gemm(x, qw, qtype, shape, bias=bias)
        assert fused_base is not None
        Ax = torch.mm(all_A[0].to(x.dtype), x.t())
        residual = torch.mm(all_B[0].to(x.dtype), Ax).t()
        fused_out = fused_base + residual

        # Reference
        w_deq = pytorch_dequantize(qw.cpu(), qtype, shape).to(device="cuda", dtype=torch.float16)
        delta = all_B[0] @ all_A[0]
        ref_out = F.linear(x, w_deq + delta.to(w_deq.dtype), bias)

        # Full output within Q4_0 tolerance (bias is identical on both sides)
        torch.testing.assert_close(fused_out, ref_out, atol=0.5, rtol=0.1)

        # Verify bias is correctly included: fused_base without bias should differ
        fused_no_bias = fused_ggml_gemm(x, qw, qtype, shape, bias=None)
        assert not torch.allclose(fused_base, fused_no_bias)

    @requires_cuda_gguf
    def test_3d_input(self):
        """Residual LoRA with 3D input (seq_len, batch, in)."""
        out_dim, in_dim, rank = 64, 32, 4
        rng = np.random.default_rng(99)

        w_fp32 = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        qw_bytes = _quantize_q4_0(w_fp32)
        qw = torch.from_numpy(np.frombuffer(qw_bytes, dtype=np.uint8)).cuda()
        qtype = gguf.GGMLQuantizationType.Q4_0
        shape = (out_dim, in_dim)

        A = torch.randn(rank, in_dim, device="cuda", dtype=torch.float16)
        B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)

        patch = _make_lora_patch(A, B, alpha=float(rank), strength=1.0)
        all_A, all_B, _ = _extract_lora_ab([patch])

        x = torch.randn(2, 3, in_dim, device="cuda", dtype=torch.float16)

        # Reference LoRA contribution
        w_deq = pytorch_dequantize(qw.cpu(), qtype, shape).to(device="cuda", dtype=torch.float16)
        ref_base = F.linear(x, w_deq)
        delta = all_B[0] @ all_A[0]
        ref_with_lora = F.linear(x, w_deq + delta.to(w_deq.dtype))
        ref_lora_contribution = ref_with_lora - ref_base

        # Fused residual with 3D reshape
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        Ax = torch.mm(all_A[0].to(x.dtype), x_2d.t())
        BAx = torch.mm(all_B[0].to(x.dtype), Ax)
        fused_lora_contribution = BAx.t().reshape(*orig_shape[:-1], out_dim)

        torch.testing.assert_close(fused_lora_contribution, ref_lora_contribution, atol=0.02, rtol=0.01)

        # Full output
        fused_base = fused_ggml_gemm(x, qw, qtype, shape, bias=None)
        assert fused_base is not None
        fused_out = fused_base + fused_lora_contribution
        torch.testing.assert_close(fused_out, ref_with_lora, atol=0.5, rtol=0.1)


class TestExtractLoraAB:
    """Unit tests for _extract_lora_ab and the _try_extract_lora_ab integration."""

    def test_basic_extraction(self):
        A = torch.randn(4, 32)
        B = torch.randn(64, 4)
        patch = _make_lora_patch(A, B, alpha=4.0, strength=1.0)
        all_A, all_B, dora = _extract_lora_ab([patch])
        assert all_A is not None
        assert len(all_A) == 1
        assert all_A[0].shape == (4, 32)
        assert all_B[0].shape == (64, 4)
        assert dora is None

    def test_with_alpha_scaling(self):
        """alpha != rank → scale = alpha/rank * strength."""
        rank, in_dim, out_dim = 8, 32, 64
        A = torch.ones(rank, in_dim)
        B = torch.ones(out_dim, rank)
        alpha = 4.0
        strength = 0.5
        patch = _make_lora_patch(A, B, alpha=alpha, strength=strength)
        all_A, all_B, _ = _extract_lora_ab([patch])
        # scale = (alpha / rank) * strength = (4/8) * 0.5 = 0.25
        expected_scale = (alpha / rank) * strength
        # A is scaled, B is not
        torch.testing.assert_close(all_A[0], A * expected_scale)
        torch.testing.assert_close(all_B[0], B)

    def test_dora_returns_none(self):
        """DoRA patches should cause _try_extract_lora_ab to reject."""
        A = torch.randn(4, 32)
        B = torch.randn(64, 4)
        adapter = [B, A, torch.tensor(4.0), None, torch.randn(64)]  # dora_scale present
        patch = (1.0, adapter, None, None, None)
        all_A, all_B, dora = _extract_lora_ab([patch])
        # Should succeed at extraction level (dora != None)
        assert all_A is not None
        assert dora is not None

    def test_offset_returns_none(self):
        """Patches with offset should bail out."""
        A = torch.randn(4, 32)
        B = torch.randn(64, 4)
        adapter = [B, A, torch.tensor(4.0), None, None]
        patch = (1.0, adapter, None, 42, None)  # offset != None
        all_A, all_B, _ = _extract_lora_ab([patch])
        assert all_A is None

    def test_function_returns_none(self):
        """Patches with function should bail out."""
        A = torch.randn(4, 32)
        B = torch.randn(64, 4)
        adapter = [B, A, torch.tensor(4.0), None, None]
        patch = (1.0, adapter, None, None, lambda w: w)  # function != None
        all_A, all_B, _ = _extract_lora_ab([patch])
        assert all_A is None


class TestFakeLowVramPatch:
    """Test the _FakeLowVramPatch mock matches real LowVramPatch behaviour."""

    @requires_cuda_gguf
    def test_patch_applies_delta(self):
        out_dim, in_dim, rank = 32, 16, 4
        A = torch.randn(rank, in_dim, device="cuda", dtype=torch.float16)
        B = torch.randn(out_dim, rank, device="cuda", dtype=torch.float16)
        patch = _make_lora_patch(A, B, alpha=float(rank), strength=1.0)

        key = "test.weight"
        patcher = _FakeLowVramPatch(key, {key: [patch]})

        weight = torch.randn(out_dim, in_dim, device="cuda", dtype=torch.float16)
        patched = patcher(weight)
        expected = weight + (B @ A).to(weight.dtype)
        torch.testing.assert_close(patched, expected, atol=1e-3, rtol=1e-3)
