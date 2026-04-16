
"""Tests for compact quantization (INT4/INT8) and RaylightConfig env var parsing."""
import os

import pytest
import torch
from unittest.mock import patch

pytestmark = pytest.mark.gpu

from raylight.distributed_modules.attention.backends.fusion.compress_quantize import (
    quantize_int4, dequantize_int4, quantize_int8, dequantize_int8,
)
from raylight.distributed_modules.attention.backends.fusion.utils import (
    CompactConfig as FusionCompactConfig, CompactCache,
)
from raylight.config import RaylightConfig

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Quantization round-trip tests
# ---------------------------------------------------------------------------

class TestCompactQuantization:

    @requires_cuda
    def test_quantize_int4(self, device):
        N, C = 16, 64
        x = torch.randn(N, C, dtype=torch.float16, device=device)

        packed, scale, min_val = quantize_int4(x)

        assert packed.shape == (N // 2, C)
        assert scale.shape == (1, C)
        assert min_val.shape == (1, C)
        assert packed.dtype == torch.uint8

        x_recon = dequantize_int4(packed, scale, min_val)
        assert x_recon.shape == x.shape
        assert x_recon.dtype == torch.float16

        mae = torch.mean(torch.abs(x - x_recon))
        assert mae < 0.25

    @requires_cuda
    def test_quantize_int8(self, device):
        N, C = 16, 64
        x = torch.randn(N, C, dtype=torch.float16, device=device)
        q, s, zp = quantize_int8(x)
        x_recon = dequantize_int8(q, s, zp)
        mae = torch.mean(torch.abs(x - x_recon))
        assert mae < 0.05

    @requires_cuda
    def test_compact_cache_int4(self, device):
        cache = CompactCache(quantize=True, quant_bits=4)
        N, C = 16, 64
        base = torch.randn(N, C, dtype=torch.float16, device=device)

        cache.put("k", base, None)
        stored = cache.base["k"]
        assert isinstance(stored, tuple) and len(stored) == 3
        assert stored[0].dtype == torch.uint8

        retrieved = cache.get_base("k")
        assert retrieved.shape == base.shape
        assert torch.mean(torch.abs(base - retrieved)) < 0.25

    @requires_cuda
    def test_compact_cache_int8(self, device):
        cache = CompactCache(quantize=True, quant_bits=8)
        N, C = 16, 64
        base = torch.randn(N, C, dtype=torch.float16, device=device)

        cache.put("k", base, None)
        stored = cache.base["k"]
        assert isinstance(stored, tuple) and len(stored) == 3

        retrieved = cache.get_base("k")
        assert torch.mean(torch.abs(base - retrieved)) < 0.05

    def test_default_quant_bits(self):
        config = FusionCompactConfig(quantized_cache=True)
        assert config.cache_quant_bits == 8

        cache = CompactCache(quantize=True)
        assert cache.quant_bits == 8

    @requires_cuda
    def test_quantize_int8_bf16(self, device):
        x = torch.randn(10, 32, dtype=torch.bfloat16, device=device)
        q, s, zp = quantize_int8(x)

        assert q.dtype == torch.int8
        assert s.dtype == torch.half
        assert zp.dtype == torch.int16
        assert q.shape == x.shape


# ---------------------------------------------------------------------------
# RaylightConfig env-var parsing for compact quantization
# ---------------------------------------------------------------------------

class TestCompactAttentionEnvVars:
    """Verify RaylightConfig.from_env() reads compact quantization env vars."""

    def test_env_vars_enable_quant(self):
        with patch.dict(os.environ, {
            "RAYLIGHT_COMPACT_QUANTIZED_CACHE": "1",
            "RAYLIGHT_COMPACT_CACHE_QUANT_BITS": "4",
        }, clear=False):
            cfg = RaylightConfig.from_env()
            assert cfg.compact.kv_cache_quant_enable is True
            assert cfg.compact.kv_cache_quant_bits == 4

    def test_env_vars_default_disabled(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("RAYLIGHT_COMPACT_QUANTIZED_CACHE", "RAYLIGHT_COMPACT_CACHE_QUANT_BITS")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = RaylightConfig.from_env()
            assert cfg.compact.kv_cache_quant_enable is False
            assert cfg.compact.kv_cache_quant_bits == 8

    def test_env_vars_8bit(self):
        with patch.dict(os.environ, {
            "RAYLIGHT_COMPACT_QUANTIZED_CACHE": "1",
            "RAYLIGHT_COMPACT_CACHE_QUANT_BITS": "8",
        }, clear=False):
            cfg = RaylightConfig.from_env()
            assert cfg.compact.kv_cache_quant_enable is True
            assert cfg.compact.kv_cache_quant_bits == 8
