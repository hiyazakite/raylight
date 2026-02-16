
import sys
import os
sys.path.append("/root/ComfyUI/custom_nodes/raylight/src")

import torch
import unittest
from unittest.mock import patch, MagicMock
from raylight.distributed_modules.compact.compress_quantize import quantize_int4, dequantize_int4, quantize_int8, dequantize_int8
from raylight.distributed_modules.compact.utils import CompactConfig
from raylight.distributed_modules.compact.main import CompactCache, compact_init
from raylight.distributed_modules.attention.compact import CompactAttentionBackend

class TestCompactQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def test_quantize_int4(self):
        if not torch.cuda.is_available():
            print("Skipping test_quantize_int4 because CUDA is not available")
            return
        
        # N must be even, C must be divisible by something? No, C can be anything.
        N, C = 16, 64
        x = torch.randn(N, C, dtype=torch.float16, device=self.device)
        
        packed, scale, min_val = quantize_int4(x)
        
        # Check shapes
        self.assertEqual(packed.shape, (N // 2, C))
        self.assertEqual(scale.shape, (1, C))
        self.assertEqual(min_val.shape, (1, C))
        self.assertEqual(packed.dtype, torch.uint8)
        
        x_recon = dequantize_int4(packed, scale, min_val)
        
        self.assertEqual(x_recon.shape, x.shape)
        self.assertEqual(x_recon.dtype, torch.float16)
        
        # Check error bounds (loose check for 4-bit)
        # 4-bit quantization has limited precision, so we expect some error
        mae = torch.mean(torch.abs(x - x_recon))
        print(f"INT4 MAE: {mae.item()}")
        # Random normal has std 1. 4-bit range is 16 levels. 
        # Approx error should be < 0.2
        self.assertTrue(mae < 0.25) 

    def test_quantize_int8(self):
        if not torch.cuda.is_available():
            return
        N, C = 16, 64
        x = torch.randn(N, C, dtype=torch.float16, device=self.device)
        q, s, zp = quantize_int8(x)
        x_recon = dequantize_int8(q, s, zp)
        mae = torch.mean(torch.abs(x - x_recon))
        print(f"INT8 MAE: {mae.item()}")
        self.assertTrue(mae < 0.05)

    def test_compact_cache(self):
        if not torch.cuda.is_available():
            return
        # Test CompactCache with 4-bit quantization
        cache = CompactCache(quantize=True, quant_bits=4)
        
        N, C = 16, 64
        key = "test_key"
        base = torch.randn(N, C, dtype=torch.float16, device=self.device)
        
        cache.put(key, base, None)
        
        # Check if stored as tuple (packed, scale, min_val)
        stored = cache.base[key]
        self.assertIsInstance(stored, tuple)
        self.assertEqual(len(stored), 3)
        self.assertEqual(stored[0].dtype, torch.uint8)
        
        # Check retrieval
        retrieved = cache.get_base(key)
        self.assertEqual(retrieved.shape, base.shape)
        mae = torch.mean(torch.abs(base - retrieved))
        print(f"Cache INT4 MAE: {mae.item()}")
        self.assertTrue(mae < 0.25)
        
        # Test CompactCache with 8-bit quantization
        cache8 = CompactCache(quantize=True, quant_bits=8)
        cache8.put(key, base, None)
        stored8 = cache8.base[key]
        self.assertIsInstance(stored8, tuple)
        self.assertEqual(len(stored8), 3) # q, s, zp
        
        retrieved8 = cache8.get_base(key)
        mae8 = torch.mean(torch.abs(base - retrieved8))
        print(f"Cache INT8 MAE: {mae8.item()}")
        self.assertTrue(mae8 < 0.05)

    def test_default_quant_bits(self):
        # Verify that CompactConfig defaults to 8-bit if not specified
        config = CompactConfig(quantized_cache=True)
        self.assertEqual(config.cache_quant_bits, 8)
        
        # Verify that CompactCache defaults to 8-bit
        cache = CompactCache(quantize=True)
        self.assertEqual(cache.quant_bits, 8)

    @patch("raylight.distributed_modules.attention.compact.CompactConfig")
    @patch("raylight.distributed_modules.attention.compact.compact_init")
    @patch("raylight.distributed_modules.attention.compact.compact_hello")
    @patch("raylight.distributed_modules.attention.compact.ensure_hf_fp8_cuda_kernel")
    @patch("raylight.distributed_modules.attention.compact.CompactxFuserLongContextAttention")
    @patch("raylight.distributed_modules.attention.compact.AttnType")
    def test_compact_attention_env_vars(self, mock_attn_type, mock_attn, mock_ensure, mock_hello, mock_init, mock_config):
        backend = CompactAttentionBackend()
        
        # Mock AttnType access
        mock_attn_type.__getitem__.return_value = "MOCKED_ATTN_TYPE"
        
        # Test 1: Enabled via Env Var
        with patch.dict(os.environ, {
            "RAYLIGHT_COMPACT_QUANTIZED_CACHE": "1",
            "RAYLIGHT_COMPACT_CACHE_QUANT_BITS": "4"
        }):
            backend.create_attention("SAGE_FP8_CUDA", False)
            
            # Check if CompactConfig was called with correct args
            call_args = mock_config.call_args
            self.assertTrue(call_args.kwargs.get("quantized_cache"))
            self.assertEqual(call_args.kwargs.get("cache_quant_bits"), 4)

        # Test 2: Disabled via Env Var (Default)
        with patch.dict(os.environ, {}, clear=True):
            backend.create_attention("SAGE_FP8_CUDA", False)
            call_args = mock_config.call_args
            # Default quantized_cache is False if env var is missing
            self.assertFalse(call_args.kwargs.get("quantized_cache"))
        
        # Test 3: Kwargs override Env Var
        with patch.dict(os.environ, {
            "RAYLIGHT_COMPACT_QUANTIZED_CACHE": "1",
            "RAYLIGHT_COMPACT_CACHE_QUANT_BITS": "4"
        }):
            # Pass explicit False and None
            backend.create_attention("SAGE_FP8_CUDA", False, compact_quantized_cache=False)
            call_args = mock_config.call_args
            self.assertFalse(call_args.kwargs.get("quantized_cache"))

    def test_quantize_int8_bf16(self):
        """Test INT8 quantization with BF16 input."""
        if not torch.cuda.is_available():
            # BF16 might not be supported on CPU for some operations or just skip if no GPU
            # But let's try. torch.bfloat16 is supported on CPU.
            pass

        # Create a random tensor in BF16
        input_tensor_bf16 = torch.randn(10, 32, dtype=torch.bfloat16, device=self.device)
        
        # Quantize
        try:
            q_tensor, scale, zero_point = quantize_int8(input_tensor_bf16)
        except Exception as e:
            self.fail(f"quantize_int8 failed with BF16 input: {e}")
            
        # Check output types
        self.assertEqual(q_tensor.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.half) # Scale should be FP16 as per implementation
        self.assertEqual(zero_point.dtype, torch.int16)
        
        # Basic shape check
        self.assertEqual(q_tensor.shape, input_tensor_bf16.shape)

if __name__ == '__main__':
    unittest.main()
