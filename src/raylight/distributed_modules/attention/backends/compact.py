from typing import Callable
from yunchang.kernels import AttnType
from ..interface import AttentionBackend
from ..layer import RaylightAttention
from ...sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

# Vendored Compact Modules
from raylight.distributed_modules.attention.backends.fusion.main import compact_init, CompactConfig, compact_hello
from raylight.distributed_modules.attention.backends.fusion.utils import COMPACT_COMPRESS_TYPE

class CompactAttentionBackend(AttentionBackend):
    """
    Attention backend enabling CompactFusion optimization for activation compression.
    """

    def create_attention(self, attn_type: str, sync_ulysses: bool, ring_impl_type: str = "basic", **kwargs) -> Callable:
        """
        Creates CompactFusion attention.
        
        Expected kwargs:
            compact_enabled (bool): Default True
            compact_fastpath (bool): Default True
            compact_residual (int): Default 1 (1st order)
            compact_rank (int): Default -1
            compact_quantized_cache (bool): Default True
            compact_cache_quant_bits (int): Default 8
        """
        print(f"Using CompactFusion XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}, Impl: {ring_impl_type}")
        
        # 1. Configuration Lifecycle
        # Default policy: Warmup for first 5 steps, then configured compression (default BINARY)
        import os
        
        # Priority: env var > kwargs > default (True)
        env_val = os.environ.get("RAYLIGHT_COMPACT_QUANTIZED_CACHE")
        if env_val is not None:
            quantized_cache = (env_val == "1")
        else:
            quantized_cache = kwargs.get("compact_quantized_cache", True)
        
        # Parse delta compression type
        delta_compression_str = os.environ.get("RAYLIGHT_DELTA_COMPRESSION", "BINARY").upper()
        try:
            delta_compression_type = COMPACT_COMPRESS_TYPE(delta_compression_str.lower())
        except ValueError:
            print(f"[Raylight] Warning: Unknown delta compression '{delta_compression_str}', falling back to BINARY")
            delta_compression_type = COMPACT_COMPRESS_TYPE.BINARY
            
        warmup_steps = int(os.environ.get("RAYLIGHT_COMPACT_WARMUP_STEPS", "5"))

        def default_compress_func(layer_idx, step):
             if step is None:
                 step = 0
             if step < warmup_steps:
                  return COMPACT_COMPRESS_TYPE.WARMUP
             return delta_compression_type
        
        cache_quant_bits = kwargs.get("compact_cache_quant_bits")
        if cache_quant_bits is None:
            env_bits = os.environ.get("RAYLIGHT_COMPACT_CACHE_QUANT_BITS")
            if env_bits is not None:
                cache_quant_bits = int(env_bits)

        config = CompactConfig(
             enabled=kwargs.get("compact_enabled", True),
             fastpath=kwargs.get("compact_fastpath", True),
             residual=kwargs.get("compact_residual", 1),
             comp_rank=kwargs.get("compact_rank", -1),
             quantized_cache=quantized_cache,
             cache_quant_bits=cache_quant_bits,
             ef=True, # Error Feedback required for fastpath/residual
             compress_func=default_compress_func
        )
        
        compact_init(config)
        compact_hello()
        
        # 2. Kernel Setup
        attn_enum = AttnType[attn_type]
        if attn_type == "SAGE_FP8_CUDA":
            ensure_hf_fp8_cuda_kernel()
        elif attn_type == "SAGE_FP8_SM90":
            ensure_hf_sm90_kernel()

        # 3. Instantiate Centralized Attention Class
        # This class automatically hooks into the dispatcher
        xfuser_attn = RaylightAttention(
            use_sync=sync_ulysses,
            attn_type=attn_enum, 
            ring_impl_type=ring_impl_type,
            use_pack_qkv=False,
            use_compact_ring=True # Compact backend uses compact ring by default
        )

        # 4. Wrapper Function (same signature as Standard)
        # OPT: Pre-computed softmax_scale per head_dim avoids redundant
        # exponentiation on every attention call.  head_dim is constant for
        # the lifetime of a model, so we cache after the first call.
        _softmax_scale_cache: dict[int, float] = {}

        def _attention_xfuser_compact_unmask(
                q, k, v, heads,
                join_q=None, join_k=None, join_v=None,
                mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False,
                **kwargs):

            if skip_reshape:
                b, _, _, dim_head = q.shape
                if join_q is not None:
                    j_b, _, _, j_dim_head = join_q.shape
            else:
                b, _, dim_head = q.shape
                dim_head //= heads
                # OPT 3: Reshape directly to (B, S, H, D) — xfuser_attn expects this layout.
                q, k, v = map(
                    lambda t: t.view(b, -1, heads, dim_head),
                    (q, k, v),
                )
                if join_q is not None:
                    assert join_k is not None and join_v is not None
                    j_b, _, j_dim_head = join_q.shape
                    j_dim_head //= heads
                    join_q, join_k, join_v = map(
                        lambda t: t.view(j_b, -1, heads, j_dim_head),
                        (join_q, join_k, join_v),
                    )

            if mask is not None:
                 if mask.ndim == 2: mask = mask.unsqueeze(0)
                 if mask.ndim == 3: mask = mask.unsqueeze(1)
            
            # OPT: Pre-compute softmax_scale once per head_dim (constant for model lifetime)
            scale = _softmax_scale_cache.get(dim_head)
            if scale is None:
                scale = dim_head ** -0.5
                _softmax_scale_cache[dim_head] = scale

            if join_q is not None:
                assert join_k is not None and join_v is not None
                out = xfuser_attn(
                    None,
                    q, k, v,
                    joint_strategy="rear",
                    joint_tensor_query=join_q,
                    joint_tensor_key=join_k,
                    joint_tensor_value=join_v,
                    mask=mask,
                    softmax_scale=scale,
                    mod_idx=kwargs.get("mod_idx"),
                    current_iter=kwargs.get("current_iter"),
                )
            else:
                out = xfuser_attn(
                    None,
                    q, k, v,
                    mask=mask,
                    softmax_scale=scale,
                    mod_idx=kwargs.get("mod_idx"),
                    current_iter=kwargs.get("current_iter"),
                )
            
            if not skip_output_reshape:
                out = out.reshape(b, -1, heads * dim_head)
            return out

        return _attention_xfuser_compact_unmask
