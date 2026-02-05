from typing import Callable
from yunchang.kernels import AttnType
from .interface import AttentionBackend
from ..sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

# Vendored Compact Modules
from raylight.distributed_modules.compact.main import compact_init, CompactConfig, compact_hello
from raylight.distributed_modules.compact.utils import COMPACT_COMPRESS_TYPE
from raylight.distributed_modules.compact.attn_layer import xFuserLongContextAttention as CompactxFuserLongContextAttention

class CompactAttentionBackend(AttentionBackend):
    """
    Attention backend enabling CompactFusion optimization for activation compression.
    """

    def create_attention(self, attn_type: str, sync_ulysses: bool, **kwargs) -> Callable:
        """
        Creates CompactFusion attention.
        
        Expected kwargs:
            compact_enabled (bool): Default True
            compact_fastpath (bool): Default True
            compact_residual (int): Default 1 (1st order)
            compact_rank (int): Default -1
            compact_quantized_cache (bool): Default False
        """
        print(f"Using CompactFusion XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
        
        # 1. Configuration Lifecycle
        # Default policy: Warmup for first 5 steps, then Int2
        def default_compress_func(layer_idx, step):
             if step is None:
                 step = 0
             if step < 5:
                  return COMPACT_COMPRESS_TYPE.WARMUP
             return COMPACT_COMPRESS_TYPE.INT2

        config = CompactConfig(
             enabled=kwargs.get("compact_enabled", True),
             fastpath=kwargs.get("compact_fastpath", True),
             residual=kwargs.get("compact_residual", 1),
             comp_rank=kwargs.get("compact_rank", -1),
             quantized_cache=kwargs.get("compact_quantized_cache", False),
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

        # 3. Instantiate Vendored Attention Class
        # This class (from attn_layer.py) automatically hooks into compact_fwd if config.enabled is True
        # Note: CompactxFuserLongContextAttention does not accept use_sync, we ignore sync_ulysses for now or map it if supported
        xfuser_attn = CompactxFuserLongContextAttention(attn_type=attn_enum, use_pack_qkv=False)

        # 4. Wrapper Function (same signature as Standard)
        def _attention_xfuser_compact_unmask(
                q, k, v, heads,
                join_q=None, join_k=None, join_v=None,
                mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False):

            if skip_reshape:
                b, _, _, dim_head = q.shape
                if join_q is not None:
                    j_b, _, _, j_dim_head = join_q.shape
            else:
                b, _, dim_head = q.shape
                dim_head //= heads
                q, k, v = map(
                    lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                    (q, k, v),
                )
                if join_q is not None:
                    assert join_k is not None and join_v is not None
                    j_b, _, j_dim_head = join_q.shape
                    j_dim_head //= heads
                    join_q, join_k, join_v = map(
                        lambda t: t.view(j_b, -1, heads, j_dim_head).transpose(1, 2),
                        (join_q, join_k, join_v),
                    )

            if mask is not None:
                 if mask.ndim == 2: mask = mask.unsqueeze(0)
                 if mask.ndim == 3: mask = mask.unsqueeze(1)
            
            if join_q is not None:
                assert join_k is not None and join_v is not None
                out = xfuser_attn(
                    None,
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    joint_strategy="rear",
                    joint_tensor_query=join_q.transpose(1, 2),
                    joint_tensor_key=join_k.transpose(1, 2),
                    joint_tensor_value=join_v.transpose(1, 2),
                ).transpose(1, 2)
            else:
                out = xfuser_attn(
                    None,
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                ).transpose(1, 2)
            
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            return out

        return _attention_xfuser_compact_unmask
