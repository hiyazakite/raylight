from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from raylight.config import RaylightConfig
import torch
from yunchang.kernels import AttnType
from ..layer import RaylightAttention
from ..interface import AttentionBackend
from ...sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

class StandardAttentionBackend(AttentionBackend):
    """
    The standard xFuser attention implementation.
    """

    def create_attention(self, raylight_config: 'RaylightConfig', **kwargs) -> Callable:
        # [SENIOR REFACTOR] Extract strategies from the Unified Config
        strat = raylight_config.strategy
        attn_type_name = strat.attention_type.name
        
        print(f"Using XFuser {attn_type_name} attention, Sync Ulysses: {strat.sync_ulysses}, Impl: {strat.ring_impl}")
        
        # Map our internal RaylightAttnType to yunchang's AttnType
        _yunchang_map = {
            "FLASH_ATTN": "FA",
            "FLASH_ATTN_3": "FA3",
            "SAGE_AUTO_DETECT": "SAGE_AUTO",
            "SAGE_FP16_CUDA": "SAGE_FP16",
            "SAGE_FP8_CUDA": "SAGE_FP8",
        }
        
        effective_name = _yunchang_map.get(attn_type_name, attn_type_name)
        
        try:
            attn_enum = AttnType[effective_name]
        except KeyError:
            attn_enum = AttnType.FA # Default fallback
        
        if attn_type_name == "SAGE_FP8_CUDA":
            ensure_hf_fp8_cuda_kernel()
        elif attn_type_name == "SAGE_FP8_SM90":
            ensure_hf_sm90_kernel()

        # Initialize the underlying Raylight attention class with the full config
        xfuser_attn = RaylightAttention(
            use_sync=strat.sync_ulysses, 
            attn_type=attn_enum, 
            ring_impl_type=strat.ring_impl,
            use_pack_qkv=False, 
            use_compact_ring=False,
            raylight_config=raylight_config
        )

        def _attention_xfuser_unmask(
                q,
                k,
                v,
                heads,
                join_q=None,
                join_k=None,
                join_v=None,
                mask=None,
                attn_precision=None,
                skip_reshape=False,
                skip_output_reshape=False,
                **kwargs):

            if skip_reshape:
                b, _, _, dim_head = q.shape
                if join_q is not None:
                    j_b, _, _, j_dim_head = join_q.shape
            else:
                b, _, dim_head = q.shape
                dim_head //= heads
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
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
            
            # Check if using join attention, for MMDiT model
            if join_q is not None:
                assert join_k is not None and join_v is not None
                out = xfuser_attn(
                    None,
                    q,
                    k,
                    v,
                    joint_strategy="rear",
                    joint_tensor_query=join_q,
                    joint_tensor_key=join_k,
                    joint_tensor_value=join_v,
                    mask=mask,
                    **kwargs
                )
            else:
                out = xfuser_attn(
                    None,
                    q,
                    k,
                    v,
                    mask=mask,
                    **kwargs
                )
            
            if not skip_output_reshape:
                out = (
                    out.reshape(b, -1, heads * dim_head)
                )
            return out

        return _attention_xfuser_unmask
