from typing import Callable
import torch
from yunchang.kernels import AttnType
from ..layer import RaylightAttention
from ..interface import AttentionBackend
from ...sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

class StandardAttentionBackend(AttentionBackend):
    """
    The standard xFuser attention implementation.
    """

    def create_attention(self, attn_type: str, sync_ulysses: bool, **kwargs) -> Callable:
        print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
        
        attn_enum = AttnType[attn_type]
        
        if attn_type == "SAGE_FP8_CUDA":
            ensure_hf_fp8_cuda_kernel()
        elif attn_type == "SAGE_FP8_SM90":
            ensure_hf_sm90_kernel()

        # Initialize the underlying Raylight attention class
        xfuser_attn = RaylightAttention(
            use_sync=sync_ulysses, 
            attn_type=attn_enum, 
            use_pack_qkv=False, 
            use_compact_ring=False  # Standard uses xfuser ring or patched flash
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
