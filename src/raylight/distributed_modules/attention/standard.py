from typing import Callable
from yunchang.kernels import AttnType
from raylight.distributed_modules.compact.attn_layer import xFuserLongContextAttention
from .interface import AttentionBackend
from ..sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

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

        # Initialize the underlying xFuser attention class
        xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn_enum, use_pack_qkv=False, use_compact_ring=False)

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
                # OPT 3: Reshape directly to (B, S, H, D) — xfuser_attn expects this layout.
                # Previously did view→transpose(1,2)→(B,H,S,D) then transpose(1,2) back.
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
                    q if not skip_reshape else q.transpose(1, 2),
                    k if not skip_reshape else k.transpose(1, 2),
                    v if not skip_reshape else v.transpose(1, 2),
                    joint_strategy="rear",
                    joint_tensor_query=join_q if not skip_reshape else join_q.transpose(1, 2),
                    joint_tensor_key=join_k if not skip_reshape else join_k.transpose(1, 2),
                    joint_tensor_value=join_v if not skip_reshape else join_v.transpose(1, 2),
                    mask=mask,
                    **kwargs
                )
            else:
                out = xfuser_attn(
                    None,
                    q if not skip_reshape else q.transpose(1, 2),
                    k if not skip_reshape else k.transpose(1, 2),
                    v if not skip_reshape else v.transpose(1, 2),
                    mask=mask,
                    **kwargs
                )
            
            if not skip_output_reshape:
                out = (
                    out.reshape(b, -1, heads * dim_head)
                )
            return out

        return _attention_xfuser_unmask
