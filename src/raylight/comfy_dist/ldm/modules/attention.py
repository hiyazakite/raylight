import logging
import torch

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType
from comfy import model_management


def make_attention_xfuser(attn_type):
    attention_xfuser = xFuserLongContextAttention(attn_type=attn_type)

    def _attention_xfuser(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        try:
            assert mask is None
            out = attention_xfuser(
                None,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2).flatten(2)
        except Exception as e:
            logging.warning(f"Flash Attention failed, using default SDPA: {e}")
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out

    return _attention_xfuser


xfuser_optimized_attention = make_attention_xfuser(AttnType.TORCH)

if model_management.sage_attention_enabled():
    logging.info("Using sage attention")
    xfuser_optimized_attention = make_attention_xfuser(AttnType.SAGE_AUTO)
elif model_management.flash_attention_enabled():
    logging.info("Using Flash Attention")
    xfuser_optimized_attention = make_attention_xfuser(AttnType.FA)

xfuser_optimized_attention_masked = xfuser_optimized_attention
