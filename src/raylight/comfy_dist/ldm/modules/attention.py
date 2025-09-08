import logging
import torch

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType
from comfy import model_management


def fa_attention_xfuser(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    try:
        assert mask is None
        out = xFuserLongContextAttention(attn_type=AttnType.FA)(
            None,
            q,
            k,
            v,
        ).flatten(2)
    except Exception as e:
        logging.warning(f"Flash Attention failed, using default SDPA: {e}")
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out


xfuser_optimized_attention = fa_attention_xfuser
xfuser_optimized_attention_masked = xfuser_optimized_attention
