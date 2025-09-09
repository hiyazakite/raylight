from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType
_ATTN_TYPE = None


def set_attn_type(attn):
    global _ATTN_TYPE
    _ATTN_TYPE = attn


def get_attn_type():
    if _ATTN_TYPE is None:
        raise RuntimeError("_ATTN_TYPE is not initialized")
    else:
        return _ATTN_TYPE


def make_xfuser_attention(attn_type):
    print(f"Using XFuser {attn_type} attention")
    if attn_type.upper() == "FLASH_ATTN":
        attn = AttnType.FA
    elif attn_type.upper() == "FLASH_ATTN_3":
        attn = AttnType.FA3
    elif attn_type.upper() == "SAGE_AUTO_DETECT":
        attn = AttnType.SAGE_AUTO
    elif attn_type.upper() == "SAGE_FP16_TRITON":
        attn = AttnType.SAGE_FP16_TRITON
    elif attn_type.upper() == "SAGE_FP16_CUDA":
        attn = AttnType.SAGE_FP16
    elif attn_type.upper() == "SAGE_FP8_CUDA":
        attn = AttnType.SAGE_FP8
    elif attn_type.upper() == "SAGE_FP8_SM90":
        attn = AttnType.SAGE_FP8_SM90
    else:
        attn = AttnType.TORCH

    xfuser_attn = xFuserLongContextAttention(attn_type=attn)
    xfuser_attn_fallback = xFuserLongContextAttention(attn_type=AttnType.TORCH)

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
            out = xfuser_attn(
                None,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
        except Exception as e:
            print(f"XFuser {attn_type} failed, using XFuser Torch: {e}")
            out = xfuser_attn_fallback(
                None,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out
    return _attention_xfuser
