from comfy.cli_args import args

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

from yunchang.kernels import AttnType


def make_xfuser_attention():
    args.use_flash_attention = True
    if args.use_sage_attention:
        attn_type = AttnType.SAGE_AUTO
        print(f"Using XFuser {attn_type} attention")
    elif args.use_flash_attention:
        attn_type = AttnType.FA
        print(f"Using XFuser {attn_type} attention")
    else:
        attn_type = AttnType.TORCH
        print(f"Using XFuser {attn_type} attention")
        print(args)

    xfuser_attn = xFuserLongContextAttention(attn_type)
    xfuser_attn_fallback = xFuserLongContextAttention(AttnType.TORCH)

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


xfuser_optimized_attention = make_xfuser_attention()
xfuser_optimized_attention_masked = xfuser_optimized_attention
