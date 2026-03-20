from typing import Callable, TYPE_CHECKING
from yunchang.kernels import AttnType
from ..interface import AttentionBackend
from ..layer import RaylightAttention
from ...sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

if TYPE_CHECKING:
    from raylight.config import RaylightConfig

# Vendored Compact Modules
from raylight.distributed_modules.attention.backends.fusion.main import compact_init, compact_hello
from raylight.distributed_modules.attention.backends.fusion.utils import CompactConfig as FusionCompactConfig, COMPACT_COMPRESS_TYPE

class CompactAttentionBackend(AttentionBackend):
    """
    Attention backend enabling CompactFusion optimization for activation compression.
    """

    def create_attention(self, raylight_config: 'RaylightConfig', **kwargs) -> Callable:
        # [SENIOR REFACTOR] Leverage Unified Config for CompactFusion
        strat = raylight_config.strategy
        compact_strat = raylight_config.compact
        
        print(f"Using CompactFusion XFuser {strat.attention_type.name} attention, Sync Ulysses: {strat.sync_ulysses}, Impl: {strat.ring_impl}")
        
        # 1. Configuration Lifecycle
        def default_compress_func(layer_idx, step):
             if step is None:
                 step = 0
             if step < compact_strat.warmup_steps:
                  return COMPACT_COMPRESS_TYPE.WARMUP
             # Map Raylight enum to backend enum (using name matching)
             try:
                 return COMPACT_COMPRESS_TYPE(compact_strat.delta_compression.name.lower())
             except ValueError:
                 return COMPACT_COMPRESS_TYPE.BINARY

        # Instantiate the vendored CompactConfig from the fusion backend
        
        backend_config = FusionCompactConfig(
             enabled=compact_strat.enabled,
             fastpath=kwargs.get("compact_fastpath", True),
             residual=kwargs.get("compact_residual", 1),
             comp_rank=kwargs.get("compact_rank", -1),
             quantized_cache=compact_strat.kv_cache_quant_enable,
             cache_quant_bits=compact_strat.kv_cache_quant_bits,
             ef=True, # Error Feedback required for fastpath/residual
             compress_func=default_compress_func,
             # [NEW] Propriety flags from Unified Config
             verbose_attn=raylight_config.debug.verbose_attn,
             overlap_decomp=raylight_config.debug.overlap_decomp,
             use_awl=raylight_config.debug.use_awl,
             allow_deprecated=raylight_config.debug.allow_deprecated,
        )
        
        compact_init(backend_config)
        compact_hello()
        
        # 2. Kernel Setup
        # Map our internal RaylightAttnType to yunchang's AttnType
        try:
            attn_enum = AttnType[strat.attention_type.name]
        except KeyError:
            attn_enum = AttnType.FA
            
        if strat.attention_type.name == "SAGE_FP8_CUDA":
            ensure_hf_fp8_cuda_kernel()
        elif strat.attention_type.name == "SAGE_FP8_SM90":
            ensure_hf_sm90_kernel()

        # 3. Instantiate Centralized Attention Class
        xfuser_attn = RaylightAttention(
            use_sync=strat.sync_ulysses,
            attn_type=attn_enum, 
            ring_impl_type=strat.ring_impl,
            use_pack_qkv=False,
            use_compact_ring=True,
            raylight_config=raylight_config
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
