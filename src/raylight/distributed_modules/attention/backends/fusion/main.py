import torch.distributed as dist
from raylight.distributed_modules.attention.backends.fusion.utils import (
    CompactConfig,
    CompactCache,
)
from raylight.distributed_modules.attention.backends.fusion.stats import stats_hello
from raylight.distributed_modules.attention.backends.fusion.patchpara.df_cache import AllGatherCache
import raylight.distributed_modules.attention.backends.fusion.patchpara.fwd as fwd_module

import raylight.distributed_modules.attention.backends.fusion.context as context

"""
COMPACT: Activation Compression with Delta Transmission and Error Feedback

In diffusion models, activations change only slightly from one denoising step to the next.
Simply transmitting the full activations incures high redundancy.
Rather than transmitting full activations between nodes, 
we transmit only the difference (delta) between the current activation and a maintained baseline. 
This delta is highly compressible.

Our approach works as follows:
1. Compute the delta as the difference between the current activation and a cached baseline.
2. Compress and transmit the delta.
3. At the receiver, decompress the delta and add it to the baseline to reconstruct the activation.
4. Both sender and receiver update the baseline with the same reconstructed activation, so that any error is compensated in future.

This error feedback mechanism ensures that, over time, the compression errors do not accumulate,
leading to accurate reconstruction while significantly reducing communication overhead.
"""


def compact_init(config: CompactConfig):
    context._config = config
    # Initialize cache using flags from the provided config
    context._cache = CompactCache(
        quantize=config.quantized_cache,
        quant_bits=config.cache_quant_bits if config.cache_quant_bits is not None else 8,
        allow_deprecated=config.allow_deprecated,
    )
    context._step = None
    if config.override_with_patch_gather_fwd:
        context._allgather_cache = AllGatherCache()
    
    context._current_cache_key = None

def compact_hello():
    pass

def compact_config():
    return context.compact_config()

def compact_set_step(step):
    # Reset the global attention layer index counter at the start of each step.
    # This prevents the cache from growing indefinitely across sampling steps (OOM).
    try:
        from raylight.distributed_modules.attention.layer import reset_attn_layer_idx
        reset_attn_layer_idx()
    except (ImportError, AttributeError):
        pass
    # Invalidate the mask triviality cache so that masks reused across steps
    # are re-checked with fresh content.
    try:
        from raylight.distributed_modules.attention import invalidate_mask_cache
        invalidate_mask_cache()
    except (ImportError, AttributeError):
        pass
    context.compact_set_step(step)

def compact_get_step():
    return context.compact_get_step()

def compact_cache():
    return context.compact_cache()

def allgather_cache():
    return context.allgather_cache()

def compact_reset():
    # Reset the global attention layer index counter in the centralized layer module.
    # If this is not reset between generation steps, the cache keys keep incrementing
    # infinitely causing massive cache leaks (OOM).
    try:
        from raylight.distributed_modules.attention.layer import reset_attn_layer_idx
        reset_attn_layer_idx()
    except (ImportError, AttributeError):
        pass

    # Clear mask triviality cache between generation runs.
    try:
        from raylight.distributed_modules.attention import invalidate_mask_cache
        invalidate_mask_cache()
    except (ImportError, AttributeError):
        pass

    # Clear the attention callable cache so that any env-var config changes
    # (e.g. switching COMPACT compression type) take effect on the next run.
    try:
        from raylight.distributed_modules.attention import invalidate_attn_cache
        invalidate_attn_cache()
    except (ImportError, AttributeError):
        pass

    # P2: Clear the subspace_iter Q-matrix cache so stale warm-starts from
    # a prior generation don't pollute the new run.
    try:
        from raylight.distributed_modules.attention.backends.fusion.compress_lowrank import invalidate_q_cache
        invalidate_q_cache()
    except (ImportError, AttributeError):
        pass

    if context._config is None:
        return

    # Explicitly drop old cache references before creating new ones to aid GC
    old_cache = context._cache
    context._cache = CompactCache(
        quantize=context._config.quantized_cache, 
        quant_bits=(context._config.cache_quant_bits if context._config.cache_quant_bits is not None else 8),
        allow_deprecated=context._config.allow_deprecated,
    )
    del old_cache

    from raylight.distributed_modules.attention.backends.fusion.stats import stats_clear
    stats_clear()
    try:
        from raylight.distributed_modules.attention.backends.fusion.prof import Profiler
        Profiler.instance().reset()
    except:
        pass
    context._step = None

    # Always clear AllGatherCache and patchpara buffers, even if override flag
    # is currently off. Stale buffers from a prior config would otherwise persist.
    if context._allgather_cache is not None:
        context._allgather_cache.clear()
        context._allgather_cache = None
    if context._config.override_with_patch_gather_fwd:
        context._allgather_cache = AllGatherCache()
    fwd_module.clear_buffers()

    context._current_cache_key = None
    
    # Reset global attention layer index counter in the centralized layer module
    try:
        from raylight.distributed_modules.attention.layer import reset_attn_layer_idx
        reset_attn_layer_idx()
        
        # ALSO: Clear GGUF dequantization cache to prevent VRAM accumulation
        try:
            from raylight.expansion.comfyui_gguf.ops import GGMLLayer
            GGMLLayer.clear_dequant_cache()
        except (ImportError, AttributeError):
            pass
    except (ImportError, AttributeError):
        pass

def compact_get_current_cache_key():
    """
    FOR TESTING ONLY
    """
    return context._current_cache_key

from raylight.distributed_modules.attention.backends.fusion.ops import (
    compact_compress,
    compact_decompress,
    compact_all_gather,
)


def compact_evict_cuda_caches() -> int:
    """Free CompactFusion CUDA caches under memory pressure.

    Called by the CUDA malloc interceptor (via MemoryPolicy) when an
    allocation fails.  Frees communication buffers and cached baselines
    that can be cheaply rebuilt (the next attention call will fall back
    to warmup / uncompressed behaviour for one step).

    Returns approximate bytes freed.
    """
    import torch
    freed = 0

    # 1. AllGatherCache recv/send buffers
    if context._allgather_cache is not None:
        try:
            freed += context._allgather_cache.tensors_size()
            context._allgather_cache.clear()
        except Exception:
            pass

    # 2. PatchPara static comm buffers
    try:
        for _key, buf_list in fwd_module._buffers.items():
            if isinstance(buf_list, list):
                for t in buf_list:
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        freed += t.nelement() * t.element_size()
        fwd_module.clear_buffers()
    except Exception:
        pass

    if freed > 0:
        print(
            f"[CompactFusion] Evicted {freed / 1024**2:.0f} MB "
            f"of CUDA comm caches under memory pressure"
        )

    return freed