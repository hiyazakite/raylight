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
    # Flush dequant cache at step boundaries — the underlying quantized data
    # is about to be overwritten by new compress/put calls, so stale dequant
    # entries must be dropped.
    if context._cache is not None and hasattr(context._cache, '_dequant_cache'):
        context._cache._dequant_cache.clear()

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