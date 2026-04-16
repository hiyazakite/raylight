"""LTXAV model-level performance optimizations.

These patches are independent of sequence parallelism and can benefit both
distributed and single-GPU execution:

- PE caching: reuse positional embeddings across denoising timesteps
- AdaLN caching: batch redundant get_ada_values computations per block
- Debug gating: gate CUDA-syncing debug output behind RAYLIGHT_DEBUG=1
"""

import logging
import os



logger = logging.getLogger(__name__)
RAYLIGHT_DEBUG = os.environ.get("RAYLIGHT_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# Positional Embedding Caching
# ---------------------------------------------------------------------------

def _get_pe_cache_key(pixel_coords_chunk, frame_rate, dtype, ref_audio_seq_len=0):
    """Build a cache key from coord shapes, values, frame_rate, and dtype."""
    if isinstance(pixel_coords_chunk, (list, tuple)):
        shapes = tuple(tuple(p.shape) for p in pixel_coords_chunk)
    else:
        shapes = tuple(pixel_coords_chunk.shape)
    
    # We include ref_audio_seq_len specifically as a proxy for the coordinate shift
    # introduced by ID-LoRA.
    return (shapes, frame_rate, dtype, ref_audio_seq_len)


def prepare_pe_cached(model, pixel_coords_chunk, frame_rate, input_dtype):
    """Compute positional embeddings with caching across timesteps.

    PEs depend only on spatial coords, frame_rate, and dtype — all of which are
    constant during a single generation run. Caching eliminates redundant PE
    computation on every denoising step.
    """
    ref_audio_seq_len = getattr(model, "_ref_audio_seq_len", 0)
    cache_key = _get_pe_cache_key(pixel_coords_chunk, frame_rate, input_dtype, ref_audio_seq_len)

    cached = getattr(model, "_raylight_pe_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    pe = model._prepare_positional_embeddings(pixel_coords_chunk, frame_rate, input_dtype)
    model._raylight_pe_cache = (cache_key, pe)
    return pe


# ---------------------------------------------------------------------------
# AdaLN Caching
# ---------------------------------------------------------------------------

def _make_ada_cache_key(table, timestep):
    """Build a cache key for an (ada_table, timestep) pair.

    Uses data_ptr + shape so that views into the same underlying parameter
    (e.g. table[:4, :] vs table[4:, :]) get separate cache entries, while
    repeated calls with the exact same view hit the cache.
    """
    if hasattr(timestep, "num_frames"):  # CompressedTimestep
        ts_key = (id(timestep), timestep.num_frames, timestep.patches_per_frame)
    else:
        ts_key = (table.data_ptr(), timestep.data_ptr(), timestep.shape)
    return (table.data_ptr(), tuple(table.shape), ts_key)


def _caching_get_ada_values(self, scale_shift_table, batch_size, timestep, indices=slice(None, None)):
    """Drop-in replacement for BasicAVTransformerBlock.get_ada_values with caching.

    First call for a given (table, timestep) computes ALL ada values and caches
    them. Subsequent calls with different ``indices`` return slices from the
    cache, avoiding redundant reshape/add/unbind operations.
    """
    cache = getattr(self, "_ada_cache", None)
    if cache is None:
        cache = {}
        self._ada_cache = cache

    key = _make_ada_cache_key(scale_shift_table, timestep)
    if key not in cache:
        # Compute FULL result (all indices) and cache it
        cache[key] = self._original_get_ada_values(
            scale_shift_table, batch_size, timestep
        )

    full_result = cache[key]

    # Apply the requested indices slice to the cached tuple
    if indices == slice(None, None):
        return full_result
    return full_result[indices]


def patch_ada_caching(block):
    """Monkey-patch a BasicAVTransformerBlock to cache get_ada_values results.

    Also wraps forward() to clear the cache after each block execution so
    stale tensors are not retained between timesteps.

    IMPORTANT: This function is called on every ON_LOAD callback (i.e. every
    denoising step in lowvram mode). Both patches MUST be fully idempotent.
    """
    if not hasattr(block, "_original_get_ada_values"):
        block._original_get_ada_values = block.get_ada_values
        block.get_ada_values = lambda *args, **kwargs: _caching_get_ada_values(block, *args, **kwargs)

    # Guard block.forward wrapping separately — if already wrapped, skip to
    # avoid stacking clearing_forward closures on every denoising step.
    if not hasattr(block, "_ada_cache_forward_patched"):
        original_forward = block.forward

        def clearing_forward(*args, **kwargs):
            try:
                return original_forward(*args, **kwargs)
            finally:
                block._ada_cache = None

        block.forward = clearing_forward
        block._ada_cache_forward_patched = True


# ---------------------------------------------------------------------------
# Cross-Attention K/V Projection Cache
# ---------------------------------------------------------------------------

def get_cross_kv_cached(attn, context):
    """Cache K/V projections (including K normalization) for constant context.

    For text cross-attention in LTX models, the text context does not change
    across denoising steps of a single generation.  Caching to_k + k_norm +
    to_v eliminates 3 operations per block per step after the first step.

    Cache invalidation is step-counter based, NOT data_ptr based.

    Why not data_ptr: _prepare_context calls caption_projection(context) which
    allocates a NEW tensor on every denoising step.  With TP=2, each rank is a
    separate process with an independent CUDA allocator.  Python's memory
    recycler can hand the same virtual address to one rank's new context while
    the other rank gets a fresh address — producing a cache HIT on one rank and
    a MISS on the other.  That asymmetry skips k_norm's all_reduce on one side
    only, which deadlocks NCCL.

    Why step-counter is safe: the step counter (set_denoising_step) is updated by
    the sampling callback which runs AFTER each NCCL-synchronized denoising
    step.  Both TP ranks always have the same step value because they cannot
    advance past a denoising step without completing the same NCCL calls.

    Returns (k_normed, v) — k already has k_norm applied, so the caller must
    NOT apply k_norm again on cache hits.  This is signalled by the _cache_kv
    flag on the attention module.

    IMPORTANT: Only call for cross-attention to a constant context (attn2 /
    audio_attn2).  Do NOT call for self-attention or AV cross-attention whose
    context changes every step.
    """
    from raylight.distributed_modules.utils import get_denoising_step
    step = get_denoising_step()

    # Use cached K/V for step > 0 only.
    # step == 0: always recompute — refreshes cache for new generation AND ensures
    #            all TP ranks call k_norm's all_reduce symmetrically.
    # step == None: compact not initialised, treat as step 0 (recompute, safe).
    cache = getattr(attn, '_cross_kv_cache', None)
    if cache is not None and step is not None and step != 0:
        return cache[0], cache[1]

    k = attn.k_norm(attn.to_k(context))
    v = attn.to_v(context)
    attn._cross_kv_cache = (k, v)
    return k, v
