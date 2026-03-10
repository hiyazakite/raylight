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

def _get_pe_cache_key(pixel_coords_chunk, frame_rate, dtype):
    """Build a cache key from coord shapes, frame_rate, and dtype."""
    if isinstance(pixel_coords_chunk, (list, tuple)):
        shapes = tuple(tuple(p.shape) for p in pixel_coords_chunk)
    else:
        shapes = tuple(pixel_coords_chunk.shape)
    return (shapes, frame_rate, dtype)


def prepare_pe_cached(model, pixel_coords_chunk, frame_rate, input_dtype):
    """Compute positional embeddings with caching across timesteps.

    PEs depend only on spatial coords, frame_rate, and dtype — all of which are
    constant during a single generation run. Caching eliminates redundant PE
    computation on every denoising step.
    """
    cache_key = _get_pe_cache_key(pixel_coords_chunk, frame_rate, input_dtype)

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
    """
    if hasattr(block, "_original_get_ada_values"):
        return  # Already patched

    block._original_get_ada_values = block.get_ada_values
    block.get_ada_values = lambda *args, **kwargs: _caching_get_ada_values(block, *args, **kwargs)

    original_forward = block.forward

    def clearing_forward(*args, **kwargs):
        try:
            return original_forward(*args, **kwargs)
        finally:
            block._ada_cache = None

    block.forward = clearing_forward
