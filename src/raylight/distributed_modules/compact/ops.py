import torch
import torch.distributed as dist
import raylight.distributed_modules.compact.context as context
from raylight.distributed_modules.compact.utils import (
    COMPACT_COMPRESS_TYPE,
    ALLOW_DEPRECATED,
)
from raylight.distributed_modules.compact.prof import Profiler
from raylight.distributed_modules.compact.stats import stats_log
from raylight.distributed_modules.compact.slowpath import slowpath_compress, slowpath_decompress, sim_compress
from raylight.distributed_modules.compact.fastpath import (
    binary_quant_fastpath, 
    binary_dequant_fastpath, 
    int2_quant_fastpath, 
    int2_dequant_fastpath
)

@Profiler.prof_func("compact._compress_fn")
def _compress_fn(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, rank: int):
    if context._config and context._config.simulate_compress:
        # NOTE: if simulation enabled, directly return the simulated compress-then-decompress result
        return sim_compress(x, compress_type, (context._config.sparse_ratio if context._config.sparse_ratio is not None else 0), rank)

    return slowpath_compress(x, compress_type, rank=rank, sparse_ratio=(context._config.sparse_ratio if context._config and context._config.sparse_ratio is not None else 0))

            
@Profiler.prof_func("compact._decompress_fn")
def _decompress_fn(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, shape: tuple, rank: int):
    if context._config and context._config.simulate_compress:
        return x.view(shape)  # no need for further decompression
    return slowpath_decompress(x, shape, compress_type, rank=rank, sparse_ratio=(context._config.sparse_ratio if context._config and context._config.sparse_ratio is not None else 0))

def _compact_compress_fastpath(cache_key, x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, update_cache: bool, rank: int):
    assert compress_type == COMPACT_COMPRESS_TYPE.BINARY or compress_type == COMPACT_COMPRESS_TYPE.INT2
    assert context._config.compress_residual == 1
    # Base should be (N, C) now - returns None if shape mismatch or cache miss
    base = context._cache.get_base(cache_key, expected_shape=x.shape, expected_dtype=x.dtype) 
    
    if base is None:
        if update_cache:
            context._cache.put(cache_key, x, None)
        return x  # Return uncompressed tensor (warmup behavior)
    
    fastpath_func = binary_quant_fastpath if compress_type == COMPACT_COMPRESS_TYPE.BINARY else int2_quant_fastpath
    try:
        # Call with N, C layout
        q, scale_u_nk, scale_v_ck, new_base = fastpath_func(
            x_tensor_nc=x,
            base_tensor_nc=base,
            rank=rank,
            update_cache=update_cache,
        )
    except AssertionError as e:
        msg = f"[CompactFusion] Error (key='{cache_key}'): {str(e)}"
        print(msg)
        raise AssertionError(msg) from e
    # q: (N, C//8), scale_u_nk: (N, K), scale_v_ck: (C, K), new_base: (N, C)
    if update_cache:
        # new_base is already N, C
        context._cache.put(cache_key, new_base, None)

    q_flat_half = q.view(torch.half).flatten()
    u_flat_half = scale_u_nk.flatten() # New: U is (N, K)
    v_flat_half = scale_v_ck.flatten() # New: V is (C, K)
    compressed = torch.cat([q_flat_half, u_flat_half, v_flat_half], dim=0)

    if context._config.log_compress_stats:
        log_base = base # Already N, C
        log_recv_activation = new_base # Already N, C
        stats_log().log(
            cache_key,
            log_base,
            None,
            x,
            log_recv_activation,
            compressed,
            1,
        )
    return compressed

@Profiler.prof_func("compact.compact_compress")
def compact_compress(
    cache_key,
    x: torch.Tensor,
    compress_type: COMPACT_COMPRESS_TYPE,
    update_cache: bool = False,
):
    context._current_cache_key = cache_key
    assert x.is_contiguous()
    if not context._config or not context._config.enabled:
        return x
    
    original_shape = x.shape
    if len(x.shape) >= 4:
        x = x.view(-1, x.shape[-2] * x.shape[-1])
    elif len(x.shape) == 3:
        shape = (x.shape[0] * x.shape[1], x.shape[2])
        x = x.view(shape)
    assert x.ndim == 2

    # NOTE: rank check
    rank = context._config.comp_rank if context._config.comp_rank is not None else -1
    if compress_type == COMPACT_COMPRESS_TYPE.BINARY and rank != -1:
        assert ALLOW_DEPRECATED, "Binary compression with rank != -1 is deprecated"
    
    def cond_cache_put(key, val, delta):
        if update_cache and context._cache is not None:
            context._cache.put(key, val, delta)

    if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
        if context._config.fastpath:
            assert context._config.compress_residual == 1
            cond_cache_put(cache_key, x, None)
        else:
            if context._config.compress_residual == 1:
                cond_cache_put(cache_key, x, None)
            elif context._config.compress_residual == 2:
                base = context._cache.get_base(cache_key) if context._cache else None
                if base is None:
                    cond_cache_put(cache_key, x, None)
                else:
                    cond_cache_put(cache_key, x, x - base)
        return x.view(original_shape)
    else:
        if context._cache is None:
            return x.view(original_shape)

        if context._config.fastpath and (compress_type == COMPACT_COMPRESS_TYPE.BINARY or compress_type == COMPACT_COMPRESS_TYPE.INT2):
            compressed = _compact_compress_fastpath(cache_key, x, compress_type, update_cache, rank)
        else:
            if context._config.compress_residual == 0:
                compressed = _compress_fn(x, compress_type, rank)
                if context._config.log_compress_stats:
                    reconstructed_local = _decompress_fn(compressed, compress_type, x.shape, rank)
                    stats_log().log(
                        cache_key, 
                        base=None, 
                        delta_base=None, 
                        before_comp_activation=x, 
                        recv_activation=reconstructed_local, 
                        compressed_tensor=compressed, 
                        compress_residual=context._config.compress_residual,
                    )
            elif context._config.compress_residual == 1:
                base = context._cache.get_base(cache_key, expected_shape=x.shape, expected_dtype=x.dtype)
                if base is None:
                    cond_cache_put(cache_key, x, None)
                    return x.view(original_shape)
                delta = x - base
                compressed = _compress_fn(delta, compress_type, rank)
                recv_delta = _decompress_fn(compressed, compress_type, x.shape, rank)
                reconstructed = base + recv_delta
                cond_cache_put(cache_key, reconstructed if context._config.error_feedback else x, None)
                if context._config.log_compress_stats:
                    stats_log().log(
                        cache_key, 
                        base=base, 
                        delta_base=None, 
                        before_comp_activation=x, 
                        recv_activation=reconstructed, 
                        compressed_tensor=compressed, 
                        compress_residual=context._config.compress_residual,
                    )
            elif context._config.compress_residual == 2:
                base = context._cache.get_base(cache_key, expected_shape=x.shape, expected_dtype=x.dtype)
                if base is None:
                    cond_cache_put(cache_key, x, None)
                    return x.view(original_shape)
                delta_base = context._cache.get_delta_base(cache_key)
                delta_delta = x - (base + (delta_base if delta_base is not None else 0))
                compressed = _compress_fn(delta_delta, compress_type, rank)
                recv_delta_delta = _decompress_fn(compressed, compress_type, x.shape, rank)
                new_base = base + (delta_base if delta_base is not None else 0) + recv_delta_delta
                new_delta_base = (delta_base if delta_base is not None else 0) + recv_delta_delta
                cond_cache_put(
                    cache_key,
                    new_base,
                    _decay_delta_base(new_delta_base),
                )
                if context._config.log_compress_stats:
                    stats_log().log(
                        cache_key, 
                        base, 
                        delta_base, 
                        x, 
                        new_base, 
                        compressed, 
                        context._config.compress_residual,
                    )
                return compressed
            else:
                raise ValueError("Invalid compress_residual value")
        return compressed

def _decay_delta_base(delta_base):
    if context._config:
        return delta_base * context._config.delta_decay_factor
    return delta_base


def _compact_decompress_fastpath(cache_key, compressed: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, shape: tuple, update_cache: bool, rank: int):
    assert compress_type in [COMPACT_COMPRESS_TYPE.BINARY, COMPACT_COMPRESS_TYPE.INT2]
    assert context._config and context._config.compress_residual == 1

    N, C = shape
    assert C % (8 if compress_type == COMPACT_COMPRESS_TYPE.BINARY else 4) == 0

    ITEM_PER_BYTE = 8 if compress_type == COMPACT_COMPRESS_TYPE.BINARY else 4
    
    q_numel_uint8 = N * (C // ITEM_PER_BYTE)
    q_numel_half = q_numel_uint8 // 2
    u_numel_half = N * rank
    v_numel_half = C * rank

    expected_numel = q_numel_half + u_numel_half + v_numel_half
    assert compressed.numel() == expected_numel
    
    q_half, scale_u_flat, scale_v_flat = torch.split(
        compressed,
        [q_numel_half, u_numel_half, v_numel_half]
    )

    packed_ncx = q_half.view(torch.uint8).view(N, C // ITEM_PER_BYTE)
    scale_u_nk = scale_u_flat.view(N, rank)
    scale_v_ck = scale_v_flat.view(C, rank)

    base_nc = context._cache.get_base(cache_key, expected_shape=(N, C)) if context._cache else None

    fastpath_func = binary_dequant_fastpath if compress_type == COMPACT_COMPRESS_TYPE.BINARY else int2_dequant_fastpath
    reconstructed_nc = fastpath_func(
        packed=packed_ncx,
        scale_u_nk=scale_u_nk,
        scale_v_ck=scale_v_ck,
        base_nc=base_nc if base_nc is not None else torch.zeros_like(scale_u_nk.new_empty(N, C)),
    )
    if update_cache and context._cache:
        context._cache.put(cache_key, reconstructed_nc, None)
    return reconstructed_nc

@Profiler.prof_func("compact.compact_decompress")
def compact_decompress(
    cache_key,
    compressed: torch.Tensor,
    compress_type: COMPACT_COMPRESS_TYPE,
    shape: tuple,
    update_cache: bool = False,
):
    context._current_cache_key = cache_key
    if not context._config or not context._config.enabled:
        return compressed
    
    original_shape = shape
    if len(shape) >= 4:
        dim_0 = 1
        for i in range(len(shape) - 2):
            dim_0 *= shape[i]
        shape = (dim_0, shape[-2] * shape[-1])
    elif len(shape) == 3:
        shape = (shape[0] * shape[1], shape[2])
    else:
        assert len(shape) == 2

    def cond_cache_put(key, val, delta):
        if update_cache and context._cache:
            context._cache.put(key, val, delta)

    rank = context._config.comp_rank if context._config.comp_rank is not None else 1
    if rank == -1:
        rank = 1

    if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
        val = compressed.view(shape)
        if context._config.fastpath:
            assert context._config.compress_residual == 1
            cond_cache_put(cache_key, val, None)
        else:
            if context._config.compress_residual == 1:
                cond_cache_put(cache_key, val, None)
            elif context._config.compress_residual == 2:
                base = context._cache.get_base(cache_key) if context._cache else None
                if base is None:
                    cond_cache_put(cache_key, val, None)
                else:
                    cond_cache_put(cache_key, val, val - base)
        return val.view(original_shape)
    else:
        N, C = shape
        if compressed.numel() == N * C:
             val = compressed.view(original_shape)
             cache_val = val
             if cache_val.ndim > 2:
                 cache_val = cache_val.flatten(0, -2)
             cond_cache_put(cache_key, cache_val, None)
             return val

        if context._cache is None:
             return compressed.view(original_shape)

        if context._config.fastpath and (compress_type == COMPACT_COMPRESS_TYPE.BINARY or compress_type == COMPACT_COMPRESS_TYPE.INT2):
            reconstructed = _compact_decompress_fastpath(cache_key, compressed, compress_type, shape, update_cache, rank)
        else:
            if context._config.compress_residual == 0:
                reconstructed = _decompress_fn(compressed, compress_type, shape, rank)
            elif context._config.compress_residual == 1:
                base = context._cache.get_base(cache_key)
                recv_delta = _decompress_fn(compressed, compress_type, shape, rank)
                reconstructed = base + recv_delta if base is not None else recv_delta
                cond_cache_put(cache_key, reconstructed, None)
            elif context._config.compress_residual == 2:
                base = context._cache.get_base(cache_key)
                delta_base = context._cache.get_delta_base(cache_key)
                recv_delta_delta = _decompress_fn(compressed, compress_type, shape, rank)
                reconstructed = (base if base is not None else 0) + (delta_base if delta_base is not None else 0) + recv_delta_delta
                new_delta_base = (delta_base if delta_base is not None else 0) + recv_delta_delta
                cond_cache_put(cache_key, reconstructed, _decay_delta_base(new_delta_base))
            else:
                raise ValueError("Invalid compress_residual value")
        return reconstructed.view(original_shape)

def compact_all_gather(
    tag,
    x: torch.Tensor,
    comp_type: COMPACT_COMPRESS_TYPE,
    group=None,
):
    # raise NotImplementedError("Compact all gather is inconsistent with ring impl.")
    assert context._config and context._config.enabled
    rank = dist.get_rank(group)
    my_key = f"{tag}-{rank}"
    to_send = compact_compress(
        my_key,
        x,
        comp_type,
        update_cache=False,
    )
    world_size = dist.get_world_size(group)
    buf_list = [torch.empty_like(to_send) for _ in range(world_size)]
    with Profiler.scope("compact.all_gather"):
        dist.all_gather(buf_list, to_send, group=group, async_op=False)
    decompressed_list = [
        compact_decompress(
            f"{tag}-{i}",
            buf,
            comp_type,
            x.shape,
            update_cache=True,
        )
        for i, buf in enumerate(buf_list)
    ]
    return decompressed_list
