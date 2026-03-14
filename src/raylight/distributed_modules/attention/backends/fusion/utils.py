import torch
import torch.distributed as dist
from raylight.distributed_modules.attention.backends.fusion.prof import Profiler
from enum import Enum
from raylight.distributed_modules.attention.backends.fusion.patchpara.df_utils import PatchConfig
import os

ALLOW_DEPRECATED = os.environ.get("COMPACT_ALLOW_DEPRECATED", "0") == "1"

class COMPACT_COMPRESS_TYPE(Enum):
    """
    Enumeration of compression types for compact communication.

    SPARSE: Uses top-k sparsity to compress tensors
    QUANT: Uses quantization to compress tensors
    HYBRID: Combines topk sparsity and quantization for compression
    """

    WARMUP = "warmup"
    SPARSE = "sparse"
    BINARY = "binary"
    INT2 = "int2"
    INT2_MINMAX = "int2-minmax"
    INT4 = "int4"
    IDENTITY = "identity"  # go thorugh the entire pipeline, but no compression
    LOW_RANK = "low-rank"
    LOW_RANK_Q = "low-rank-int4"
    LOW_RANK_AWL = "low-rank-awl" # attn aware lowrank


class CompactConfig:

    def __init__(
        self,
        enabled: bool = False,
        override_with_patch_gather_fwd: bool = False,
        patch_gather_fwd_config: PatchConfig = None,
        compress_func: callable = None,
        sparse_ratio=None,
        comp_rank=None,
        residual: int = 0,
        ef: bool = False,
        simulate: bool = False,
        log_stats: bool = False,
        check_consist: bool = False,
        fastpath: bool = False,
        quantized_cache: bool = True,
        cache_quant_bits: int = 8,
        delta_decay_factor: float | None = None
    ):
        """
        Initialize compression settings.
        Args:
            enabled (bool): Enable/disable compression.
            compress_func (callable): (layer_idx, step) -> compress_type, step starts from 0.
            residual (int): 0: no residual, 1: 1st order residual, 2: 2nd order residual.
            ef (bool): Enable/disable EF compression.
            simulate (bool): Enable/disable simulation compression.
            log_stats (bool): Enable/disable logging of compression stats.
            quantized_cache (bool): Enable quantization for base tensor in CompactCache.
            delta_decay_factor (float): Decay factor applied to delta_base in 2nd order residual.
        """
        self.enabled = enabled
        self.compress_func = compress_func
        self.sparse_ratio = sparse_ratio
        self.comp_rank = comp_rank
        assert residual in [0, 1, 2]
        self.compress_residual = residual
        self.error_feedback = ef
        self.simulate_compress = simulate
        # STATS related
        self.log_compress_stats = log_stats
        self.check_cache_consistency = check_consist
        self.fastpath = fastpath
        # Cache behavior flags
        self.quantized_cache = quantized_cache
        self.cache_quant_bits = None
        # Updated attributes
        self.delta_decay_factor = delta_decay_factor
        
        self.override_with_patch_gather_fwd = override_with_patch_gather_fwd
        self.patch_gather_fwd_config = patch_gather_fwd_config
        
        
        if residual == 0:
            assert not ef, "No residual does not support error feedback."
        if residual == 2:
            assert ef, "2nd order compression requires error feedback enabled."
        if self.fastpath:
            assert ef, "Fastpath requires error feedback enabled."
            assert not simulate, "Fastpath does not support simulation."
            assert residual == 1, "Fastpath requires 1st order residual."

        if residual == 2:
            assert ef, "2nd order compression requires error feedback enabled."
        if self.fastpath:
            assert ef, "Fastpath requires error feedback enabled."
            assert not simulate, "Fastpath does not support simulation."
            assert not simulate, "Fastpath does not support simulation."
        
        if self.quantized_cache:
            if cache_quant_bits is None:
                 # Default to 8-bit for backward compatibility/safety if not specified
                 self.cache_quant_bits = 8
            else:
                 self.cache_quant_bits = cache_quant_bits
            assert self.cache_quant_bits in [4, 8], "Only 4-bit and 8-bit cache quantization are supported."
        else:
            self.cache_quant_bits = None

        if self.override_with_patch_gather_fwd:
            assert self.enabled, "Compact must be enabled if override_with_patch_gather_fwd is True"
            assert self.patch_gather_fwd_config is not None, "patch_gather_fwd_config must be set if override_with_patch_gather_fwd is True"
            if self.patch_gather_fwd_config.use_compact:
                assert not self.patch_gather_fwd_config.async_comm, "Compact does not support async communication"
            elif self.patch_gather_fwd_config.async_comm:
                assert not self.patch_gather_fwd_config.use_compact, "Async communication does not support compression"
        else:
            assert self.patch_gather_fwd_config is None, "patch_gather_fwd_config must be None if override_with_patch_gather_fwd is False"

    def get_compress_type(self):
        """
        For naming the result file.
        """
        if self.compress_func is None or not self.enabled:
            return "NO_COMPACT"
        compress_type = self.compress_func(0, 4)
        if isinstance(compress_type, COMPACT_COMPRESS_TYPE):
            return compress_type.name
        return str(compress_type)

from raylight.distributed_modules.attention.backends.fusion.compress_quantize import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4
from raylight.distributed_modules.attention.backends.fusion.compress_lowrank import subspace_iter


class CompactCache:
    def __init__(self, quantize=True, quant_bits=8):
        self.quantize = quantize
        self.quant_bits = quant_bits
        self.base = {}
        self.delta_base = {}
        if quantize:
            assert ALLOW_DEPRECATED or quant_bits in [4, 8]
        self.passed_count = 0

    # @Profiler.prof_func("compact.CompactCache.put")
    def put(self, key, base, delta_base):
        # Quantize base if needed
        if self.quantize:
            if self.quant_bits == 8:
                base = quantize_int8(base)
            elif self.quant_bits == 4:
                if base.ndim == 2 and (base.shape[0] % 2) != 0:
                    # INT4 packing stores two rows per byte along N, so odd N needs
                    # one extra row. Duplicate the last row so min/max statistics stay
                    # unchanged, then remember the original length for cropping.
                    padded_base = torch.cat([base, base[-1:].contiguous()], dim=0)
                    packed, scale, min_val = quantize_int4(padded_base)
                    base = (packed, scale, min_val, base.shape[0])
                else:
                    base = quantize_int4(base)
        self.base[key] = base
        # from raylight.distributed_modules.attention.backends.fusion.main import compact_get_step
        # from raylight.distributed_modules.collector.collector import collect
        # if "k" in key:
        #     collect(base, "kbase", compact_get_step(), int(key.split("-")[0]))
        # elif "v" in key:
        #     collect(base, "vbase", compact_get_step(), int(key.split("-")[0]))
        # Compress or store delta_base
        if delta_base is not None:
            self.delta_base[key] = delta_base
        else:
            self.delta_base[key] = None

    # @Profiler.prof_func("compact.CompactCache.get_base")
    def get_base(self, key, expected_shape: tuple = None, expected_dtype: torch.dtype = None):
        base = self.base.get(key, None)
        if self.quantize:
            if base is not None:
                if self.quant_bits == 8:
                    base = dequantize_int8(*base)
                elif self.quant_bits == 4:
                    if isinstance(base, tuple) and len(base) == 4:
                        packed, scale, min_val, orig_n = base
                        base = dequantize_int4(packed, scale, min_val)[:orig_n]
                    else:
                        base = dequantize_int4(*base)
        
        if base is not None:
            # Check shape
            if expected_shape is not None and base.shape != expected_shape:
                return None
            # Check dtype
            if expected_dtype is not None and base.dtype != expected_dtype:
                return None
        return base

    # @Profiler.prof_func("compact.CompactCache.get_delta_base") 
    def get_delta_base(self, key):
        # Retrieve stored item for delta_base
        stored_item = self.delta_base.get(key, None)
        return stored_item

    def check_consistency(self, group=None):
        """
        Checks cache consistency for all keys across all GPUs in the specified group.
        Args:
            group: Optional process group to check consistency within. If None, uses the default world group.
        """
        if group is None:
            group = dist.group.WORLD
        world_size = dist.get_world_size(group)
        if world_size <= 1:
            return # No need for consistency check with a single process
        # Iterate through all keys present in the local cache
        # Assumes all ranks have the same keys
        for key in sorted(self.base.keys()):
            local_base = self.get_base(key)
            # Reconstruct/retrieve delta_base before checking
            local_delta_base = self.get_delta_base(key)

            # Flatten and concatenate tensors if they exist
            tensors_to_check = []
            if local_base is not None:
                tensors_to_check.append(local_base.flatten())
            if local_delta_base is not None:
                tensors_to_check.append(local_delta_base.flatten())
            
            if tensors_to_check:
                # Concatenate all tensors into a single flat tensor
                combined_tensor = torch.cat(tensors_to_check)
                tensor_to_reduce = combined_tensor.clone().detach().float()
                dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM, group=group)
                average_tensor = tensor_to_reduce / world_size
                assert torch.allclose(combined_tensor.float(), average_tensor, atol=1e-2), f'Inconsistent cache at key {key}, max diff: {torch.max(torch.abs(combined_tensor.float() - average_tensor)):.6f}'
        self.passed_count += 1


def get_emoji():
    import random
    emojis = [
        "☝️ 😅",
        "👊🤖🔥",
        "🙏 🙏 🙏",
        "🐳🌊🐚",
        "☘️ ☘️ 🍀",
        "🎊🎉🎆",
        "🌇🌆🌃",
        "🍾🍾🍾",
        "🅰  🅲  🅲  🅴  🅿  🆃  🅴  🅳",
        "🖼️ 🖌️ 🎨",
        "🐳  🅲  🅾  🅼  🅿  🅰  🅲  🆃",
        "╰(*°▽°*)╯",
        "ヾ(≧▽≦*)o",
        "⚡️ 🔗 ⚡️",
        "💾 ➡️ 🚀"
    ]
    return random.choice(emojis)