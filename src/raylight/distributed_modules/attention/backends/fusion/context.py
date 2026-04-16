from typing import Optional
from raylight.distributed_modules.attention.backends.fusion.utils import CompactConfig, CompactCache
from raylight.distributed_modules.attention.backends.fusion.patchpara.df_cache import AllGatherCache

_config: Optional[CompactConfig] = None
_cache: Optional[CompactCache] = None
# _step is now backed by the general denoising step counter in utils.py.
# compact_get_step / compact_set_step are kept for backward compatibility.
from raylight.distributed_modules.utils import get_denoising_step as _get_denoising_step, set_denoising_step as _set_denoising_step
_allgather_cache: Optional[AllGatherCache] = None
_current_cache_key: Optional[str] = None

def compact_config():
    global _config
    return _config

def compact_cache():
    global _cache
    return _cache

def compact_get_step():
    return _get_denoising_step()

def compact_set_step(step):
    _set_denoising_step(step)

def allgather_cache():
    global _allgather_cache
    return _allgather_cache

def compact_get_current_cache_key():
    global _current_cache_key
    return _current_cache_key

def compact_set_current_cache_key(key):
    global _current_cache_key
    _current_cache_key = key
