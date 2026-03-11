from raylight.distributed_modules.compact.utils import CompactConfig, CompactCache
from raylight.distributed_modules.compact.patchpara.df_cache import AllGatherCache

_config: CompactConfig = None
_cache: CompactCache = None
_step: int = None
_allgather_cache: AllGatherCache = None
_current_cache_key: str = None

def compact_config():
    global _config
    return _config

def compact_cache():
    global _cache
    return _cache

def compact_get_step():
    global _step
    return _step

def compact_set_step(step):
    global _step
    _step = step

def allgather_cache():
    global _allgather_cache
    return _allgather_cache

def compact_get_current_cache_key():
    global _current_cache_key
    return _current_cache_key
