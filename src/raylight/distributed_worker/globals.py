# From XDiT, mochi-xdit project
# https://github.com/xdit-project/mochi-xdit
from typing import Tuple


_USE_XDIT = False


def set_use_xdit(use_dit: bool) -> None:
    """Set whether to use DIT model.

    Args:
        use_dit: Boolean flag indicating whether to use xdur
    """
    global _USE_XDIT
    _USE_XDIT = use_dit
    print(f"The xDiT flag use_xdit={use_dit}")


def is_use_xdit() -> bool:
    return _USE_XDIT


_ULYSSES_DEGREE = None
_RING_DEGREE = None
_CFG_PARALLEL = None


def set_usp_config(ulysses_degree: int, ring_degree: int, cfg_parallel: bool) -> None:
    global _ULYSSES_DEGREE, _RING_DEGREE, _CFG_PARALLEL
    _ULYSSES_DEGREE = ulysses_degree
    _RING_DEGREE = ring_degree
    _CFG_PARALLEL = cfg_parallel
    print(f"Now we use xdit with ulysses degree {ulysses_degree}, ring degree {ring_degree}, and CFG parallel {cfg_parallel}")


def get_usp_config() -> Tuple[int, int, bool]:
    return _ULYSSES_DEGREE, _RING_DEGREE, _CFG_PARALLEL


_USE_FSDP = False


def set_use_fsdp(use_fsdp: bool) -> None:

    global _USE_FSDP
    _USE_FSDP = use_fsdp
    print(f"The FSDP flag use_fsdp={use_fsdp}")


def is_use_fsdp() -> bool:
    return _USE_FSDP
