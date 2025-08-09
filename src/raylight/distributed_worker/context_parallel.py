# From XDiT, mochi-xdit project
# https://github.com/xdit-project/mochi-xdit
import torch
import torch.distributed as dist

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_RANK = None
_CONTEXT_PARALLEL_GROUP_SIZE = None
_CONTEXT_PARALLEL_GROUP_RANKS = None


def get_cp_rank_size():
    if _CONTEXT_PARALLEL_GROUP:
        return _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE
    else:
        return 0, 1


def local_shard(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    if not _CONTEXT_PARALLEL_GROUP:
        return x

    cp_rank, cp_size = get_cp_rank_size()
    return x.tensor_split(cp_size, dim=dim)[cp_rank]


def set_cp_group(cp_group, ranks, global_rank):
    global _CONTEXT_PARALLEL_GROUP, _CONTEXT_PARALLEL_RANK, _CONTEXT_PARALLEL_GROUP_SIZE, _CONTEXT_PARALLEL_GROUP_RANKS
    if _CONTEXT_PARALLEL_GROUP is not None:
        raise RuntimeError("CP group already initialized.")
    _CONTEXT_PARALLEL_GROUP = cp_group
    _CONTEXT_PARALLEL_RANK = dist.get_rank(cp_group)
    _CONTEXT_PARALLEL_GROUP_SIZE = dist.get_world_size(cp_group)
    _CONTEXT_PARALLEL_GROUP_RANKS = ranks

    assert _CONTEXT_PARALLEL_RANK == ranks.index(
        global_rank
    ), f"Rank mismatch: {global_rank} in {ranks} does not have position {_CONTEXT_PARALLEL_RANK} "
    assert _CONTEXT_PARALLEL_GROUP_SIZE == len(
        ranks
    ), f"Group size mismatch: {_CONTEXT_PARALLEL_GROUP_SIZE} != len({ranks})"


def get_cp_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise RuntimeError("CP group not initialized")
    return _CONTEXT_PARALLEL_GROUP


def is_cp_active():
    return _CONTEXT_PARALLEL_GROUP is not None

