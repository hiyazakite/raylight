import torch
import torch.distributed as dist
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

def extract_local_tensor(
    value: torch.Tensor, 
    rank: int = None, 
    world_size: int = None, 
    ring_impl_type: str = "basic",
    dim: int = 1
) -> torch.Tensor:
    """
    Extracts the local shard of a tensor based on the sequence parallel rank and world size.
    Supports basic (chunked) and zigzag partitioning.
    """
    if rank is None:
        rank = get_sequence_parallel_rank()
    if world_size is None:
        world_size = get_sequence_parallel_world_size()

    if world_size == 1:
        return value

    if ring_impl_type == "basic":
        return value.chunk(world_size, dim=dim)[rank].contiguous()
    
    if ring_impl_type == "zigzag":
        # Zigzag partitioning requires rd (ring degree) and ud (ulysses degree)
        # For our current implementation, we assume world_size is the ring degree
        # if ud is not explicitly handled yet.
        # In a hybrid setup, world_size = rd * ud.
        # However, Raylight currently often operates with a single SP group.
        
        # We need the ring process group to get the ring rank/size.
        # For now, we'll implement it assuming the full SP group is the ring group
        # if not otherwise specified.
        
        rd = world_size
        r_rank = rank
        
        input_dim = value.dim()
        assert input_dim >= 2
        batch_size, seqlen, *rest = value.shape

        # Partition into 2*rd chunks
        value_chunks = value.chunk(2 * rd, dim=dim)
        
        # Zigzag: concatenate chunk[r_rank] and chunk[2*rd - r_rank - 1]
        local_value = torch.cat(
            [value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]], dim=dim
        )
        
        return local_value.contiguous()

    raise ValueError(f"Unsupported ring_impl_type: {ring_impl_type}")
