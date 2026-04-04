from typing import Optional, TYPE_CHECKING
import uuid

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

if TYPE_CHECKING:
    from raylight.distributed_actor.actor_config import ActorConfig

def prepare_fsdp_model_for_sampling(work_model, config: "ActorConfig", state_dict: Optional[dict] = None) -> bool:
    """
    Handles FSDP-specific model preparation including weight baking.
    
    Args:
        work_model: The model patcher instance.
        config: Actor configuration.
        state_dict: Optional state dict to inject if missing.
        
    Returns:
        bool: True if the model weights were modified (baked).
    """
    model_was_modified = False
    
    # Propagate state_dict to patcher if not present (crucial for lazy/parallel path)
    if getattr(work_model, "fsdp_state_dict", None) is None:
        if state_dict is not None:
                print(f"[RayActor {config.local_rank}] Injecting saved FSDP state dict into model patcher...")
                work_model.set_fsdp_state_dict(state_dict)

    # CRITICAL FOR FSDP: Bake LoRAs into weights before wrapping!
    # [OPTIMIZATION] We now do this in a streaming fashion inside the FSDP sharding loop
    # to avoid the massive RAM spike of duplicating the model.
    # See `fsdp.py` for the implementation.
    
    # if hasattr(work_model, "patches") and work_model.patches:
    #         print(f"[RayActor {config.local_rank}] FSDP: Baking {len(work_model.patches)} patches into weights before sharding...")
    #         model_was_modified = True
            
    #         # Force in-place update so the underlying model instance (which FSDP wraps) is modified
    #         prev_inplace = work_model.weight_inplace_update
    #         work_model.weight_inplace_update = True
            
    #         # force_patch_weights=True permanently modifies the weights in work_model.model
    #         work_model.load(device_to="cpu", force_patch_weights=True)
            
    #         # Free memory: We don't need backups of the unbaked weights since we are committing to them
    #         if hasattr(work_model, "backup"):
    #             work_model.backup.clear()

    #         # Restore inplace flag
    #         work_model.weight_inplace_update = prev_inplace
            
    #         # Mark as consistently baked so we can detect this state later
    #         work_model.is_fsdp_baked = True
            
    #         # CRITICAL: Prevent FSDP wrapper from reloading stale state_dict over our baked weights
    #         if hasattr(work_model, "set_fsdp_state_dict"):
    #             print(f"[RayActor {config.local_rank}] FSDP: Clearing fsdp_state_dict to preserve baked weights.")
    #             work_model.set_fsdp_state_dict(None)
            
    #         # Clear patches to prevent double application or tracking issues
    #         work_model.patches.clear()
    #         if hasattr(work_model, "patches_uuid"):
    #             work_model.patches_uuid = uuid.uuid4()
    
    work_model.patch_fsdp()
    
    return model_was_modified


def bake_lora_hooks(work_model, local_rank: int = 0) -> int:
    """Bake LowVramPatch hooks into FSDP weights in-place, then clear hooks.

    After calling this, patched modules have no hooks and their weights
    contain the LoRA-fused values.  Uses ``patch_weight_to_device()``
    which is device_mesh-aware (correct for FSDP sharded params).

    Args:
        work_model: The FSDPModelPatcher instance.
        local_rank: For logging.

    Returns:
        Number of weight keys baked.
    """
    patches = getattr(work_model, "patches", None)
    if not patches:
        return 0

    patch_keys = list(patches.keys())
    baked = 0

    for key in patch_keys:
        try:
            work_model.patch_weight_to_device(key)
            baked += 1
        except Exception as e:
            print(f"[RayActor {local_rank}] Failed to bake key {key}: {e}")

    # Clear hooks on all modules now that weights are baked
    diff_model = getattr(work_model, "model", None)
    if diff_model is not None:
        for m in diff_model.modules():
            if hasattr(m, "weight_function") and m.weight_function:
                m.weight_function = []
            if hasattr(m, "bias_function") and m.bias_function:
                m.bias_function = []

    # Clear patches dict — prevents hooks from being re-installed on next load()
    patches.clear()
    if hasattr(work_model, "patches_uuid"):
        work_model.patches_uuid = uuid.uuid4()

    # Mark as baked
    work_model.is_fsdp_baked = True

    if baked > 0:
        print(f"[RayActor {local_rank}] Baked {baked} LoRA patches into weights "
              f"(hooks cleared, patches purged).")
    return baked


# ---------------------------------------------------------------------------
# Hybrid FSDP + TP DeviceMesh helpers
# ---------------------------------------------------------------------------


def build_hybrid_mesh(fsdp_size: int, tp_size: int) -> DeviceMesh:
    """Build a 2-D DeviceMesh with axes (fsdp, tp).

    Example — 8 GPUs, fsdp_size=4, tp_size=2::

        rank layout:  [[ 0,  1],   # FSDP-group 0, TP-ranks 0-1
                       [ 2,  3],   # FSDP-group 1, TP-ranks 2-3
                       [ 4,  5],   # FSDP-group 2, TP-ranks 4-5
                       [ 6,  7]]   # FSDP-group 3, TP-ranks 6-7

    FSDP2 shards along the first axis; TP communicates along the second.
    """
    world = dist.get_world_size()
    assert world == fsdp_size * tp_size, (
        f"world_size={world} != fsdp_size={fsdp_size} * tp_size={tp_size}"
    )
    return init_device_mesh(
        "cuda", (fsdp_size, tp_size), mesh_dim_names=("fsdp", "tp")
    )


def initialize_tp_from_mesh(mesh_2d: DeviceMesh) -> None:
    """Initialize TensorParallelState from the 'tp' dimension of a 2-D mesh."""
    from raylight.distributed_modules.tensor_parallel import TensorParallelState

    tp_mesh = mesh_2d["tp"]
    TensorParallelState.initialize(tp_mesh.size(), pg=tp_mesh.get_group())


def shard_model_fsdp2_with_mesh(model, mesh_2d: DeviceMesh, **kwargs):
    """Wrap *model* with FSDP2 using the 'fsdp' dimension of a 2-D mesh."""
    from torch.distributed.fsdp import fully_shard

    fully_shard(model, mesh=mesh_2d["fsdp"], **kwargs)
    return model
