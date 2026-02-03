from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from raylight.distributed_worker.worker_config import WorkerConfig

def prepare_fsdp_model_for_sampling(work_model, config: "WorkerConfig", state_dict: Optional[dict] = None) -> bool:
    """
    Handles FSDP-specific model preparation including weight baking.
    
    Args:
        work_model: The model patcher instance.
        config: Worker configuration.
        state_dict: Optional state dict to inject if missing.
        
    Returns:
        bool: True if the model weights were modified (baked).
    """
    model_was_modified = False
    
    # Propagate state_dict to patcher if not present (crucial for lazy/parallel path)
    if getattr(work_model, "fsdp_state_dict", None) is None:
        if state_dict is not None:
                print(f"[RayWorker {config.local_rank}] Injecting saved FSDP state dict into model patcher...")
                work_model.set_fsdp_state_dict(state_dict)

    # CRITICAL FOR FSDP: Bake LoRAs into weights before wrapping!
    # Since 'use_orig_params=True' is not supported in this torch version, FSDP flattens params
    # which breaks standard ComfyUI soft-patching (hooks). We must hard-patch (bake) first.
    if hasattr(work_model, "patches") and work_model.patches:
            print(f"[RayWorker {config.local_rank}] FSDP: Baking {len(work_model.patches)} patches into weights before sharding...")
            model_was_modified = True
            
            # Force in-place update so the underlying model instance (which FSDP wraps) is modified
            prev_inplace = work_model.weight_inplace_update
            work_model.weight_inplace_update = True
            
            # force_patch_weights=True permanently modifies the weights in work_model.model
            work_model.load(device_to="cpu", force_patch_weights=True)
            
            # Free memory: We don't need backups of the unbaked weights since we are committing to them
            if hasattr(work_model, "backup"):
                work_model.backup.clear()

            # Restore inplace flag
            work_model.weight_inplace_update = prev_inplace
            
            # Mark as consistently baked so we can detect this state later
            work_model.is_fsdp_baked = True
            
            # CRITICAL: Prevent FSDP wrapper from reloading stale state_dict over our baked weights
            if hasattr(work_model, "set_fsdp_state_dict"):
                print(f"[RayWorker {config.local_rank}] FSDP: Clearing fsdp_state_dict to preserve baked weights.")
                work_model.set_fsdp_state_dict(None)
            
            # Clear patches to prevent double application or tracking issues
            work_model.patches.clear()
            if hasattr(work_model, "patches_uuid"):
                work_model.patches_uuid = uuid.uuid4()
    
    work_model.patch_fsdp()
    
    return model_was_modified
