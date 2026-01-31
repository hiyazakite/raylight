import os
import uuid
from typing import Optional, Any, Set, Dict, Tuple, List, TYPE_CHECKING
import comfy.utils
import comfy.lora
from comfy.lora_convert import convert_lora
from raylight.comfy_dist.lora import load_lora as dist_load_lora

if TYPE_CHECKING:
    from raylight.distributed_worker.worker_config import WorkerConfig


class LoraManager:
    def __init__(self):
        self._applied_loras: Set[Tuple[str, float]] = set()
        self._lora_configs: Dict[str, List[Tuple[str, float]]] = {} # config_hash -> [(lora_path, strength), ...]
        self._current_lora_config_hash: Optional[str] = None

    def load_lora(
        self, 
        model: Any, 
        config: "WorkerConfig", 
        lora_path: str, 
        strength_model: float, 
        lora_config_hash: Optional[str] = None
    ) -> bool:
        """Loads a LoRA into the model on this worker via mmap.
        
        Note: The caller (RayWorker) MUST ensure the model is loaded/re-hydrated 
        before calling this method via reload_model_if_needed().
        """
        filename = os.path.basename(lora_path)
        
        if filename.startswith("._") or filename == ".DS_Store":
            print(f"[RayWorker {config.local_rank}] Skipping hidden/junk file: {filename}")
            return True

        # Store this LoRA in the config registry for re-application after offload
        if lora_config_hash is not None:
            if lora_config_hash not in self._lora_configs:
                self._lora_configs[lora_config_hash] = []
            
            lora_entry = (lora_path, strength_model)
            # Avoid duplicate entries in the same config
            if lora_entry not in self._lora_configs[lora_config_hash]:
                self._lora_configs[lora_config_hash].append(lora_entry)
                print(f"[RayWorker {config.local_rank}] Registered LoRA for config {lora_config_hash}: {filename}")
        
        # BRANCH ISOLATION: If lora_config_hash changed, reset all LoRAs first
        if lora_config_hash is not None and lora_config_hash != self._current_lora_config_hash:
            if self._current_lora_config_hash is not None:
                print(f"[RayWorker {config.local_rank}] LoRA config changed ('{self._current_lora_config_hash}' -> '{lora_config_hash}'). Resetting patches...")
                self.reset_loras(model, config)
            self._current_lora_config_hash = lora_config_hash
        
        # Create a unique signature for this LoRA
        lora_sig = (lora_path, strength_model)
        
        # IDEMPOTENCY: Skip if this exact LoRA was already applied in this config
        if lora_sig in self._applied_loras:
            print(f"[RayWorker {config.local_rank}] LoRA already applied: {filename} (strength={strength_model}). Skipping duplicate.")
            return True

        print(f"[RayWorker {config.local_rank}] Loading LoRA: {filename} (strength={strength_model})")
        
        # 1. (Caller Responsibility) Ensure model is loaded/re-hydrated
        if model is None:
             raise RuntimeError("Cannot load LoRA: Model is not loaded. Caller must hydrate model first.")
        
        # 2. Apply using core logic (shared with re-apply)
        self._apply_lora_core(model, config, lora_path, strength_model)
        
        print(f"[RayWorker {config.local_rank}] LoRA applied successfully.")
        return True

    def reset_loras(self, model: Any, config: "WorkerConfig"):
        """Clears all applied LoRAs and resets the model patches."""
        print(f"[RayWorker {config.local_rank}] Resetting LoRAs...")
        
        # Clear tracking
        self._applied_loras.clear()
        
        # FIX Bug #1: Clear the config hash so next LoRA application is seen as new
        self._current_lora_config_hash = None
        
        # Clear model patches if model exists
        if model is not None:
            # Clear the patches dict
            if hasattr(model, "patches"):
                model.patches.clear()
            
            # FIX Bug #3: Update patches_uuid to signal ComfyUI that patches changed
            if hasattr(model, "patches_uuid"):
                model.patches_uuid = uuid.uuid4()
            
            # Restore original weights from backup if any
            if hasattr(model, "backup") and model.backup:
                print(f"[RayWorker {config.local_rank}] Restoring {len(model.backup)} backed up weights...")
                for k, bk in model.backup.items():
                    if bk.inplace_update:
                        comfy.utils.copy_to_param(model.model, k, bk.weight)
                    else:
                        comfy.utils.set_attr_param(model.model, k, bk.weight)
                model.backup.clear()
        
        print(f"[RayWorker {config.local_rank}] LoRAs reset complete.")

    def reapply_loras_for_config(self, model: Any, config: "WorkerConfig", config_hash: str) -> bool:
        """Re-apply all LoRAs for a specific config_hash after model reload.
        
        Note: The caller (RayWorker) MUST ensure the model is loaded/re-hydrated.
        """
        # BRANCH ISOLATION: If config_hash is None or not registered, and we HAVE an active config,
        # it means the user likely bypassed LoRA nodes. We must RESET to clean the model.
        if config_hash is None or config_hash not in self._lora_configs:
            if self._current_lora_config_hash is not None:
                print(f"[RayWorker {config.local_rank}] No LoRAs for hash {config_hash}. Resetting current config '{self._current_lora_config_hash}'...")
                self.reset_loras(model, config)
            else:
                print(f"[RayWorker {config.local_rank}] No LoRAs registered and none active. Skipping.")
            return True
        
        # Check if we already have the right config applied
        if self._current_lora_config_hash == config_hash and self._applied_loras:
            print(f"[RayWorker {config.local_rank}] Config {config_hash} already applied. Skipping re-apply.")
            return True
        
        print(f"[RayWorker {config.local_rank}] Re-applying LoRAs for config {config_hash}...")
        
        # Reset current patches first
        self.reset_loras(model, config)
        
        if model is None:
             raise RuntimeError("Cannot re-apply LoRAs: Model is not loaded.")
        
        # Re-apply each LoRA in this config
        lora_list = self._lora_configs[config_hash]
        for lora_path, strength in lora_list:
            filename = os.path.basename(lora_path)
            print(f"[RayWorker {config.local_rank}] Re-applying LoRA: {filename} (strength={strength})")
            self._apply_lora_core(model, config, lora_path, strength)
        
        # Update tracking
        self._current_lora_config_hash = config_hash
        
        print(f"[RayWorker {config.local_rank}] Re-applied {len(lora_list)} LoRAs for config {config_hash}.")
        return True
    
    def _apply_lora_core(self, model: Any, config: "WorkerConfig", lora_path: str, strength_model: float):
        """Core LoRA application logic without registration/tracking."""
        filename = os.path.basename(lora_path)
        
        # Load LoRA State Dict (Mmap)
        lora_sd = comfy.utils.load_torch_file(lora_path)
        
        # Resolve Keys & Convert
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        
        lora_patches = convert_lora(lora_sd)
        
        # Load Patches using Raylight's distributed lora helper
        loaded_patches = dist_load_lora(lora_patches, key_map)
        
        # Apply to Model
        model.add_patches(loaded_patches, strength_model)
        
        # Track as applied
        self._applied_loras.add((lora_path, strength_model))
        
        print(f"[RayWorker {config.local_rank}] LoRA core apply complete: {filename}")
    
    def clear_gpu_refs(self, model: Any = None, config: Optional["WorkerConfig"] = None):
        """Clean up references to allow VRAM release.
        
        Args:
            model: Optional model instance. If None, this is a no-op (or should caller pass it?)
            config: Optional config for logging.
        """
        # 1. Clear weight_function/bias_function closures on ALL modules
        if model is None: return
        
        # Logging helper
        def log(msg):
            if config: print(msg)

        diffusion_model = getattr(model, "model", None)
        if diffusion_model is not None:
            cleared_funcs = 0
            for m in diffusion_model.modules():
                if hasattr(m, "weight_function") and len(m.weight_function) > 0:
                    m.weight_function = []
                    cleared_funcs += 1
                if hasattr(m, "bias_function") and len(m.bias_function) > 0:
                    m.bias_function = []
                    cleared_funcs += 1
            if cleared_funcs > 0:
                log(f"[RayWorker] Cleared {cleared_funcs} weight/bias functions.")
        
        # 2. Clear patches dict on model patcher
        if hasattr(model, "patches") and model.patches:
            patch_count = len(model.patches)
            model.patches.clear()
            log(f"[RayWorker] Cleared {patch_count} patches from patcher.")
        
        # 3. Clear .patches attribute on individual parameters (GGMLTensor carries these)
        if diffusion_model is not None:
            cleared_param_patches = 0
            for name, param in diffusion_model.named_parameters():
                if hasattr(param, "patches") and param.patches:
                    param.patches = []
                    cleared_param_patches += 1
            if cleared_param_patches > 0:
                log(f"[RayWorker] Cleared .patches on {cleared_param_patches} parameters.")

    def clear_tracking(self):
        """Clears tracking state."""
        self._applied_loras.clear()
        self._current_lora_config_hash = None
