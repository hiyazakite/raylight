import os
import uuid
from typing import Optional, Any, Set, Dict, Tuple, List, TYPE_CHECKING
import comfy.utils
import comfy.lora
from comfy.lora_convert import convert_lora
from raylight.comfy_dist.lora import load_lora as dist_load_lora

if TYPE_CHECKING:
    from raylight.distributed_actor.actor_config import ActorConfig


class LoraManager:
    def __init__(self):
        self._applied_loras: Set[Tuple[str, float]] = set()
        self._lora_configs: Dict[str, List[Tuple[str, float]]] = {} # config_hash -> [(lora_path, strength), ...]
        self._current_lora_config_hash: Optional[str] = None
        self._lora_seed: int = 318008
        # Bake caching: when set, the pinned cache holds baked (base + LoRA)
        # weights for this config — no hooks or re-baking needed on reruns.
        self._baked_config_hash: Optional[str] = None

    def load_lora(
        self, 
        model: Any, 
        config: "ActorConfig", 
        lora_path: str, 
        strength_model: float, 
        lora_config_hash: Optional[str] = None,
        seed: int = 318008,
    ) -> bool:
        """Loads a LoRA into the model on this worker via mmap.

        For INT8-quantized models the LoRA adapters are automatically rewrapped
        to use stochastic rounding in INT8 space (same logic as the standalone
        RayINT8LoraLoader nodes).  Float-precision layers are unaffected.

        Note: The caller (RayActor) MUST ensure the model is loaded/re-hydrated 
        before calling this method via reload_model_if_needed().
        """
        self._lora_seed = seed
        filename = os.path.basename(lora_path)
        
        if filename.startswith("._") or filename == ".DS_Store":
            print(f"[RayActor {config.local_rank}] Skipping hidden/junk file: {filename}")
            return True

        # Store this LoRA in the config registry for re-application after offload
        if lora_config_hash is not None:
            if lora_config_hash not in self._lora_configs:
                self._lora_configs[lora_config_hash] = []
            
            lora_entry = (lora_path, strength_model)
            # Avoid duplicate entries in the same config
            if lora_entry not in self._lora_configs[lora_config_hash]:
                self._lora_configs[lora_config_hash].append(lora_entry)
                print(f"[RayActor {config.local_rank}] Registered LoRA for config {lora_config_hash}: {filename}")
        
        # BRANCH ISOLATION: If lora_config_hash changed, reset all LoRAs first
        if lora_config_hash is not None and lora_config_hash != self._current_lora_config_hash:
            if self._current_lora_config_hash is not None:
                print(f"[RayActor {config.local_rank}] LoRA config changed ('{self._current_lora_config_hash}' -> '{lora_config_hash}'). Resetting patches...")
                self.reset_loras(model, config)
            self._current_lora_config_hash = lora_config_hash
        
        # Create a unique signature for this LoRA
        lora_sig = (lora_path, strength_model)
        
        # IDEMPOTENCY: Skip if this exact LoRA was already applied in this config
        if lora_sig in self._applied_loras:
            print(f"[RayActor {config.local_rank}] LoRA already applied: {filename} (strength={strength_model}). Skipping duplicate.")
            return True

        print(f"[RayActor {config.local_rank}] Loading LoRA: {filename} (strength={strength_model})")
        
        # 1. (Caller Responsibility) Ensure model is loaded/re-hydrated
        if model is None:
             raise RuntimeError("Cannot load LoRA: Model is not loaded. Caller must hydrate model first.")
        
        # 2. Apply using core logic (shared with re-apply)
        self._apply_lora_core(model, config, lora_path, strength_model)
        
        print(f"[RayActor {config.local_rank}] LoRA applied successfully.")
        return True

    def reset_loras(self, model: Any, config: "ActorConfig"):
        """Clears all applied LoRAs and resets the model patches."""
        print(f"[RayActor {config.local_rank}] Resetting LoRAs...")
        
        # Clear tracking
        self._applied_loras.clear()
        self._baked_config_hash = None
        
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
                print(f"[RayActor {config.local_rank}] Restoring {len(model.backup)} backed up weights...")
                # Determine target device: if model is actively on CUDA (e.g.
                # zero_ram), restored weights must stay there.  Backups live on
                # offload_device (CPU) so a device move is needed.
                target_dev = getattr(model, "current_device", None)
                need_move = target_dev is not None and target_dev.type == "cuda"

                for k, bk in model.backup.items():
                    w = bk.weight
                    if need_move and w.device != target_dev:
                        w = w.to(target_dev)
                    if bk.inplace_update:
                        comfy.utils.copy_to_param(model.model, k, w)
                    else:
                        comfy.utils.set_attr_param(model.model, k, w)
                model.backup.clear()

            # Wipe weight_function / bias_function from modules that had
            # LoRA patches attached via LowVramPatch (e.g. TPGGMLLinear).
            # Without this, stale LowVramPatch callbacks would try to look
            # up keys in the now-empty patches dict → KeyError.
            if hasattr(model, "model") and model.model is not None:
                for m in model.model.modules():
                    if hasattr(m, "weight_function") and m.weight_function:
                        m.weight_function = []
                    if hasattr(m, "bias_function") and m.bias_function:
                        m.bias_function = []

            # Sync weight-patches UUID so ComfyUI's partially_load() knows
            # the current weights already reflect the (now empty) patches
            # state.  Without this, the UUID mismatch triggers a wasteful
            # unpatch_model(cpu) → load(cuda) cycle that can leave TP-only
            # modules (e.g. TPRMSNormAcrossHeads) stranded on CPU.
            if hasattr(model, "patches_uuid") and hasattr(model, "model"):
                model.model.current_weight_patches_uuid = model.patches_uuid
        
        print(f"[RayActor {config.local_rank}] LoRAs reset complete.")

    def reapply_loras_for_config(self, model: Any, config: "ActorConfig", config_hash: str) -> bool:
        """Re-apply all LoRAs for a specific config_hash after model reload.
        
        Note: The caller (RayActor) MUST ensure the model is loaded/re-hydrated.
        """
        # BRANCH ISOLATION: If config_hash is None or not registered, and we HAVE an active config,
        # it means the user likely bypassed LoRA nodes. We must RESET to clean the model.
        if config_hash is None or config_hash not in self._lora_configs:
            if self._current_lora_config_hash is not None:
                print(f"[RayActor {config.local_rank}] No LoRAs for hash {config_hash}. Resetting current config '{self._current_lora_config_hash}'...")
                self.reset_loras(model, config)
            else:
                print(f"[RayActor {config.local_rank}] No LoRAs registered and none active. Skipping.")
            return True
        
        # Check if we already have the right config applied
        if self._current_lora_config_hash == config_hash and self._applied_loras:
            # Extra check: if pinned cache is synced with baked state, we can
            # skip entirely — weights already have LoRA fused in.
            if self._baked_config_hash == config_hash:
                print(f"[RayActor {config.local_rank}] Config {config_hash} baked and cached in pinned. Skipping re-apply.")
                return True
            print(f"[RayActor {config.local_rank}] Config {config_hash} already applied. Skipping re-apply.")
            return True
        
        print(f"[RayActor {config.local_rank}] Re-applying LoRAs for config {config_hash}...")
        
        # Reset current patches first
        self.reset_loras(model, config)
        
        if model is None:
             raise RuntimeError("Cannot re-apply LoRAs: Model is not loaded.")
        
        # Re-apply each LoRA in this config
        lora_list = self._lora_configs[config_hash]
        for lora_path, strength in lora_list:
            filename = os.path.basename(lora_path)
            print(f"[RayActor {config.local_rank}] Re-applying LoRA: {filename} (strength={strength})")
            self._apply_lora_core(model, config, lora_path, strength)
        
        # Update tracking
        self._current_lora_config_hash = config_hash
        
        print(f"[RayActor {config.local_rank}] Re-applied {len(lora_list)} LoRAs for config {config_hash}.")
        return True
    
    def _apply_lora_core(self, model: Any, config: "ActorConfig", lora_path: str, strength_model: float):
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

        # INT8 auto-detection: rewrap adapters for quantized layers so the
        # LoRA delta is applied correctly in INT8 space via stochastic rounding.
        try:
            from raylight.comfy_extra_dist.int8.int8_quant import wrap_patches_for_int8
            loaded_patches = wrap_patches_for_int8(model, loaded_patches, seed=self._lora_seed)
        except Exception as e:
            print(f"[RayActor {config.local_rank}] INT8 LoRA wrap skipped: {e}")

        # Apply to Model
        model.add_patches(loaded_patches, strength_model)
        
        # Track as applied
        self._applied_loras.add((lora_path, strength_model))
        
        print(f"[RayActor {config.local_rank}] LoRA core apply complete: {filename}")
    
    def clear_gpu_refs(self, model: Any = None, config: Optional["ActorConfig"] = None):
        """Clean up references to allow VRAM release.
        
        Args:
            model: Optional model instance. If None, this is a no-op (or should caller pass it?)
            config: Optional config for logging.
        """
        # Fused single-pass: clear weight/bias functions AND per-param
        # .patches in one traversal of the module tree (avoids 3 separate
        # iterations over the same model).
        if model is None: return
        
        # Logging helper
        def log(msg):
            if config: print(msg)

        diffusion_model = getattr(model, "model", None)
        if diffusion_model is not None:
            cleared_funcs = 0
            cleared_param_patches = 0
            for m in diffusion_model.modules():
                # Clear weight/bias function closures
                if hasattr(m, "weight_function") and m.weight_function:
                    m.weight_function = []
                    cleared_funcs += 1
                if hasattr(m, "bias_function") and m.bias_function:
                    m.bias_function = []
                    cleared_funcs += 1
                # Clear per-parameter .patches (GGMLTensor) in same visit
                for p in m.parameters(recurse=False):
                    if hasattr(p, "patches") and p.patches:
                        p.patches = []
                        cleared_param_patches += 1
            if cleared_funcs > 0:
                log(f"[RayActor] Cleared {cleared_funcs} weight/bias functions.")
            if cleared_param_patches > 0:
                log(f"[RayActor] Cleared .patches on {cleared_param_patches} parameters.")
        
        # Clear patches dict on model patcher
        if hasattr(model, "patches") and model.patches:
            patch_count = len(model.patches)
            model.patches.clear()
            log(f"[RayActor] Cleared {patch_count} patches from patcher.")

    def mark_baked(self, config_hash: str) -> None:
        """Record that the pinned cache now holds baked state for this config."""
        self._baked_config_hash = config_hash

    def is_baked_for(self, config_hash: str) -> bool:
        """Check if the pinned cache holds baked state for this config."""
        return self._baked_config_hash == config_hash

    def clear_tracking(self):
        """Clears tracking state."""
        self._applied_loras.clear()
        self._current_lora_config_hash = None
        self._baked_config_hash = None
