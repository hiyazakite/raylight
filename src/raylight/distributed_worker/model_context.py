"""Unified model lifecycle management with mmap support.

Provides a common abstraction for loading, offloading, and reloading models
across different formats (GGUF, Safetensors, FSDP) with optional mmap caching.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import os

import torch

from raylight.utils.checksum import compute_model_checksum, verify_model_checksum

if TYPE_CHECKING:
    from .worker_config import WorkerConfig
    from .managers.lora_manager import LoraManager


@dataclass
class ModelState:
    """Immutable snapshot of model loading parameters."""
    unet_path: str
    model_options: Dict[str, Any]
    dequant_dtype: Optional[str] = None
    patch_dtype: Optional[str] = None
    
    @property
    def cache_key(self) -> str:
        """Unique key for caching (file path)."""
        return self.unet_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage in reload_params."""
        return asdict(self)


class ModelContext(ABC):
    """Abstract base for unified model lifecycle management.
    
    Handles load → offload → reload cycle with optional mmap caching.
    Each format (GGUF, Safetensors, FSDP) implements format-specific behavior.
    """
    
    def __init__(self, use_mmap: bool = True):
        self.use_mmap = use_mmap
    
    # ─── Abstract Methods (format-specific) ──────────────────
    
    @abstractmethod
    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig") -> Dict[str, torch.Tensor]:
        """Load state dict with mmap (zero-copy where possible)."""
        pass
    
    @abstractmethod
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig") -> Dict[str, torch.Tensor]:
        """Load state dict without mmap (fallback)."""
        pass
    
    @abstractmethod
    def instantiate_model(self, sd: Dict[str, torch.Tensor], state: ModelState, config: "WorkerConfig", metadata: Any = None) -> Any:
        """Create ModelPatcher from state dict."""
        pass
    
    @abstractmethod
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """Fast VRAM transfer for soft-reloaded models."""
        pass
    
    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """Standard offload: Move weights to CPU and release tracking."""
        print(f"[{self.__class__.__name__}] Standard Offload: Releasing VRAM...")
        
        # Share the common offload cleanup logic
        self._common_offload_cleanup(model, lora_manager, config)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _common_offload_cleanup(self, model: Any, lora_manager: Optional["LoraManager"], config: "WorkerConfig"):
        """Shared post-offload cleanup for all model formats."""
        # Clear LoRA tracking for fresh re-application after reload
        if lora_manager:
            lora_manager.clear_tracking()
            print(f"[RayWorker {config.local_rank}] LoRA tracking cleared.")
        
        # Move buffers to CPU if model still exists
        if model is not None:
            diffusion_model = getattr(model, "model", None)
            if diffusion_model is not None:
                try:
                    for name, buf in diffusion_model.named_buffers():
                        if buf is not None and buf.device.type == 'cuda':
                            buf.data = buf.data.to('cpu')
                except Exception as e:
                    print(f"[RayWorker {config.local_rank}] Buffer cleanup warning: {e}")
                
                # Delete cached PE tensors
                for attr in ['_cached_pe', 'cached_pe', 'pe_cache', '_pe']:
                    if hasattr(diffusion_model, attr):
                        delattr(diffusion_model, attr)
                    if hasattr(diffusion_model, 'diffusion_model'):
                        inner = diffusion_model.diffusion_model
                        if hasattr(inner, attr):
                            delattr(inner, attr)
        
        # Force garbage collection and CUDA cleanup
        from raylight.utils.common import cleanup_memory
        cleanup_memory()
        mem_now = torch.cuda.memory_allocated(config.device) / 1e9
        print(f"[RayWorker {config.local_rank}] Post-Offload VRAM: {mem_now:.2f}GB")

    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor], config: "WorkerConfig") -> Dict[str, torch.Tensor]:
        """Optional pre-processing of state dict before instantiation."""
        return sd
    
    # ─── Shared Logic ────────────────────────────────────────

    def load(self, state: ModelState, config: "WorkerConfig", state_cache: Any) -> Tuple[Any, Dict[str, Any]]:
        """Unified model loading logic.
        
        1. Check cache for existing state dict.
        2. If not in cache, load from disk (mmap or standard).
        3. Prepare state dict (e.g., strip prefixes).
        4. Instantiate model.
        5. Perform post-load setup.
        """
        sd = None
        metadata = {}
        
        # 1. Check cache
        cached = state_cache.get(state.cache_key)
        if cached is not None:
            print(f"[ModelContext] Cache Hit: {os.path.basename(state.cache_key)}")
            if isinstance(cached, tuple) and len(cached) >= 2 and isinstance(cached[0], dict):
                 sd = cached[0]
                 metadata = cached[1]
                 # Retrieve checksum if present (v3 format: sd, meta, checksum)
                 if len(cached) >= 3:
                     stored_checksum = cached[2]
                     verify_model_checksum(sd, stored_checksum, metadata, context_tag="ModelContext")
            else:
                 sd = cached
        else:
            # 2. Disk load (mmap or standard based on toggle)
            if self.use_mmap:
                print(f"[ModelContext] Mmap Load: {os.path.basename(state.cache_key)}")
                res = self.load_state_dict_mmap(state, config)
                
                # Metadata aware return
                if isinstance(res, tuple) and len(res) == 2:
                    sd, metadata = res
                    # Compute checksum on fresh load
                    checksum = compute_model_checksum(sd, metadata)
                    # Cache tuple with checksum
                    state_cache.put(state.cache_key, (sd, metadata, checksum))
                    print(f"[ModelContext] Cached model with checksum: {checksum}")
                else:
                    sd = res
                    # Standard cache (tuple with empty meta and checksum)
                    checksum = compute_model_checksum(sd)
                    state_cache.put(state.cache_key, (sd, {}, checksum))
            else:
                print(f"[ModelContext] Standard Load: {os.path.basename(state.cache_key)}")
                sd = self.load_state_dict_standard(state, config)
        
        if sd is None:
            raise RuntimeError(f"Failed to load state dict for {state.unet_path}")
            
        # 3. Prepare state dict (e.g. strip prefixes) without modifying cache
        # Note: If prepare_state_dict modifies in-place, we must copy first if we want cache pristine.
        # But GGUFContext usually returns a new dict if keys change.
        # For safety with cache references, we can check logic.
        if not isinstance(sd, dict):
            raise RuntimeError(f"State dict must be a dict, got {type(sd)} for {state.unet_path}")
            
        sd_to_use = self.prepare_state_dict(sd, config)
        
        # 4. Instantiate model
        model = self.instantiate_model(sd_to_use, state, config, metadata=metadata)
        

        
        # Post-load setup
        reload_params = state.to_dict()
        if model is not None:
             model.unet_path = state.unet_path
             if self.use_mmap:
                 model.mmap_cache = sd

        return model, reload_params
    

    
    def reload_if_needed(self, model: Any, target_device: torch.device, reload_params: Dict[str, Any], config: "WorkerConfig", state_cache: Any) -> Tuple[Any, Dict[str, Any]]:
        """Shared reload logic - checks device and triggers appropriate reload.
        
        Returns:
            (model, reload_params) - potentially new model instance if full reload
        """
        current = getattr(model, "current_device", None) if model else None
        
        if model is not None and str(current) == str(target_device):
            print(f"[ModelContext] Already on {target_device}, skipping reload.")
            return model, reload_params
        
        if model is not None:
            # Soft reload - model object exists but on wrong device
            print(f"[ModelContext] Hot-loading to {target_device}...")
            self.hot_load(model, target_device, reload_params, state_cache)
            return model, reload_params
        else:
            # Full reload - model is None
            if not reload_params:
                 raise RuntimeError("Cannot reload model: No reload_params found.")
            
            print(f"[ModelContext] Full reload to {target_device}...")
            state = ModelState(**reload_params)
            new_model, new_params = self.load(state, config, state_cache)
            return new_model, new_params


class GGUFContext(ModelContext):
    """Context for GGUF model loading with mmap support."""
    
    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig") -> Any:
        from raylight.expansion.comfyui_gguf.loader import gguf_sd_loader
        sd, extra = gguf_sd_loader(state.unet_path)
        # Return tuple to cache metadata
        metadata = extra.get("metadata", {})
        return sd, metadata
    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor], config: "WorkerConfig") -> Dict[str, torch.Tensor]:
        """GGUF: Legacy behavior was NOT to strip prefixes here (Comfy handles it)."""
        # The stripping logic below was present in legacy code but Unused.
        # Activating it caused regression. Restoring pass-through behavior.
        # keys = list(sd.keys())
        # if any(k.startswith("diffusion_model.") for k in keys[:5]):
        #      # ...
        #      pass
        return sd
    
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig") -> Dict:
        # GGUF library always uses mmap internally
        return self.load_state_dict_mmap(state, config)
    
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None):
        from raylight.expansion.comfyui_gguf.ops import GGMLOps
        from raylight.expansion.comfyui_gguf.nodes import GGUFModelPatcher
        import comfy.sd
        import inspect
        
        ops = GGMLOps()
        
        # Apply dequant/patch dtypes
        if state.dequant_dtype and state.dequant_dtype not in ("default", None):
            if state.dequant_dtype == "target":
                setattr(ops.Linear, "dequant_dtype", state.dequant_dtype)
            else:
                setattr(ops.Linear, "dequant_dtype", getattr(torch, state.dequant_dtype))
        
        if state.patch_dtype and state.patch_dtype not in ("default", None):
            if state.patch_dtype == "target":
                setattr(ops.Linear, "patch_dtype", state.patch_dtype)
            else:
                setattr(ops.Linear, "patch_dtype", getattr(torch, state.patch_dtype))
        
        # CRITICAL: Use shallow dict copy, NOT deep tensor clone!
        # sd.copy() copies dict keys/values but keeps tensor data shared (mmap refs intact)
        # This matches original gguf_load_diffusion_model behavior
        isolated = sd.copy()
        
        # Build kwargs for metadata if supported
        kwargs = {}
        # Metadata logic: If cache hit provided metadata, use it.
        # Else try self._temp_gguf_metadata (legacy fallback, maybe redundant now)
        gguf_meta = metadata if metadata is not None else getattr(self, "_temp_gguf_metadata", {})
        
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = gguf_meta
        
        model = comfy.sd.load_diffusion_model_state_dict(
            isolated, 
            model_options={"custom_operations": ops},
            **kwargs
        )
        
        if model is None:
            raise RuntimeError(f"Could not load GGUF model: {state.unet_path}")
        
        model = GGUFModelPatcher.clone(model)
        model.gguf_metadata = gguf_meta
        return model
    

    
    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """GGUF Soft-Offload: Restore mmap pointers to clear VRAM without copies."""
        if model is not None:
            print(f"[GGUFContext {config.local_rank}] GGUF Soft-Offload: Performing Zero-Copy Pointer Swap...")
            
            # 1. Clear LoRA GPU refs via manager
            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)
            
            # 2. Restore mmap pointers
            model.unpatch_model(device_to=torch.device('cpu'), unpatch_weights=True)
            model.current_device = torch.device('cpu')
            
            # 3. Store in worker-level mmap cache to survive model swaps
            # REMOVED: This overwrites the (sd, meta, checksum) tuple with just the dict,
            # destroying the checksum. ModelContext.load already handles caching correctly.
            # mmap_cache = getattr(model, "mmap_cache", None)
            # if mmap_cache:
            #     unet_path = getattr(model, "unet_path", None)
            #     if unet_path:
            #         worker_mmap_cache[unet_path] = mmap_cache
            
            # 4. Shared cleanup
            self._common_offload_cleanup(model, lora_manager, config)
            print(f"[GGUFContext {config.local_rank}] GGUF Soft-Offload Complete.")
    
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        if model is None:
            return
            
        print(f"[GGUFContext] Hot-loading to {device}...")

        # Re-hydrate mmap_cache from worker cache if needed
        unet_path = getattr(model, "unet_path", None)
        
        if not getattr(model, "mmap_cache", None):
            if unet_path and unet_path in state_cache:
                print(f"[GGUFContext] Re-hydrating mmap_cache from LRU cache: {os.path.basename(unet_path)}")
                cached = state_cache.get(unet_path)
                
                # Handle metadata/checksum cache format (tuple)
                if isinstance(cached, tuple) and len(cached) >= 2 and isinstance(cached[0], dict):
                    model.mmap_cache = cached[0]
                else:
                    # Legacy or direct dict
                    model.mmap_cache = cached
        
        # ALWAYS Verify checksum if available in cache (ensure we didn't corrupt it)
        if getattr(model, "mmap_cache", None) and unet_path and unet_path in state_cache:
             cached = state_cache.get(unet_path)
             if isinstance(cached, tuple) and len(cached) >= 3:
                 stored_sum = cached[2]
                 # cached[1] is metadata
                 meta = cached[1] if isinstance(cached[1], dict) else None
                 if isinstance(model.mmap_cache, dict):
                     verify_model_checksum(model.mmap_cache, stored_sum, meta, context_tag="GGUFContext")
        
        if hasattr(model, "load"):
            model.load(device, force_patch_weights=True)
            model.current_device = device


from raylight.expansion.comfyui_lazytensors.loader import SafetensorMmapWrapper


class LazyTensorContext(ModelContext):
    """Context for Safetensors model loading with streaming mmap support.
    
    Features:
    - Zero-copy mmap sharing via OS page cache (like GGUF)
    - Streaming GPU transfer (per-tensor to avoid RAM spike)
    - Fallback to full clone if streaming fails
    """
    
    def __init__(self, use_mmap: bool = True):
        super().__init__(use_mmap)
        self._streaming_enabled = True  # Can be disabled if issues detected
    
    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig") -> Dict:
        import safetensors.torch
        return safetensors.torch.load_file(state.unet_path, device="cpu")
    
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig") -> Dict:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path)
    
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None):
        """Instantiate model with lazy state dict (zero-copy until load)."""
        import comfy.sd
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps
        
        print("[LazyTensorContext] Wrapping state dict with LazySafetensors...")
        lazy_sd = wrap_state_dict_lazy(sd)
        
        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        load_options["custom_operations"] = SafetensorOps
        
        model = comfy.sd.load_diffusion_model_state_dict(lazy_sd, model_options=load_options)
        
        if model is None:
            raise RuntimeError(f"Could not load model: {state.unet_path}")
        
        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype
            
        # Store mmap cache on the patcher for offload pointer-swapping
        model.mmap_cache = sd
        return model
    
    def instantiate_model_with_fallback(self, sd: Dict, state: ModelState, config: "WorkerConfig"):
        """Try streaming approach with fallback to full clone."""
        import comfy.sd
        
        try:
            if self._streaming_enabled:
                # Streaming: shallow copy + deferred GPU transfer
                model = self.instantiate_model(sd, state, config)
                print("[LazyTensorContext] Streaming instantiation successful")
                return model
        except Exception as e:
            print(f"[LazyTensorContext] Streaming failed: {e}")
            print("[LazyTensorContext] Falling back to full clone...")
            self._streaming_enabled = False  # Disable for future loads
        
        # Fallback: full clone (safe but higher RAM)
        isolated = {k: v.clone() for k, v in sd.items()}
        
        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        
        model = comfy.sd.load_diffusion_model_state_dict(isolated, model_options=load_options)
        
        if model is None:
            raise RuntimeError(f"Could not load model: {state.unet_path}")
        
        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype
        
        return model
    
    def stream_to_device(self, model: Any, device: torch.device) -> bool:
        """Stream model weights to GPU per-tensor (avoids RAM spike).
        
        Returns True if streaming was used, False if fallback applied.
        """
        mmap_cache = getattr(model, "mmap_cache", None)
        
        if not mmap_cache or not self._streaming_enabled:
            # Fallback: standard model.to()
            print(f"[LazyTensorContext] Using standard .to({device})...")
            if model is not None and hasattr(model, "model"):
                model.model.to(device)
            return False
        
        try:
            print(f"[LazyTensorContext] Streaming to {device} per-tensor...")
            wrapper = SafetensorMmapWrapper(mmap_cache)
            
            if model is not None and hasattr(model, "model"):
                transferred = wrapper.stream_to_model(model.model, device)
                print(f"[LazyTensorContext] Streamed {transferred} parameters to {device}")
            
            return True
        except Exception as e:
            print(f"[LazyTensorContext] Streaming transfer failed: {e}, falling back...")
            if model is not None and hasattr(model, "model"):
                model.model.to(device)
            return False
    
    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """Pointer-swap offload: swap GPU tensors back to mmap refs (like GGUF)."""
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import swap_model_to_mmap
         
        if model is not None:
            print(f"[LazyTensorContext {config.local_rank}] Soft-Offload: Performing Pointer Swap...")
            
            # 1. Clear LoRA GPU refs via manager
            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)
            
            # 3. Unpatch first!
            if hasattr(model, "unpatch_model"):
                model.unpatch_model(device_to=None, unpatch_weights=True)
            
            # 4. Swap GPU tensors to mmap refs
            mmap_cache = getattr(model, "mmap_cache", None)
            diffusion_model = getattr(model, "model", None)
            if diffusion_model is not None and mmap_cache:
                swap_model_to_mmap(diffusion_model, mmap_cache)
                
                # Store in worker-level mmap cache to survive model swaps
                unet_path = getattr(model, "unet_path", None)
                if unet_path:
                    worker_mmap_cache[unet_path] = mmap_cache
            
            model.current_device = torch.device('cpu')
            
            # 5. Shared cleanup
            self._common_offload_cleanup(model, lora_manager, config)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[LazyTensorContext {config.local_rank}] Soft-Offload Complete.")

    
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """Hot reload: re-materialize from mmap refs to GPU (like GGUF)."""
        print(f"[LazyTensorContext] Hot reload to {device}...")
        
        # Verify checksum if available
        unet_path = getattr(model, "unet_path", None)
        if unet_path and unet_path in state_cache:
            cached = state_cache.get(unet_path)
            if isinstance(cached, tuple) and len(cached) >= 3:
                stored_sum = cached[2]
                meta = cached[1] if isinstance(cached[1], dict) else None
                if isinstance(model.mmap_cache, dict):
                    verify_model_checksum(model.mmap_cache, stored_sum, meta, context_tag="LazyTensorContext")
        
        # Model structure still exists, just reload weights to GPU
        if model is not None and hasattr(model, 'load'):
            model.load(device)
            model.current_device = device
        else:
            # Should have been handled by shared reload_if_needed logic for Full Reload
            pass


class FSDPContext(ModelContext):
    """Context for FSDP model loading with optional mmap for safetensors."""
    
    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig") -> Dict:
        # Use safetensors mmap if file is .safetensors
        if state.unet_path.lower().endswith(".safetensors"):
            import safetensors.torch
            return safetensors.torch.load_file(state.unet_path, device="cpu")
        return self.load_state_dict_standard(state, config)
    
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig") -> Dict:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path)
    
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None):
        from raylight.comfy_dist.sd import fsdp_load_diffusion_model_stat_dict
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps
        
        # Apply One-Time FSDP Patches (like legacy _apply_fsdp_patches)
        import comfy.model_patcher as model_patcher
        import comfy.model_management as model_management
        from raylight.comfy_dist.model_management import cleanup_models_gc
        from raylight.comfy_dist.model_patcher import LowVramPatch
        model_patcher.LowVramPatch = LowVramPatch
        model_management.cleanup_models_gc = cleanup_models_gc
        
        # JIT Loading: Wrap with LazySafetensor if not already (reusing logic from LazyTensors)
        # This prevents the "huge RAM spike" by delaying materialization until key access
        if self.use_mmap:
            print("[FSDPContext] Wrapping state dict with LazySafetensors for JIT loading...")
            sd = wrap_state_dict_lazy(sd)
        
        # Inject SafetensorOps to handle zero-copy assignment
        # Unconditionally overwrite to ensure we use SafetensorLayer (like LazyTensorContext)
        model_options = state.model_options.copy()
        model_options["custom_operations"] = SafetensorOps

        model, state_dict = fsdp_load_diffusion_model_stat_dict(
            sd,
            config.local_rank,
            getattr(config, "device_mesh", None), # Relying on device_mesh being in config
            config.parallel_dict.get("fsdp_cpu_offload", False),
            model_options=model_options
        )
        
        if model is None:
            raise RuntimeError(f"Could not load FSDP model: {state.unet_path}")
        
        # FSDP special: attach state dict to model for retrieval? 
        # Or return it? For now, we attach it to the model patcher if possible.
        # But wait, original code set `worker.state_dict`.
        # To remain stateless, we should return it or attach to model.
        # Let's attach to model.
        model.fsdp_state_dict = state_dict
        return model
    
    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """
        FSDP offload: release both model and state dict
        FSDP requires full reload currently due to its nature (sharding)
        """
        print(f"[FSDPContext {config.local_rank}] FSDP Hard-Offload: Releasing all resources...")
        
        # 0. Clear GPU refs (weight_functions, patches) that might hold cyclic refs
        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)
        
        if model is not None:
            # Explicitly release DTensor/FSDP storage
            if hasattr(model, "release_memory"):
                model.release_memory()

        # 2. Trigger shared cleanup
        self._common_offload_cleanup(model, lora_manager, config)
        
        # 4. Final VRAM flush
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """
        FSDP requires full reload currently due to its nature (sharding)
        """
        # This shouldn't be reached if we follow reload_if_needed logic correctly for "None" model
        pass


class VAEContext(LazyTensorContext):
    """Context for VAE model loading with streaming mmap support."""
    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor], config: "WorkerConfig") -> Dict[str, torch.Tensor]:
        """VAE: No prefix stripping needed (unlike UNET)."""
        return sd
        
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None):
        """Instantiate VAE model using comfy.sd.VAE (Standard ComfyUI path)."""
        import comfy.sd
        import comfy.utils
        from safetensors.torch import safe_open
        
        print(f"[VAEContext] Standard VAE init with {len(sd)} keys")
        
        metadata = None
        try:
            with safe_open(state.unet_path, framework="pt") as f:
                metadata = f.metadata()
            if metadata:
                print(f"[VAEContext] Loaded metadata: contains 'config': {'config' in metadata}")
        except Exception:
            pass  # Safetensors not installed or error
        
        # DEBUG: Check VAE version detection
        if "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd:
            tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
            print(f"[VAEContext DEBUG] conv1 shape: {tensor_conv1.shape} (chan={tensor_conv1.shape[0]})")
            if tensor_conv1.shape[0] == 512:
                print("[VAEContext DEBUG] Detected LTX VAE version 0 (LTX1)")
            elif tensor_conv1.shape[0] == 1024:
                if "encoder.down_blocks.1.conv.conv.bias" in sd:
                    print("[VAEContext DEBUG] Detected LTX VAE version 2 (LTX2 full)")
                else:
                    print("[VAEContext DEBUG] Detected LTX VAE version 1 (LTX2 basic)")
        
        # ComfyUI VAE init - PASS METADATA!
        try:
             vae_model = comfy.sd.VAE(sd=sd, metadata=metadata)
             vae_model.throw_exception_if_invalid()
        except Exception as e:
             raise RuntimeError(f"Could not load VAE model: {state.unet_path}. Error: {e}")
             
        # Store original sd for potential streaming optimization
        vae_model.mmap_cache = sd
        return vae_model

    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """Hot reload VAE using stream_vae_to_device logic to restore from mmap."""
        print(f"[VAEContext] Hot-loading VAE to {device} via mmap stream...")
        
        # Verify checksum if available
        unet_path = getattr(model, "unet_path", None)
        if unet_path and unet_path in state_cache:
            cached = state_cache.get(unet_path)
            if isinstance(cached, tuple) and len(cached) >= 3:
                stored_sum = cached[2]
                meta = cached[1] if isinstance(cached[1], dict) else None
                if isinstance(model.mmap_cache, dict):
                    verify_model_checksum(model.mmap_cache, stored_sum, meta, context_tag="VAEContext")

        self.stream_vae_to_device(model, device)



    def stream_vae_to_device(self, vae_model, device: torch.device) -> bool:
        """Stream VAE weights to GPU per-tensor (avoids RAM spike)."""
        from raylight.expansion.comfyui_lazytensors.loader import SafetensorMmapWrapper
        from raylight.distributed_modules.utils import align_model_to_cuda
        
        mmap_cache = getattr(vae_model, "mmap_cache", None)
        
        if not mmap_cache:
            # Fallback: standard model.to()
            print(f"[VAEContext] Using standard .to({device})...")
            if hasattr(vae_model, "first_stage_model"):
                vae_model.first_stage_model.to(device)
            return False
        
        try:
            print(f"[VAEContext] Streaming VAE to {device} per-tensor...")
            wrapper = SafetensorMmapWrapper(mmap_cache)
            
            if hasattr(vae_model, "first_stage_model"):
                transferred = wrapper.stream_to_model(vae_model.first_stage_model, device)
                print(f"[VAEContext] Streamed {transferred} parameters to {device}")
                
                # Align stragglers
                align_model_to_cuda(vae_model.first_stage_model)
                print("[VAEContext] Aligned VAE stragglers to CUDA")
            
            return True
        except Exception as e:
            print(f"[VAEContext] Streaming transfer failed: {e}, falling back...")
            if hasattr(vae_model, "first_stage_model"):
                vae_model.first_stage_model.to(device)
            return False

    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """Zero-copy soft-offload for VAE."""
        
        if model is None:
            return
        
        mmap_cache = getattr(model, "mmap_cache", None)
        first_stage = getattr(model, "first_stage_model", None)
        
        if mmap_cache and first_stage:
            # ZERO-COPY
            print(f"[VAEContext {config.local_rank}] Zero-Copy Offload: Restoring mmap pointers...")
            
            # Param map logic...
            param_map = {name: param for name, param in first_stage.named_parameters()}
            buffer_map = {name: buf for name, buf in first_stage.named_buffers()}
            
            restored = 0
            for name, mmap_tensor in mmap_cache.items():
                target_param = None
                if name in param_map:
                    target_param = param_map[name]
                elif name in buffer_map:
                    target_param = buffer_map[name]
                
                if target_param is None:
                    simple_name = name.replace("first_stage_model.", "")
                    if simple_name in param_map:
                        target_param = param_map[simple_name]
                    elif simple_name in buffer_map:
                         target_param = buffer_map[simple_name]

                if target_param is not None:
                    target_param.data = mmap_tensor
                    restored += 1
            
            print(f"[VAEContext {config.local_rank}] Zero-Copy: Restored {restored}/{len(mmap_cache)} tensors to mmap.")
            
            # Move stragglers
            stragglers_moved = 0
            for name, param in first_stage.named_parameters():
                if param.device.type == 'cuda':
                    param.data = param.data.to('cpu')
                    stragglers_moved += 1
            for name, buf in first_stage.named_buffers():
                if buf.device.type == 'cuda':
                    buf.data = buf.data.to('cpu')
                    stragglers_moved += 1
            
            if stragglers_moved > 0:
                print(f"[VAEContext {config.local_rank}] Moved {stragglers_moved} stragglers to CPU.")
                    
        else:
            # Fallback
            print(f"[VAEContext {config.local_rank}] VAE Offload: No mmap cache, using standard offload...")
            if first_stage:
                first_stage.to("cpu")
            if hasattr(model, "mmap_cache"):
                model.mmap_cache = None
            
        # Force cleanup
        from raylight.utils.common import cleanup_memory
        cleanup_memory()
        
        mem_now = torch.cuda.memory_allocated(config.device) / 1e9
        print(f"[VAEContext {config.local_rank}] Post-Offload VRAM: {mem_now:.2f}GB")


def get_context(path: str, config: "WorkerConfig", model_type: str = "unet") -> ModelContext:
    """Factory function to select appropriate context based on config and file type.
    
    Args:
        path: Path to model file
        config: Worker configuration
        model_type: "unet" or "vae"
        
    Returns:
        Appropriate ModelContext subclass instance
    """
    use_mmap = config.parallel_dict.get("use_mmap", True)
    
    if model_type == "vae":
        return VAEContext(use_mmap=use_mmap)
        
    if getattr(config, "is_fsdp", False):
        if path.lower().endswith(".gguf"):
            raise ValueError(
                "[Raylight] FSDP is not supported for GGUF models. "
                "GGUF quantization is incompatible with FSDP sharding. "
                "Please use a standard Safetensors model or disable FSDP/Context Parallel to use GGUF."
            )
        return FSDPContext(use_mmap=use_mmap)
    if path.lower().endswith(".gguf"):
        return GGUFContext(use_mmap=use_mmap)
    return LazyTensorContext(use_mmap=use_mmap)
