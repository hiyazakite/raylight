"""Unified model lifecycle management with mmap support.

Provides a common abstraction for loading, offloading, and reloading models
across different formats (GGUF, Safetensors, FSDP) with optional mmap caching.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, TYPE_CHECKING
import os
import logging

import torch

if TYPE_CHECKING:
    from .ray_worker import RayWorker


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
    
    def __init__(self, worker: "RayWorker"):
        self.worker = worker
        self.use_mmap = worker.parallel_dict.get("use_mmap", True)
    
    # ─── Abstract Methods (format-specific) ──────────────────
    
    @abstractmethod
    def load_state_dict_mmap(self, state: ModelState) -> Dict[str, torch.Tensor]:
        """Load state dict with mmap (zero-copy where possible)."""
        pass
    
    @abstractmethod
    def load_state_dict_standard(self, state: ModelState) -> Dict[str, torch.Tensor]:
        """Load state dict without mmap (fallback)."""
        pass
    
    @abstractmethod
    def instantiate_model(self, sd: Dict[str, torch.Tensor], state: ModelState) -> Any:
        """Create ModelPatcher from state dict."""
        pass
    
    @abstractmethod
    def hot_load(self, device: torch.device):
        """Fast VRAM transfer for soft-reloaded models."""
        pass
    
    def offload(self):
        """Standard offload: Move weights to CPU and release tracking."""
        print(f"[{self.__class__.__name__}] Standard Offload: Releasing VRAM...")
        
        # Share the common offload cleanup logic
        self.worker._common_offload_cleanup()
        
        # Clear model and tracking
        self.worker.model = None
        self.worker.current_unet_path = None
        self.worker.current_sd_ref = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Optional pre-processing of state dict before instantiation.
        
        Args:
            sd: Original state dict
            
        Returns:
            Processed state dict
        """
        return sd
    
    # ─── Shared Logic ────────────────────────────────────────
    
    def load(self, state: ModelState):
        """Full load with cache check (shared logic for all formats)."""
        cache = self.worker.state_cache
        
        # 1. Cache check (only if mmap enabled)
        if self.use_mmap and state.cache_key in cache:
            print(f"[ModelContext] LRU Cache Hit: {os.path.basename(state.cache_key)}")
            sd = cache.get(state.cache_key)
        else:
            # 2. Disk load (mmap or standard based on toggle)
            if self.use_mmap:
                print(f"[ModelContext] Mmap Load: {os.path.basename(state.cache_key)}")
                sd = self.load_state_dict_mmap(state)
                cache.put(state.cache_key, sd)
            else:
                print(f"[ModelContext] Standard Load: {os.path.basename(state.cache_key)}")
                sd = self.load_state_dict_standard(state)
        
        # 3. Instantiate model
        if sd is None:
            raise RuntimeError(f"Failed to load state dict for {state.unet_path}")
            
        self.worker.model = self.instantiate_model(sd, state)
        self._post_load(state, sd)
    
    def reload_if_needed(self, target_device: torch.device):
        """Shared reload logic - checks device and triggers appropriate reload."""
        model = self.worker.model
        current = getattr(model, "current_device", None) if model else None
        
        if model is not None and str(current) == str(target_device):
            print(f"[ModelContext] Already on {target_device}, skipping reload.")
            return
        
        if model is not None:
            # Soft reload - model object exists but on wrong device
            print(f"[ModelContext] Hot-loading to {target_device}...")
            self.hot_load(target_device)
        else:
            # Full reload - model is None
            print(f"[ModelContext] Full reload to {target_device}...")
            state = ModelState(**self.worker.reload_params)
            self.load(state)
    
    def _post_load(self, state: ModelState, sd: Dict):
        """Common post-load setup for all formats."""
        self.worker.reload_params = state.to_dict()
        if self.worker.model is not None:
            self.worker.model.unet_path = state.unet_path
            
            if self.use_mmap:
                self.worker.model.mmap_cache = sd
        
        # Clear LoRA tracking for fresh re-application
        if hasattr(self.worker, "lora_manager"):
            self.worker.lora_manager.clear_tracking()


class GGUFContext(ModelContext):
    """Context for GGUF model loading with mmap support."""
    
    def load_state_dict_mmap(self, state: ModelState) -> Dict:
        from raylight.expansion.comfyui_gguf.loader import gguf_sd_loader
        sd, extra = gguf_sd_loader(state.unet_path)
        self.worker.gguf_metadata = extra.get("metadata", {})
        return sd
    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """GGUF: Strip prefixes and prepare for detection."""
        keys = list(sd.keys())
        
        # The auto-detection (unet_prefix_from_state_dict) failed in previous runs (detected 'model.'), so we force 'diffusion_model.'
        if any(k.startswith("diffusion_model.") for k in keys[:5]):
             print(f"[GGUFContext {self.worker.local_rank}] Detected 'diffusion_model.' prefix. Force-stripping for detection...")
             import comfy.utils
             sd = comfy.utils.state_dict_prefix_replace(sd, {"diffusion_model.": ""}, filter_keys=True)
        
        return sd
    
    def load_state_dict_standard(self, state: ModelState) -> Dict:
        # GGUF library always uses mmap internally
        return self.load_state_dict_mmap(state)
    
    def instantiate_model(self, sd: Dict, state: ModelState):
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
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = getattr(self.worker, "gguf_metadata", {})
        
        model = comfy.sd.load_diffusion_model_state_dict(
            isolated, 
            model_options={"custom_operations": ops},
            **kwargs
        )
        
        if model is None:
            raise RuntimeError(f"Could not load GGUF model: {state.unet_path}")
        
        model = GGUFModelPatcher.clone(model)
        model.gguf_metadata = getattr(self.worker, "gguf_metadata", {})
        return model
    
    def offload(self):
        """GGUF Soft-Offload: Restore mmap pointers to clear VRAM without copies."""
        if self.worker.model is not None:
            print(f"[GGUFContext {self.worker.local_rank}] GGUF Soft-Offload: Performing Zero-Copy Pointer Swap...")
            
            # 1. Clear LoRA GPU refs via manager
            self.worker.lora_manager.clear_gpu_refs()
            
            # 2. Restore mmap pointers
            self.worker.model.unpatch_model(device_to=torch.device('cpu'), unpatch_weights=True)
            self.worker.model.current_device = torch.device('cpu')
            
            # 3. Store in worker-level mmap cache to survive model swaps
            mmap_cache = getattr(self.worker.model, "mmap_cache", None)
            if mmap_cache:
                unet_path = getattr(self.worker.model, "unet_path", None)
                if unet_path:
                    self.worker.worker_mmap_cache[unet_path] = mmap_cache
            
            # 4. Shared cleanup
            self.worker._common_offload_cleanup()
            print(f"[GGUFContext {self.worker.local_rank}] GGUF Soft-Offload Complete.")
        else:
            self.worker.is_model_loaded = False
    
    def hot_load(self, device: torch.device):
        if self.worker.model is None:
            return

        # Re-hydrate mmap_cache from worker cache if needed
        if not getattr(self.worker.model, "mmap_cache", None):
            unet_path = getattr(self.worker.model, "unet_path", None)
            if unet_path and unet_path in self.worker.state_cache:
                print(f"[GGUFContext] Re-hydrating mmap_cache from LRU cache...")
                self.worker.model.mmap_cache = self.worker.state_cache.get(unet_path)
        
        if hasattr(self.worker.model, "load"):
            self.worker.model.load(device, force_patch_weights=True)
            self.worker.model.current_device = device


from raylight.expansion.comfyui_lazytensors.loader import SafetensorMmapWrapper


# Check for optional fastsafetensors
try:
    import fastsafetensors
    HAS_FASTSAFETENSORS = True
except ImportError:
    HAS_FASTSAFETENSORS = False


class SafetensorsContext(ModelContext):
    """Context for Safetensors model loading with streaming mmap support.
    
    Features:
    - Zero-copy mmap sharing via OS page cache (like GGUF)
    - Streaming GPU transfer (per-tensor to avoid RAM spike)
    - Optional fastsafetensors for GDS acceleration
    - Fallback to full clone if streaming fails
    """
    
    def __init__(self, worker: "RayWorker"):
        super().__init__(worker)
        self.use_fastsafe = worker.parallel_dict.get("use_fastsafe", False) and HAS_FASTSAFETENSORS
        self._streaming_enabled = True  # Can be disabled if issues detected
    
    def load_state_dict_mmap(self, state: ModelState) -> Dict:
        if self.use_fastsafe:
            try:
                print(f"[SafetensorsContext] Using fastsafetensors (GDS if available)...")
                from fastsafetensors import SafeTensorsFileLoader, SingleGroup
                
                # Use SingleGroup for single-device usage (no ProcessGroup needed)
                loader = SafeTensorsFileLoader(state.unet_path, pg=SingleGroup(), device=str(self.worker.device))
                
                # Map the file for rank 0
                loader.add_filenames({0: [state.unet_path]})
                
                # Load to device
                buffer = loader.copy_file_to_device()
                
                # Build state dict
                sd = {}
                for key in buffer.get_keys():
                    sd[key] = buffer.get_tensor(key)
                
                print(f"[SafetensorsContext] fastsafetensors loaded {len(sd)} tensors")
                return sd
            except Exception as e:
                print(f"[SafetensorsContext] fastsafetensors failed: {e}, falling back to standard mmap")
                import safetensors.torch
                return safetensors.torch.load_file(state.unet_path, device="cpu")
        else:
            import safetensors.torch
            return safetensors.torch.load_file(state.unet_path, device="cpu")
    
    def load_state_dict_standard(self, state: ModelState) -> Dict:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path)
    
    def instantiate_model(self, sd: Dict, state: ModelState):
        """Instantiate model with lazy state dict (zero-copy until load)."""
        import comfy.sd
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps
        
        print(f"[SafetensorsContext] Wrapping state dict with LazySafetensors...")
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
    
    def instantiate_model_with_fallback(self, sd: Dict, state: ModelState):
        """Try streaming approach with fallback to full clone."""
        import comfy.sd
        
        try:
            if self._streaming_enabled:
                # Streaming: shallow copy + deferred GPU transfer
                model = self.instantiate_model(sd, state)
                print(f"[SafetensorsContext] Streaming instantiation successful")
                return model
        except Exception as e:
            print(f"[SafetensorsContext] Streaming failed: {e}")
            print(f"[SafetensorsContext] Falling back to full clone...")
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
    
    def stream_to_device(self, device: torch.device) -> bool:
        """Stream model weights to GPU per-tensor (avoids RAM spike).
        
        Returns True if streaming was used, False if fallback applied.
        """
        mmap_cache = getattr(self.worker.model, "mmap_cache", None)
        
        if not mmap_cache or not self._streaming_enabled:
            # Fallback: standard model.to()
            print(f"[SafetensorsContext] Using standard .to({device})...")
            if self.worker.model is not None and hasattr(self.worker.model, "model"):
                self.worker.model.model.to(device)
            return False
        
        try:
            print(f"[SafetensorsContext] Streaming to {device} per-tensor...")
            wrapper = SafetensorMmapWrapper(mmap_cache)
            
            if self.worker.model is not None and hasattr(self.worker.model, "model"):
                transferred = wrapper.stream_to_model(self.worker.model.model, device)
                print(f"[SafetensorsContext] Streamed {transferred} parameters to {device}")
            
            return True
        except Exception as e:
            print(f"[SafetensorsContext] Streaming transfer failed: {e}, falling back...")
            if self.worker.model is not None and hasattr(self.worker.model, "model"):
                self.worker.model.model.to(device)
            return False
    
    def offload(self):
        """Pointer-swap offload: swap GPU tensors back to mmap refs (like GGUF)."""
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import swap_model_to_mmap
        
        if self.worker.model is not None:
            print(f"[SafetensorsContext {self.worker.local_rank}] Safetensors Soft-Offload: Performing Pointer Swap...")
            
            # 1. Clear LoRA GPU refs via manager
            self.worker.lora_manager.clear_gpu_refs()
            
            # 2. Extract model
            model = self.worker.model
            
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
                    self.worker.worker_mmap_cache[unet_path] = mmap_cache
            
            model.current_device = torch.device('cpu')
            
            # 5. Shared cleanup
            self.worker._common_offload_cleanup()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[SafetensorsContext {self.worker.local_rank}] Safetensors Soft-Offload Complete.")
        else:
            self.worker.is_model_loaded = False
    
    def hot_load(self, device: torch.device):
        """Hot reload: re-materialize from mmap refs to GPU (like GGUF)."""
        print(f"[SafetensorsContext] Hot reload to {device}...")
        
        # Model structure still exists, just reload weights to GPU
        if self.worker.model is not None and hasattr(self.worker.model, 'load'):
            self.worker.model.load(device)
            self.worker.model.current_device = device
        else:
            # Fallback: full reload from cache
            state = ModelState(**self.worker.reload_params)
            self.load(state)
            if self.worker.model is not None and hasattr(self.worker.model, 'load'):
                self.worker.model.load(device)


class FSDPContext(ModelContext):
    """Context for FSDP model loading with optional mmap for safetensors."""
    
    def load_state_dict_mmap(self, state: ModelState) -> Dict:
        # Use safetensors mmap if file is .safetensors
        if state.unet_path.lower().endswith(".safetensors"):
            import safetensors.torch
            return safetensors.torch.load_file(state.unet_path, device="cpu")
        return self.load_state_dict_standard(state)
    
    def load_state_dict_standard(self, state: ModelState) -> Dict:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path)
    
    def instantiate_model(self, sd: Dict, state: ModelState):
        from raylight.comfy_dist.sd import fsdp_load_diffusion_model_stat_dict
        
        model, state_dict = fsdp_load_diffusion_model_stat_dict(
            sd,
            self.worker.local_rank,
            self.worker.device_mesh,
            self.worker.is_cpu_offload,
            model_options=state.model_options
        )
        
        if model is None:
            raise RuntimeError(f"Could not load FSDP model: {state.unet_path}")
        
        self.worker.state_dict = state_dict
        return model
    
    def offload(self):
        # FSDP offload: release both model and state dict
        self.worker.model = None
        self.worker.state_dict = None
    
    def hot_load(self, device: torch.device):
        # FSDP requires full reload
        state = ModelState(**self.worker.reload_params)
        self.load(state)


def get_context(worker: "RayWorker", path: str) -> ModelContext:
    """Factory function to select appropriate context based on config and file type.
    
    Args:
        worker: RayWorker instance
        path: Path to model file
        
    Returns:
        Appropriate ModelContext subclass instance
    """
    if worker.parallel_dict.get("is_fsdp"):
        return FSDPContext(worker)
    if path.lower().endswith(".gguf"):
        return GGUFContext(worker)
    return SafetensorsContext(worker)
