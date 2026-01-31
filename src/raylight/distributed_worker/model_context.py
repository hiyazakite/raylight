"""Unified model lifecycle management with mmap support.

Provides a common abstraction for loading, offloading, and reloading models
across different formats (GGUF, Safetensors, FSDP) with optional mmap caching.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, TYPE_CHECKING
import os

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
                print("[GGUFContext] Re-hydrating mmap_cache from LRU cache...")
                self.worker.model.mmap_cache = self.worker.state_cache.get(unet_path)
        
        if hasattr(self.worker.model, "load"):
            self.worker.model.load(device, force_patch_weights=True)
            self.worker.model.current_device = device


from raylight.expansion.comfyui_lazytensors.loader import SafetensorMmapWrapper


class LazyTensorContext(ModelContext):
    """Context for Safetensors model loading with streaming mmap support.
    
    Features:
    - Zero-copy mmap sharing via OS page cache (like GGUF)
    - Streaming GPU transfer (per-tensor to avoid RAM spike)
    - Fallback to full clone if streaming fails
    """
    
    def __init__(self, worker: "RayWorker"):
        super().__init__(worker)
        self._streaming_enabled = True  # Can be disabled if issues detected
    
    def load_state_dict_mmap(self, state: ModelState) -> Dict:
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
    
    def instantiate_model_with_fallback(self, sd: Dict, state: ModelState):
        """Try streaming approach with fallback to full clone."""
        import comfy.sd
        
        try:
            if self._streaming_enabled:
                # Streaming: shallow copy + deferred GPU transfer
                model = self.instantiate_model(sd, state)
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
    
    def stream_to_device(self, device: torch.device) -> bool:
        """Stream model weights to GPU per-tensor (avoids RAM spike).
        
        Returns True if streaming was used, False if fallback applied.
        """
        mmap_cache = getattr(self.worker.model, "mmap_cache", None)
        
        if not mmap_cache or not self._streaming_enabled:
            # Fallback: standard model.to()
            print(f"[LazyTensorContext] Using standard .to({device})...")
            if self.worker.model is not None and hasattr(self.worker.model, "model"):
                self.worker.model.model.to(device)
            return False
        
        try:
            print(f"[LazyTensorContext] Streaming to {device} per-tensor...")
            wrapper = SafetensorMmapWrapper(mmap_cache)
            
            if self.worker.model is not None and hasattr(self.worker.model, "model"):
                transferred = wrapper.stream_to_model(self.worker.model.model, device)
                print(f"[LazyTensorContext] Streamed {transferred} parameters to {device}")
            
            return True
        except Exception as e:
            print(f"[LazyTensorContext] Streaming transfer failed: {e}, falling back...")
            if self.worker.model is not None and hasattr(self.worker.model, "model"):
                self.worker.model.model.to(device)
            return False
    
    def offload(self):
        """Pointer-swap offload: swap GPU tensors back to mmap refs (like GGUF)."""
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import swap_model_to_mmap
        
        if self.worker.model is not None:
            print(f"[LazyTensorContext {self.worker.local_rank}] Soft-Offload: Performing Pointer Swap...")
            
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
            
            print(f"[LazyTensorContext {self.worker.local_rank}] Soft-Offload Complete.")
        else:
            self.worker.is_model_loaded = False
    
    def hot_load(self, device: torch.device):
        """Hot reload: re-materialize from mmap refs to GPU (like GGUF)."""
        print(f"[LazyTensorContext] Hot reload to {device}...")
        
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
            self.worker.local_rank,
            self.worker.device_mesh,
            self.worker.is_cpu_offload,
            model_options=model_options
        )
        
        if model is None:
            raise RuntimeError(f"Could not load FSDP model: {state.unet_path}")
        
        self.worker.state_dict = state_dict
        return model
    
    def offload(self):
        """
        FSDP offload: release both model and state dict
        FSDP requires full reload currently due to its nature (sharding)
        TODO: Investigate hot loading for FSDP - we could flush the shards, remap the mmap cache, and reload the state dict but this is a lot of work
        """
        print(f"[FSDPContext {self.worker.local_rank}] FSDP Hard-Offload: Releasing all resources...")
        
        # 0. Clear GPU refs (weight_functions, patches) that might hold cyclic refs
        self.worker.lora_manager.clear_gpu_refs()
        
        if self.worker.model is not None:
            # Explicitly release DTensor/FSDP storage
            if hasattr(self.worker.model, "release_memory"):
                self.worker.model.release_memory()

        # 2. Trigger shared cleanup (clears LoRA tracking, moves buffers to CPU, deleted cached attributes)
        self.worker._common_offload_cleanup()
        
        # 3. Destroy references to allow GC
        self.worker.model = None
        self.worker.state_dict = None
        
        # 4. Final VRAM flush
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def hot_load(self, device: torch.device):
        """
        FSDP requires full reload currently due to its nature (sharding)
        TODO: See offload for TODO
        """
        state = ModelState(**self.worker.reload_params)
        self.load(state)



class VAEContext(LazyTensorContext):
    """Context for VAE model loading with streaming mmap support."""
    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """VAE: No prefix stripping needed (unlike UNET)."""
        return sd
        
    def instantiate_model(self, sd: Dict, state: ModelState):
        """Instantiate VAE model using comfy.sd.VAE (Standard ComfyUI path)."""
        import comfy.sd
        import comfy.utils
        
        print(f"[VAEContext] Standard VAE init with {len(sd)} keys")
        
        # CRITICAL: Load metadata from safetensor file
        # Metadata contains VAE config that's essential for proper version detection
        # The old working code used: comfy.utils.load_torch_file(vae_path, return_metadata=True)
        metadata = None
        try:
            # Load just the metadata (not the tensors - those are already in sd)
            import safetensors
            with safetensors.safe_open(state.unet_path, framework="pt") as f:
                metadata = f.metadata()
            if metadata:
                print(f"[VAEContext] Loaded metadata: contains 'config': {'config' in metadata}")
        except Exception as e:
            print(f"[VAEContext] Warning: Could not load metadata: {e}")
        
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
             
             # NOTE: Do NOT move to GPU here. Let the caller (VaeManager) handle 
             # device placement via streaming or .to() so we can control the flow.

        except Exception as e:
             raise RuntimeError(f"Could not load VAE model: {state.unet_path}. Error: {e}")
             
        # Store original sd for potential streaming optimization
        vae_model.mmap_cache = sd
        return vae_model

    def stream_vae_to_device(self, vae_model, device: torch.device) -> bool:
        """Stream VAE weights to GPU per-tensor (avoids RAM spike).
        
        Handles both the streaming of weights from mmap AND the alignment 
        of stragglers (buffers not in safetensor file) to GPU.
        
        Returns True if streaming was used, False if fallback applied.
        """
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
                
                # Align stragglers (buffers not in safetensor) to CUDA
                # This handles components like timestep_scale_multiplier
                align_model_to_cuda(vae_model.first_stage_model)
                print("[VAEContext] Aligned VAE stragglers to CUDA")
            
            return True
        except Exception as e:
            print(f"[VAEContext] Streaming transfer failed: {e}, falling back...")
            if hasattr(vae_model, "first_stage_model"):
                vae_model.first_stage_model.to(device)
            return False

    def offload(self, vae_model=None, release_reference: bool = False):
        """Zero-copy soft-offload for VAE: Restores mmap pointers instead of copying to CPU.
        
        This matches the GGUF/LazyTensorContext pattern for consistent memory handling.
        
        Args:
            vae_model: Optional VAE model to offload. If None, uses worker.vae_model.
            release_reference: If True, also clears worker.vae_model reference.
        """
        from raylight.utils.common import cleanup_memory
        
        model = vae_model if vae_model is not None else getattr(self.worker, 'vae_model', None)
        
        if model is None:
            return
        
        mmap_cache = getattr(model, "mmap_cache", None)
        first_stage = getattr(model, "first_stage_model", None)
        
        if mmap_cache and first_stage:
            # ZERO-COPY: Restore mmap pointers instead of copying to CPU
            print(f"[VAEContext {self.worker.local_rank}] Zero-Copy Offload: Restoring mmap pointers...")
            
            # Build param map for first_stage_model
            param_map = {name: param for name, param in first_stage.named_parameters()}
            buffer_map = {name: buf for name, buf in first_stage.named_buffers()}
            
            restored = 0
            for name, mmap_tensor in mmap_cache.items():
                # VAE keys map directly (no prefix stripping needed unlike diffusion models)
                if name in param_map:
                    param_map[name].data = mmap_tensor
                    restored += 1
                elif name in buffer_map:
                    buffer_map[name].data = mmap_tensor
                    restored += 1
            
            print(f"[VAEContext {self.worker.local_rank}] Zero-Copy: Restored {restored}/{len(mmap_cache)} tensors to mmap.")
            
            # Move any remaining GPU tensors to CPU (stragglers not in mmap)
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
                print(f"[VAEContext {self.worker.local_rank}] Moved {stragglers_moved} stragglers to CPU.")
                    
        else:
            # Fallback: standard .to('cpu') if no mmap cache available
            print(f"[VAEContext {self.worker.local_rank}] VAE Offload: No mmap cache, using standard offload...")
            if first_stage:
                first_stage.to("cpu")
            # Only clear mmap cache in fallback path (not needed)
            if hasattr(model, "mmap_cache"):
                model.mmap_cache = None
            
        # Force cleanup
        cleanup_memory()
        
        # Report VRAM status (consistent with other contexts)
        mem_now = torch.cuda.memory_allocated(self.worker.device) / 1e9
        print(f"[VAEContext {self.worker.local_rank}] Post-Offload VRAM: {mem_now:.2f}GB")


def get_context(worker: "RayWorker", path: str, model_type: str = "unet") -> ModelContext:
    """Factory function to select appropriate context based on config and file type.
    
    Args:
        worker: RayWorker instance
        path: Path to model file
        model_type: "unet" or "vae"
        
    Returns:
        Appropriate ModelContext subclass instance
    """
    if model_type == "vae":
        return VAEContext(worker)
        
    if worker.parallel_dict.get("is_fsdp"):
        return FSDPContext(worker)
    if path.lower().endswith(".gguf"):
        return GGUFContext(worker)
    return LazyTensorContext(worker)

