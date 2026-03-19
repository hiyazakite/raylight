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
from raylight.utils.cache import CachedState
from raylight.distributed_modules.fsdp_utils import prefetch_state_dict
from concurrent.futures import ThreadPoolExecutor


if TYPE_CHECKING:
    from .worker_config import WorkerConfig
    from .managers.lora_manager import LoraManager


# ---------------------------------------------------------------------------
# VRAM budget helpers
# ---------------------------------------------------------------------------

def _compute_vram_budget(
    device: torch.device,
    model_size: int,
    min_inference_mem: Optional[int] = None,
    vram_limit_bytes: int = 0,
) -> int:
    """Decide how much VRAM to allocate for model weights.

    Uses the same heuristic as ComfyUI's ``load_models_gpu`` so the split
    between CUDA-resident and CPU-offloaded modules stays consistent.

    Parameters
    ----------
    vram_limit_bytes : int
        Optional hard cap on VRAM for weights (from user setting).
        0 = auto (use all available minus inference reserve).

    Returns
    -------
    0
        Model fits entirely in free VRAM → caller should do a full load.
    >0
        Partial budget in bytes → pass as ``lowvram_model_memory`` to
        ``model.load()`` so ComfyUI puts only this much on CUDA and
        streams the rest per-layer via ``comfy_cast_weights`` hooks.
    """
    if not torch.cuda.is_available():
        return 0

    # Mirror ComfyUI: minimum_inference_memory ≈ 0.8 GB + reserve_vram
    if min_inference_mem is None:
        try:
            import comfy.model_management as _mm
            min_inference_mem = int(_mm.minimum_inference_memory())
        except Exception:
            min_inference_mem = int(0.8 * (1024 ** 3))  # 0.8 GB fallback

    free = torch.cuda.mem_get_info(device)[0]

    # User-specified VRAM cap: treat as the effective "free" memory.
    if vram_limit_bytes > 0:
        effective_free = min(free, vram_limit_bytes)
    else:
        effective_free = free

    # Comfortable headroom → full load
    if model_size <= effective_free - min_inference_mem:
        return 0

    # Budget = effective free minus inference reserve (at least 1 byte to
    # avoid the ``lowvram_model_memory=0`` path which ComfyUI treats as
    # "load all as lowvram").
    budget = max(1, effective_free - min_inference_mem)
    return budget


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
    
    def __init__(self, use_mmap: bool = True, cache_in_ram: bool = True):
        self.use_mmap = use_mmap
        self.cache_in_ram = cache_in_ram
    
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
    def instantiate_model(self, sd: Dict[str, torch.Tensor], state: ModelState, config: "WorkerConfig", metadata: Any = None, **kwargs) -> Any:
        """Create ModelPatcher from state dict."""
        pass
    
    @abstractmethod
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """Fast VRAM transfer for soft-reloaded models."""
        pass
    
    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig") -> bool:
        """Standard offload: Move weights to CPU and release tracking.
        
        Returns:
            bool: True if the model object was destroyed/invalidated (Hard Offload).
                  False if the model object persists (Soft Offload).
        """
        print(f"[{self.__class__.__name__}] Standard Offload: Releasing VRAM...")
        
        # Share the common offload cleanup logic
        self._common_offload_cleanup(model, lora_manager, config)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return False

    def _common_offload_cleanup(self, model: Any, lora_manager: Optional["LoraManager"], config: "WorkerConfig"):
        """Shared post-offload cleanup for all model formats."""
        # Clear LoRA tracking for fresh re-application after reload
        if lora_manager:
            lora_manager.clear_tracking()
            print(f"[RayWorker {config.local_rank}] LoRA tracking cleared.")
        
        # Move weights and buffers to CPU if model still exists
        if model is not None:
            diffusion_model = getattr(model, "model", None)
            if diffusion_model is not None:
                try:
                    # Move whole module to CPU to ensure weights are released from VRAM
                    # For GGUF, this is usually a no-op as weights are already CPU mmap'd
                    # and unpatch_model handles it, but for standard models/VAEs it's critical.
                    diffusion_model.to('cpu')
                    
                    for name, buf in diffusion_model.named_buffers():
                        if buf is not None and buf.device.type == 'cuda':
                            buf.data = buf.data.to('cpu', non_blocking=True)
                except Exception as e:
                    print(f"[RayWorker {config.local_rank}] Module/Buffer cleanup warning: {e}")
                
                # Sync any non_blocking transfers before proceeding
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Delete cached PE tensors
                inner = getattr(diffusion_model, 'diffusion_model', None)
                for attr in ['_cached_pe', 'cached_pe', 'pe_cache', '_pe']:
                    if hasattr(diffusion_model, attr):
                        delattr(diffusion_model, attr)
                    if inner is not None and hasattr(inner, attr):
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

    def load(self, state: ModelState, config: "WorkerConfig", state_cache: Any) -> Any:
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
        cached = None
        if self.cache_in_ram:
            cached = state_cache.get(state.cache_key)
            if cached is not None:
                print(f"[ModelContext] Cache Hit: {os.path.basename(state.cache_key)}")
                
                # Handle structured CachedState
                if isinstance(cached, CachedState):
                    sd = cached.state_dict
                    metadata = cached.metadata
                    if cached.checksum:
                        verify_model_checksum(sd, cached.checksum, metadata, context_tag="ModelContext")
                else:
                     # Should not happen if cache is strictly typed
                     print(f"[ModelContext] Warning: Invalid cache entry type: {type(cached)}")
                     return None, {}
        
        if sd is None:
            # 2. Disk load (mmap or standard based on toggle)
            if self.use_mmap:
                print(f"[ModelContext] Mmap Load: {os.path.basename(state.cache_key)}")
                res = self.load_state_dict_mmap(state, config)
                
                # Metadata aware return
                if isinstance(res, tuple) and len(res) == 2:
                    sd, metadata = res
                else:
                    sd = res
                
                # Compute checksum
                checksum = compute_model_checksum(sd, metadata)
                
                # Store as CachedState ONLY if caching enabled
                if self.cache_in_ram:
                    cached_entry = CachedState(state_dict=sd, metadata=metadata, checksum=checksum)
                    state_cache.put(state.cache_key, cached_entry)
                    print(f"[ModelContext] Cached model with checksum: {checksum}")
                else:
                    print("[ModelContext] Skipping RAM Cache (cache_in_ram=False)")

            else:
                print(f"[ModelContext] Standard Load: {os.path.basename(state.cache_key)}")
                res = self.load_state_dict_standard(state, config)
                if isinstance(res, tuple) and len(res) == 2:
                    sd, metadata = res
                else:
                    sd = res
        
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
        if model is not None:
             model.unet_path = state.unet_path
             model.load_config = state
             if self.use_mmap and self.cache_in_ram:
                 model.mmap_cache = sd

        return model
    

    
    def reload_if_needed(self, model: Any, target_device: torch.device, config: "WorkerConfig", state_cache: Any) -> Any:
        """Shared reload logic - checks device and triggers appropriate reload.
        
        Returns:
            model - potentially new model instance if full reload
        """
        current = getattr(model, "current_device", None) if model else None
        
        if model is not None and str(current) == str(target_device):
            print(f"[ModelContext] Already on {target_device}, skipping reload.")
            return model
        
        if model is not None:
            # Soft reload - model object exists but on wrong device
            print(f"[ModelContext] Hot-loading to {target_device}...")
            # Use attached load_config if needed (though hot_load doesn't rely on it usually)
            reload_params = model.load_config.to_dict() if hasattr(model, "load_config") else {}
            self.hot_load(model, target_device, reload_params, state_cache)
            return model
        else:
             raise RuntimeError("Cannot reload model: Model is None (Lost context).")


class GGUFContext(ModelContext):
    """Context for GGUF model loading with mmap support."""

    # ─── Pinned mmap_cache swap ──────────────────────────────

    def load(self, state: ModelState, config: "WorkerConfig", state_cache: Any) -> Any:
        """Override base load to pin mmap_cache for DMA-speed reloads.

        After the base class sets ``model.mmap_cache = sd`` (mmap-backed),
        we swap every tensor with a pinned-RAM copy.  All subsequent
        ``model.load()`` / ``unpatch_model()`` read from pinned tensors,
        so CUDA transfers use DMA instead of page-faulting through mmap.

        For multi-GPU DP/xDiT the pinned copy lives in ``/dev/shm`` so
        all workers share ONE host-RAM copy instead of N.
        """
        model = super().load(state, config, state_cache)

        if model is not None and getattr(model, "mmap_cache", None):
            pinned, shm_handle = self._pin_mmap_cache(
                model.mmap_cache, config, state.unet_path
            )
            if pinned is not None:
                model.mmap_cache = pinned
                # Keep SharedMemory alive as long as model lives
                # (torch.frombuffer views reference the buffer).
                if shm_handle is not None:
                    model._pinned_shm = shm_handle
                # Replace mmap-backed sd in state_cache with the pinned
                # dict so the OS can unmap the GGUF file.
                cached = state_cache.get(state.cache_key)
                if isinstance(cached, CachedState):
                    cached.state_dict = pinned
                # Replace model's own nn.Parameters with pinned copies.
                # GGUF model.load() does NOT move quantised weights to GPU
                # (they stay on CPU, dequantised on-the-fly during forward),
                # so the original mmap-backed GGMLTensors set during
                # instantiate_model remain as the model's parameters.
                # Swapping them here removes the last mmap reference and
                # lets the OS unmap the file.
                self._swap_params_to_pinned(model, pinned)
                print("[GGUFContext] mmap_cache swapped to pinned RAM.")

        return model

    @staticmethod
    def _swap_params_to_pinned(model, pinned: Dict[str, torch.Tensor]) -> None:
        """Replace mmap-backed model parameters with their pinned copies."""
        import gc
        import comfy.utils

        diff_model = getattr(model, "model", None)
        if diff_model is None:
            return

        # Build param name → param lookup once
        param_dict = dict(diff_model.named_parameters())

        swapped = 0
        for key, pinned_tensor in pinned.items():
            # sd keys may or may not have a 'diffusion_model.' prefix
            for full_key in (key, f"diffusion_model.{key}"):
                if full_key in param_dict:
                    comfy.utils.set_attr_param(diff_model, full_key, pinned_tensor)
                    swapped += 1
                    break

        if swapped > 0:
            gc.collect()
            print(f"[GGUFContext] Replaced {swapped}/{len(pinned)} model params with pinned tensors.")

    @staticmethod
    def _pin_mmap_cache(
        mmap_cache: Dict[str, torch.Tensor],
        config: "WorkerConfig",
        model_path: str,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Any]:
        """Replace mmap-backed tensors with pinned-RAM copies.

        Returns ``(pinned_dict, shm_handle)``.  The caller must keep
        ``shm_handle`` alive for as long as the pinned views are in use
        (only non-None for the shared-memory multi-GPU path).

        Single-GPU  → private ``pin_memory()`` per tensor.
        Multi-GPU   → one shared ``/dev/shm`` buffer + ``cudaHostRegister``.
        FSDP        → skip (shards differ across ranks).
        """
        is_shared = config.global_world_size > 1 and not config.is_fsdp
        if is_shared:
            return GGUFContext._pin_mmap_shared(mmap_cache, config, model_path)
        else:
            return GGUFContext._pin_mmap_private(mmap_cache), None

    @staticmethod
    def _pin_mmap_private(mmap_cache: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single-GPU: pre-allocate pinned destination + copy once (no double-copy)."""
        from raylight.expansion.comfyui_gguf.ops import GGMLTensor

        pinned: Dict[str, torch.Tensor] = {}
        total_bytes = 0
        for key, tensor in mmap_cache.items():
            is_ggml = isinstance(tensor, GGMLTensor)
            raw = tensor.as_subclass(torch.Tensor) if is_ggml else tensor
            # Pre-allocate pinned destination and copy directly — avoids
            # clone() (pageable alloc) + pin_memory() (second alloc + copy).
            p = torch.empty_like(raw).pin_memory()
            p.copy_(raw)
            total_bytes += p.nbytes
            if is_ggml:
                pinned[key] = GGMLTensor(
                    p,
                    tensor_type=tensor.tensor_type,
                    tensor_shape=tensor.tensor_shape,
                )
            else:
                pinned[key] = p
        print(
            f"[GGUFContext] Pinned mmap→RAM (private): "
            f"{len(pinned)} tensors, {total_bytes / 1e9:.2f} GB"
        )
        return pinned

    @staticmethod
    def _pin_mmap_shared(
        mmap_cache: Dict[str, torch.Tensor],
        config: "WorkerConfig",
        model_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Multi-GPU shared: one ``/dev/shm`` buffer + ``cudaHostRegister``.

        Returns ``(pinned_dict, shm_handle)``.

        Rank 0 (writer) creates the segment and copies mmap data in.
        Other ranks wait at a barrier, then attach and create views.
        All ranks get a dict of pinned GGMLTensor views into the same
        physical pages — N workers share 1× model-size of host RAM.
        """
        import ctypes
        import hashlib
        import time
        from multiprocessing.shared_memory import SharedMemory
        import torch.distributed as dist
        from raylight.expansion.comfyui_gguf.ops import GGMLTensor
        from raylight.distributed_modules.pinned_cache import (
            _align_up, _cuda_host_register, _ALIGNMENT,
        )

        is_writer = config.local_rank == 0
        cache_id = hashlib.md5(model_path.encode()).hexdigest()[:16]
        shm_name = f"raylight_gguf_{cache_id}"

        # --- deterministic layout from sorted keys -----------------------
        layout = []   # (key, offset, nbytes, storage_shape, dtype, tensor_type, tensor_shape)
        offset = 0
        for key in sorted(mmap_cache.keys()):
            tensor = mmap_cache[key]
            is_ggml = isinstance(tensor, GGMLTensor)
            raw = tensor.as_subclass(torch.Tensor) if is_ggml else tensor
            aligned = _align_up(offset)
            nb = raw.nbytes
            layout.append((
                key, aligned, nb,
                tuple(raw.shape), raw.dtype,
                getattr(tensor, "tensor_type", None),
                getattr(tensor, "tensor_shape", raw.shape),
            ))
            offset = aligned + nb
        total_bytes = _align_up(offset)

        if total_bytes == 0:
            return mmap_cache, None  # nothing to pin

        # --- writer: create + copy + register + barrier ------------------
        if is_writer:
            # Clean up stale segment
            try:
                old = SharedMemory(name=shm_name, create=False)
                old.close(); old.unlink()
            except FileNotFoundError:
                pass

            shm = SharedMemory(name=shm_name, create=True, size=total_bytes)
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
            ok = _cuda_host_register(ptr, total_bytes)
            if not ok:
                print("[GGUFContext] WARNING: cudaHostRegister failed (DMA will use staging buffer).")

            for key, off, nb, s_shape, dtype, tt, ts in layout:
                src = mmap_cache[key]
                raw = src.as_subclass(torch.Tensor) if isinstance(src, GGMLTensor) else src
                dst = torch.frombuffer(memoryview(shm.buf)[off:off + nb], dtype=dtype).reshape(s_shape)
                dst.copy_(raw)

            if dist.is_initialized():
                dist.barrier()

            pinned: Dict[str, torch.Tensor] = {}
            for key, off, nb, s_shape, dtype, tt, ts in layout:
                view = torch.frombuffer(memoryview(shm.buf)[off:off + nb], dtype=dtype).reshape(s_shape)
                if tt is not None:
                    pinned[key] = GGMLTensor(view, tensor_type=tt, tensor_shape=ts)
                else:
                    pinned[key] = view

            print(
                f"[GGUFContext] Pinned mmap→shm (Writer): "
                f"{len(pinned)} tensors, {total_bytes / 1e9:.2f} GB "
                f"(shm={shm_name})"
            )
            return pinned, shm

        # --- reader: barrier + attach + create views ---------------------
        if dist.is_initialized():
            dist.barrier()

        retries = 50
        for i in range(retries):
            try:
                shm = SharedMemory(name=shm_name, create=False)
                break
            except FileNotFoundError:
                if i < retries - 1:
                    time.sleep(0.2)
        else:
            raise RuntimeError(
                f"[GGUFContext] Reader timed out waiting for shm '{shm_name}'"
            )

        if shm.size < total_bytes:
            raise RuntimeError(
                f"[GGUFContext] shm size mismatch: need {total_bytes}, got {shm.size}"
            )

        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        _cuda_host_register(ptr, total_bytes)

        pinned = {}
        for key, off, nb, s_shape, dtype, tt, ts in layout:
            view = torch.frombuffer(memoryview(shm.buf)[off:off + nb], dtype=dtype).reshape(s_shape)
            if tt is not None:
                pinned[key] = GGMLTensor(view, tensor_type=tt, tensor_shape=ts)
            else:
                pinned[key] = view

        print(
            f"[GGUFContext] Pinned mmap→shm (Reader): "
            f"{len(pinned)} tensors, {total_bytes / 1e9:.2f} GB "
            f"(shm={shm_name})"
        )
        return pinned, shm

    # ─── Disk loading ────────────────────────────────────────

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
    
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None, **kwargs):
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
        
        print(f"[GGUFContext DEBUG] Instantiate Config | Dequant: {state.dequant_dtype} | Patch: {state.patch_dtype}")

        
        # Use helper for consistency (optional refactor later, but leaving inline for now to minimize churn)
        
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
        
        model_options = state.model_options.copy()
        model_options["custom_operations"] = ops
        
        # Apply cache_patched_weights if present in model_options
        if model_options.get("cache_patched_weights"):
             setattr(ops.Linear, "cache_patched_weights", True)
             print(f"[GGUFContext] Enabling cache_patched_weights for {state.unet_path}")

        model = comfy.sd.load_diffusion_model_state_dict(
            isolated, 
            model_options=model_options,
            **kwargs
        )
        
        if model is None:
            raise RuntimeError(f"Could not load GGUF model: {state.unet_path}")
        
        model = GGUFModelPatcher.clone(model)
        model.gguf_metadata = gguf_meta

        return model
    

    
    def _apply_ops_config(self, ops, dequant_dtype, patch_dtype):
        """Helper to apply GGUF ops configuration."""
        print(f"[GGUFContext DEBUG] Applying Ops Config | Dequant: {dequant_dtype} | Patch: {patch_dtype}")
        if dequant_dtype and dequant_dtype not in ("default", None):
            if dequant_dtype == "target":
                setattr(ops.Linear, "dequant_dtype", dequant_dtype)
            else:
                setattr(ops.Linear, "dequant_dtype", getattr(torch, dequant_dtype, None))
        
        if patch_dtype and patch_dtype not in ("default", None):
            if patch_dtype == "target":
                setattr(ops.Linear, "patch_dtype", patch_dtype)
            else:
                setattr(ops.Linear, "patch_dtype", getattr(torch, patch_dtype, None))
        
        print(f"[GGUFContext DEBUG] Ops Config Result | Linear.dequant_dtype: {getattr(ops.Linear, 'dequant_dtype', 'None')} | Linear.patch_dtype: {getattr(ops.Linear, 'patch_dtype', 'None')}")
        
        # Restore cache_patched_weights during hot-reload
        if dequant_dtype == "cache_patched_weights" or (isinstance(dequant_dtype, dict) and dequant_dtype.get("cache_patched_weights")):
             setattr(ops.Linear, "cache_patched_weights", True)


    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """GGUF Soft-Offload via pointer swap.

        Because mmap_cache was already swapped to pinned RAM at load time,
        ``GGUFModelPatcher.unpatch_model()`` replaces CUDA params with the
        pinned-backed ``mmap_cache`` entries.  Old CUDA tensors are GC'd,
        VRAM is freed, and no extra host-RAM copy is needed.

        Partial-load aware: when model was partially loaded (some modules on
        CUDA, rest in lowvram hooks), the pointer swap still works because
        ``unpatch_model`` iterates the full mmap_cache and restores every
        matched parameter — including those that were moved to CUDA.
        """
        if model is None:
            return False

        inner = getattr(model, "model", None)
        is_partial = inner is not None and getattr(inner, "model_lowvram", False)
        tag = "partial" if is_partial else "full"
        print(f"[GGUFContext {config.local_rank}] GGUF Soft-Offload ({tag}): pointer swap to pinned mmap_cache...")

        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)

        model.unpatch_model(device_to=torch.device('cpu'), unpatch_weights=True)
        model.current_device = torch.device('cpu')

        self._common_offload_cleanup(model, lora_manager, config)
        print(f"[GGUFContext {config.local_rank}] GGUF Soft-Offload Complete.")

        return False
    
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """GGUF Hot-Reload: budget-aware fast path.

        Full VRAM (no budget)
        ~~~~~~~~~~~~~~~~~~~~~
        Bypasses ``model.load()`` entirely — re-enables the lowvram
        ``comfy_cast_weights`` hooks in a single tight loop over modules
        then re-attaches LoRA ``.patches`` directly.  No data movement;
        quantised weights stay on CPU and dequant on-the-fly.

        Partial VRAM (budget > 0)
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        Delegates to ``model.load(device, lowvram_model_memory=budget)``
        which lets ComfyUI decide which modules' quantised weights fit on
        CUDA (faster dequant — no CPU→GPU copy per forward) and which
        stay CPU-streamed via hooks.
        """
        if model is None:
            return

        import time
        t0 = time.perf_counter()

        # Restore GGUF ops configuration (class attrs may have been reset)
        if hasattr(model, "model_options"):
            ops = model.model_options.get("custom_operations")
            if ops:
                self._apply_ops_config(
                    ops,
                    reload_params.get("dequant_dtype"),
                    reload_params.get("patch_dtype"),
                )

        # Re-hydrate mmap_cache if somehow lost (safety net)
        if not getattr(model, "mmap_cache", None):
            unet_path = getattr(model, "unet_path", None)
            if unet_path and unet_path in state_cache:
                cached = state_cache.get(unet_path)
                if isinstance(cached, CachedState):
                    print(f"[GGUFContext] Re-hydrating mmap_cache from cache")
                    model.mmap_cache = cached.state_dict

        inner = getattr(model, "model", None)
        if inner is None:
            # Fallback to standard load if model structure is unexpected
            if hasattr(model, "load"):
                model.load(device, force_patch_weights=True)
                model.current_device = device
            print(f"[GGUFContext] Hot-load complete (fallback).")
            return

        # ── Budget calculation ────────────────────────────────────
        vram_limit = getattr(model, "vram_limit_bytes", 0)
        model_bytes = model.model_size() if hasattr(model, "model_size") else 0
        budget = (
            _compute_vram_budget(device, model_bytes, vram_limit_bytes=vram_limit)
            if vram_limit > 0 and model_bytes > 0
            else 0
        )

        if budget > 0:
            # ── PARTIAL CUDA: budget-aware load ──────────────────
            # Let ComfyUI split: budget-worth of modules get their
            # quantised weights on CUDA (fast on-device dequant),
            # the rest stay on CPU with lowvram streaming hooks.
            budget_mb = budget / (1024 ** 2)
            model_mb = model_bytes / (1024 ** 2)
            print(
                f"[GGUFContext] Hot-Reload (partial): "
                f"model {model_mb:.0f} MB, VRAM budget {budget_mb:.0f} MB..."
            )
            try:
                model.load(device, force_patch_weights=True,
                           lowvram_model_memory=budget)
                model.current_device = device
                dt = (time.perf_counter() - t0) * 1000
                print(f"[GGUFContext] Hot-Reload (partial) complete ({dt:.0f} ms).")
                return
            except Exception as e:
                print(f"[GGUFContext] Partial hot-reload failed: {e}, falling back to full lowvram...")

        # ── FULL LOWVRAM: fast path (no budget) ──────────────────
        print(f"[GGUFContext] Hot-loading to {device} (fast path)...")
        try:
            self._fast_reload(model, inner, device)
        except Exception as e:
            print(f"[GGUFContext] Fast reload failed: {e}, falling back to model.load()...")
            if hasattr(model, "load"):
                model.load(device, force_patch_weights=True)

        model.current_device = device
        dt = (time.perf_counter() - t0) * 1000
        print(f"[GGUFContext] Hot-load complete ({dt:.0f} ms).")

    def _fast_reload(self, model: Any, inner: Any, device: torch.device):
        """Fast-path internals: re-enable lowvram hooks + LoRA patches."""
        from comfy.patcher_extension import CallbacksMP

        with model.use_ejected():
            model.unpatch_hooks()

            force_cast = getattr(model, "force_cast_weights", False)
            wrapper_patches = getattr(model, "weight_wrapper_patches", {})

            # (1) Re-enable lowvram hooks on every castable module
            for n, m in inner.named_modules():
                if not hasattr(m, "comfy_cast_weights"):
                    continue
                # Initialise function lists (cleared by wipe_lowvram_weight)
                m.weight_function = []
                m.bias_function = []
                m.comfy_force_cast_weights = force_cast
                # Save + override cast flag
                if not hasattr(m, "prev_comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
                m.comfy_patched_weights = True
                # Re-attach any weight_wrapper_patches
                wk = f"{n}.weight"
                bk = f"{n}.bias"
                if wk in wrapper_patches:
                    m.weight_function.extend(wrapper_patches[wk])
                if bk in wrapper_patches:
                    m.bias_function.extend(wrapper_patches[bk])

            # (2) Re-attach LoRA .patches to quantized GGMLTensors
            patches = getattr(model, "patches", None)
            if patches:
                from raylight.expansion.comfyui_gguf.ops import move_patch_to_device
                from raylight.expansion.comfyui_gguf.dequant import is_quantized
                mmap_cache = getattr(model, "mmap_cache", {})
                patch_dev = (model.load_device
                             if getattr(model, "patch_on_device", False)
                             else model.offload_device)
                for key, patch_list in patches.items():
                    w = mmap_cache.get(key)
                    if w is not None and is_quantized(w):
                        w.patches = [(move_patch_to_device(patch_list, patch_dev), key)]

            # (3) Set model state flags
            patches = getattr(model, "patches", None)
            inner.model_lowvram = True
            inner.device = device
            inner.model_loaded_weight_memory = 0
            inner.model_offload_buffer_memory = 0
            inner.lowvram_patch_counter = len(patches) if patches else 0
            inner.current_weight_patches_uuid = getattr(model, "patches_uuid", None)

            # (4) ON_LOAD callbacks + forced hooks
            for cb in model.get_all_callbacks(CallbacksMP.ON_LOAD):
                cb(model, device, 0, False, False)
            model.apply_hooks(getattr(model, "forced_hooks", None), force_apply=True)


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
    
    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig"):
        import safetensors.torch
        import safetensors
        sd = safetensors.torch.load_file(state.unet_path, device="cpu")
        # Read safetensors header metadata (needed for model variant detection,
        # e.g. LTX 2.3 embeds {"config": {"transformer": {"cross_attention_adaln": true}}})
        try:
            with safetensors.safe_open(state.unet_path, framework="pt") as f:
                metadata = f.metadata() or {}
        except Exception:
            metadata = {}
        return sd, metadata
    
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig"):
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)
    
    @staticmethod
    def _make_pinned_cache(config: "WorkerConfig", model_path: str):
        """Select shared or private pinned cache based on worker topology.

        FSDP is excluded because each rank holds different shards.
        All other multi-GPU modes (DP, xDiT) have identical full-weight
        replicas — share one pinned buffer to save host RAM.
        """
        is_shared = config.global_world_size > 1 and not config.is_fsdp
        if is_shared:
            from raylight.distributed_modules.pinned_cache import (
                SharedPinnedParamCache, make_cache_id,
            )
            cache_id = make_cache_id(model_path)
            is_writer = config.local_rank == 0
            tag = "Writer" if is_writer else "Reader"
            print(f"[LazyTensorContext] Shared pinned cache ({tag}): "
                  f"{config.global_world_size} workers share 1 buffer.")
            return SharedPinnedParamCache(cache_id=cache_id, is_writer=is_writer)
        else:
            from raylight.distributed_modules.pinned_cache import PinnedParamCache
            return PinnedParamCache()

    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None, **kwargs):
        """Instantiate model with lazy state dict (zero-copy until load)."""
        import comfy.sd
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
        from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher
        
        print("[LazyTensorContext] Wrapping state dict with LazySafetensors...")
        lazy_sd = wrap_state_dict_lazy(sd)
        
        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        load_options["custom_operations"] = SafetensorOps
        
        model = comfy.sd.load_diffusion_model_state_dict(
            lazy_sd, model_options=load_options, metadata=metadata,
        )
        
        if model is None:
            raise RuntimeError(f"Could not load model: {state.unet_path}")
        
        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype
            
        # Store mmap cache on the patcher for offload pointer-swapping
        model.mmap_cache = sd

        # Upgrade to RaylightModelPatcher for pinned-cache-aware unpatch_model
        model.__class__ = RaylightModelPatcher

        # Attach pinned cache — shared for DP, private for single-GPU.
        # Will be lazily built on first offload (params on CUDA after first forward).
        model.pinned_param_cache = self._make_pinned_cache(config, state.unet_path)
        print("[LazyTensorContext] Pinned param cache attached (will build on first offload).")

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
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

        isolated = {k: v.clone() for k, v in sd.items()}
        
        load_options = state.model_options.copy()
        cast_dtype = load_options.pop("dtype", None)
        
        model = comfy.sd.load_diffusion_model_state_dict(isolated, model_options=load_options)
        
        if model is None:
            raise RuntimeError(f"Could not load model: {state.unet_path}")
        
        if cast_dtype and hasattr(model, "model"):
            model.model.manual_cast_dtype = cast_dtype

        model.__class__ = RaylightModelPatcher
        model.pinned_param_cache = self._make_pinned_cache(config, state.unet_path)

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
    
    @staticmethod
    def _fast_reload_full(model, inner, device):
        """Bypass model.load() after reload_to_cuda — params already on CUDA.

        model.load() spends most of its time in ``_load_list()`` (iterates
        every module, computes sizes, sorts) then calls per-module ``.to()``
        + ``torch.cuda.synchronize()`` for each.  All of that is wasted
        when the pinned cache already DMA'd everything to CUDA.

        This fast path re-applies LoRA patches and hooks directly, mirroring
        what GGUFContext._fast_reload does for GGUF models.
        """
        from comfy.patcher_extension import CallbacksMP

        with model.use_ejected():
            model.unpatch_hooks()

            patches = getattr(model, "patches", {})
            wrapper_patches = getattr(model, "weight_wrapper_patches", {})
            force_cast = getattr(model, "force_cast_weights", False)

            # (1) Re-apply LoRA / weight patches to CUDA params.
            if patches:
                for key in patches:
                    model.patch_weight_to_device(key, device_to=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # (2) Mark all modules as patched (prevents model.load from
            #     re-doing work if something later calls it).
            for n, m in inner.named_modules():
                if hasattr(m, "comfy_cast_weights"):
                    if not hasattr(m, "prev_comfy_cast_weights"):
                        m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_force_cast_weights = force_cast
                    # Reset + re-attach wrapper patches (reset avoids
                    # duplicates on repeated reloads).
                    wk = f"{n}.weight"
                    bk = f"{n}.bias"
                    m.weight_function = list(wrapper_patches.get(wk, []))
                    m.bias_function = list(wrapper_patches.get(bk, []))
                m.comfy_patched_weights = True

            # (3) Set model state flags
            inner.model_lowvram = False
            inner.device = device
            inner.model_loaded_weight_memory = sum(
                p.nbytes for p in inner.parameters()
            )
            inner.model_offload_buffer_memory = 0
            inner.lowvram_patch_counter = 0
            inner.current_weight_patches_uuid = getattr(model, "patches_uuid", None)

            # (4) ON_LOAD callbacks + forced hooks
            for cb in model.get_all_callbacks(CallbacksMP.ON_LOAD):
                cb(model, device, 0, False, True)
            model.apply_hooks(getattr(model, "forced_hooks", None), force_apply=True)

        model.inject_model()

    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig") -> bool:
        """Fast pinned-RAM offload: cache CUDA params, free VRAM via storage resize.

        Falls back to legacy mmap pointer-swap if no pinned cache is available.
        """
        from raylight.expansion.comfyui_lazytensors.lazy_tensor import swap_model_to_mmap

        if model is None:
            return False

        pinned_cache = getattr(model, "pinned_param_cache", None)
        diffusion_model = getattr(model, "model", None)

        # --- Pinned-cache fast path ---
        if pinned_cache is not None and diffusion_model is not None:
            print(f"[LazyTensorContext {config.local_rank}] Pinned-RAM Offload: CUDA → pinned CPU...")

            # Capture partial-load flag BEFORE unpatch_model clears it.
            is_partial = getattr(diffusion_model, "model_lowvram", False)

            # 1. Clear LoRA GPU refs
            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)

            # 2. Unpatch LoRA weight patches (restores base weights)
            if hasattr(model, "unpatch_model"):
                model.unpatch_model(device_to=None, unpatch_weights=True)

            # 3. Offload via pinned cache (lazy build + storage resize)
            #    If model was partially loaded (lowvram), only snapshot the
            #    CUDA-resident params — the CPU ones never left pinned RAM.
            try:
                if is_partial:
                    pinned_cache.offload_cuda_only(diffusion_model)
                else:
                    pinned_cache.offload_to_cpu(diffusion_model)
                # Drop mmap reference — pinned cache is now the sole reload source.
                # Frees ~model_size of OS page cache RAM.
                if hasattr(model, 'mmap_cache'):
                    del model.mmap_cache
            except Exception as e:
                print(f"[LazyTensorContext {config.local_rank}] Pinned offload error: {e}")

            model.current_device = torch.device('cpu')

            # 4. Clean up cached PE tensors and other non-param CUDA attributes
            for target in (diffusion_model, getattr(diffusion_model, 'diffusion_model', None)):
                if target is None:
                    continue
                for attr in ('_cached_pe', 'cached_pe', 'pe_cache', '_pe'):
                    if hasattr(target, attr):
                        delattr(target, attr)

            if lora_manager:
                lora_manager.clear_tracking()

            from raylight.utils.common import cleanup_memory
            cleanup_memory()

            mem = torch.cuda.memory_allocated(config.device) / 1e9
            print(f"[LazyTensorContext {config.local_rank}] Post-Offload VRAM: {mem:.2f} GB")
            return False

        # --- Legacy fallback: mmap pointer swap ---
        print(f"[LazyTensorContext {config.local_rank}] Soft-Offload: Performing Pointer Swap (legacy)...")

        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)

        if hasattr(model, "unpatch_model"):
            model.unpatch_model(device_to=None, unpatch_weights=True)

        mmap_cache = getattr(model, "mmap_cache", None)
        if diffusion_model is not None and mmap_cache:
            swap_model_to_mmap(diffusion_model, mmap_cache)

        model.current_device = torch.device('cpu')
        self._common_offload_cleanup(model, lora_manager, config)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[LazyTensorContext {config.local_rank}] Soft-Offload Complete (legacy).")
        return False

    
    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """Hot reload: restore from pinned RAM (fast) or mmap (legacy).

        Budget-aware: if the model does not fit entirely in VRAM the
        pinned-cache path keeps CPU-resident params in pinned RAM and
        passes ``lowvram_model_memory`` to ``model.load()`` so ComfyUI
        splits modules into CUDA-resident vs streamed-per-layer.
        """
        pinned_cache = getattr(model, "pinned_param_cache", None)
        diffusion_model = getattr(model, "model", None)

        # --- Eagerly build cache if not built yet (first run) ---
        # When the model was just loaded (params on CPU, cache empty),
        # build now so the budget-aware path can use it immediately.
        if (
            pinned_cache is not None
            and not pinned_cache.built
            and diffusion_model is not None
        ):
            print("[LazyTensorContext] Building pinned cache from CPU params...")
            try:
                pinned_cache.build(diffusion_model)
                # Params now point at shm views.  Drop the original CPU
                # state-dict references so the ~model-size heap copy is GC'd.
                if hasattr(model, 'mmap_cache'):
                    del model.mmap_cache
                if state_cache is not None:
                    unet_path = getattr(model, 'unet_path', None)
                    if unet_path:
                        cached = state_cache.get(unet_path)
                        if cached is not None and hasattr(cached, 'state_dict'):
                            cached.state_dict = None
            except Exception as e:
                print(f"[LazyTensorContext] Eager cache build failed: {e}")

        # --- Pinned-cache fast path ---
        if pinned_cache is not None and pinned_cache.built and diffusion_model is not None:
            vram_limit = getattr(model, "vram_limit_bytes", 0)
            model_bytes = model.model_size() if hasattr(model, "model_size") else 0
            budget = _compute_vram_budget(device, model_bytes, vram_limit_bytes=vram_limit) if model_bytes > 0 else 0

            if budget == 0:
                # ── FULL CUDA: model fits in VRAM ──
                import time as _time
                t0 = _time.perf_counter()
                print(f"[LazyTensorContext] Pinned-RAM Hot-Reload (full): pinned CPU → CUDA...")
                try:
                    pinned_cache.reload_to_cuda(diffusion_model)
                    diffusion_model.device = device

                    # Fast path: bypass model.load() entirely.
                    # model.load() runs _load_list() (pure-Python iteration
                    # over every module + sort) then per-module .to(device) +
                    # synchronize — all wasted when params are already on CUDA.
                    # Instead, directly re-apply LoRA patches + hooks.
                    self._fast_reload_full(model, diffusion_model, device)

                    model.current_device = device
                    dt = (_time.perf_counter() - t0) * 1000
                    print(f"[LazyTensorContext] Pinned-RAM Hot-Reload (full) complete ({dt:.0f} ms).")
                    return
                except Exception as e:
                    print(f"[LazyTensorContext] Pinned-RAM full reload failed: {e}, falling back to model.load()...")
                    try:
                        if hasattr(model, 'load'):
                            model.load(device)
                        model.current_device = device
                        return
                    except Exception as e2:
                        print(f"[LazyTensorContext] model.load() fallback also failed: {e2}, trying mmap...")
            else:
                # ── PARTIAL CUDA: model exceeds VRAM ──
                # Do NOT bulk-copy to CUDA.  Let model.load() decide which
                # modules fit: those that do get DMA'd from pinned RAM,
                # the rest stay pinned and are streamed per-layer via
                # comfy_cast_weights hooks (also DMA from pinned → fast).
                budget_mb = budget / (1024 ** 2)
                model_mb = model_bytes / (1024 ** 2)
                print(
                    f"[LazyTensorContext] Pinned-RAM Hot-Reload (partial): "
                    f"model {model_mb:.0f} MB, VRAM budget {budget_mb:.0f} MB — "
                    f"streaming overflow from pinned RAM..."
                )
                try:
                    # Ensure params are on CPU in pinned RAM (they should
                    # already be after offload, but guard against edge cases).
                    pinned_cache.reload_to_cpu(diffusion_model)
                    if diffusion_model is not None:
                        model.model.device = torch.device("cpu")
                    # Let ComfyUI split: budget-worth → CUDA, rest → LowVramPatch hooks.
                    # Skip auto-restore in RaylightModelPatcher.load() — we already
                    # placed params on CPU pinned RAM; super().load() will DMA only
                    # the budget-worth of modules to CUDA (efficient).
                    if hasattr(model, 'load'):
                        model._skip_pinned_auto_restore = True
                        try:
                            model.load(device, lowvram_model_memory=budget)
                        finally:
                            model._skip_pinned_auto_restore = False
                    model.current_device = device
                    # Some params are now on CUDA — update cache state.
                    pinned_cache._on_cuda = True
                    print("[LazyTensorContext] Pinned-RAM Hot-Reload (partial) complete.")
                    return
                except Exception as e:
                    print(f"[LazyTensorContext] Pinned-RAM partial reload failed: {e}, falling back to mmap...")

        # --- Legacy fallback: mmap streaming ---
        print(f"[LazyTensorContext] Hot reload to {device} (mmap)...")

        unet_path = getattr(model, "unet_path", None)
        if unet_path and unet_path in state_cache:
            cached = state_cache.get(unet_path)
            if isinstance(cached, CachedState) and cached.checksum:
                if isinstance(getattr(model, 'mmap_cache', None), dict):
                    verify_model_checksum(model.mmap_cache, cached.checksum, cached.metadata, context_tag="LazyTensorContext")

        if model is not None and hasattr(model, 'load'):
            model.load(device)
            model.current_device = device


class FSDPContext(ModelContext):
    """Context for FSDP model loading with optional mmap for safetensors."""
    
    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig") -> Dict:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)
    
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig") -> Dict:
        import comfy.utils
        return comfy.utils.load_torch_file(state.unet_path, return_metadata=True)

    def load(self, state: ModelState, config: "WorkerConfig", state_cache: Any) -> Any:
        """Override load to implement prefetch optimization for Safetensors."""
        is_safetensors = state.unet_path.lower().endswith(".safetensors")
        
        # Check cache logic duplicated from base (allow Base load to handle cache hits)
        if self.cache_in_ram and state_cache.get(state.cache_key) is not None:
             return super().load(state, config, state_cache)
             
        if is_safetensors and not self.use_mmap: 
             print(f"[FSDPContext] Prefetch Optimization Active for {os.path.basename(state.unet_path)}")
             
             # 1. Start Background Load
             executor = ThreadPoolExecutor(max_workers=1)
             future = executor.submit(prefetch_state_dict, state.unet_path)
             
             # 2. Fast Header Read (Dummy SD)
             from safetensors import safe_open
             dummy_sd = {}
             metadata = None
             try:
                 with safe_open(state.unet_path, framework="pt") as f:
                     metadata = f.metadata()
                     for k in f.keys():
                         t_slice = f.get_slice(k)
                         shape = t_slice.get_shape()
                         dummy_sd[k] = torch.empty(shape, dtype=torch.float16, device="meta")
             except Exception as e:
                 print(f"[FSDPContext] Header read failed: {e}. Fallback to sync load.")
                 executor.shutdown(wait=False)
                 return super().load(state, config, state_cache)

             # 3. Instantiate with dummy_sd (Wrapping FSDP on CPU)
             try:
                 model = self.instantiate_model(dummy_sd, state, config, metadata=metadata, load_weights=False)
                 
                 # 4. Finalize
                 print("[FSDPContext] FSDP Wrapping done. Waiting for weights...")
                 real_sd = future.result()
                 
                 print("[FSDPContext] Weights loaded. Loading into FSDP model...")
                 
                 # Process keys to match model structure (strip prefixes)
                 from comfy import model_detection
                 diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(real_sd)
                 import comfy.utils
                 temp_sd = comfy.utils.state_dict_prefix_replace(
                    real_sd, {diffusion_model_prefix: ""}, filter_keys=True
                 )
                 if len(temp_sd) > 0:
                     real_sd = temp_sd
                 
                 model.load_model_weights(real_sd, "")
                 
                 # Shutdown executor
                 executor.shutdown(wait=False)
                 
                 # Post-load setup
                 if model is not None:
                     model.unet_path = state.unet_path
                     model.load_config = state
                     
                 return model

             except Exception as e:
                 print(f"[FSDPContext] Prefetch/Wrap failed: {e}. Fallback to sync load.")
                 executor.shutdown(wait=False)
                 return super().load(state, config, state_cache)
        
        return super().load(state, config, state_cache)
    
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None, load_weights: bool = True, **kwargs):
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
            model_options=model_options,
            metadata=metadata,
            load_weights=load_weights
        )
        
        if model is None:
            raise RuntimeError(f"Could not load FSDP model: {state.unet_path}")
        
        # FSDP special: attach state dict to model for retrieval? 
        # Or return it? For now, we attach it to the model patcher if possible.
        # But wait, original code set `worker.state_dict`.
        # To remain stateless, we should return it or attach to model.
        # Let's attach to model.
        # OPTIMIZATION: Do NOT attach state dict to model to prevent RAM leak.
        # model.fsdp_state_dict = state_dict
        
        # Mark as baked so RayWorker knows to reload on LoRA change
        model.is_fsdp_baked = True

        # Attach an empty shard cache now; it will be built lazily on the first
        # offload_to_cpu() call, after the first forward has triggered set_model_state_dict
        # and all DTensor shards are guaranteed to be resident on CUDA.
        # (Building here would capture 0 shards for deferred-load models like Lightricks.)
        fsdp_cpu_offload = config.parallel_dict.get("fsdp_cpu_offload", False)
        if not fsdp_cpu_offload:
            from raylight.distributed_modules.pinned_cache import FSDPShardPinnedCache
            model.fsdp_shard_cache = FSDPShardPinnedCache()
            print(f"[FSDPContext {config.local_rank}] Shard cache attached (will build on first offload).")

        return model

    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig"):
        """
        FSDP soft offload: move per-rank local shards from CUDA to pinned CPU RAM, freeing
        VRAM without destroying the FSDP-wrapped model structure.  On the next load the shards
        are restored with a fast H2D memcpy — no collective communication, no disk I/O.

        Falls back to hard offload (model destroyed) if no shard cache is available.
        """
        shard_cache = getattr(model, "fsdp_shard_cache", None) if model is not None else None

        if shard_cache is not None:
            # offload_to_cpu() handles lazy build internally — do NOT gate on shard_cache.built
            print(f"[FSDPContext {config.local_rank}] FSDP Soft-Offload: shards → pinned CPU RAM...")

            # Clear weight_function handles so LoRA patches don't hold stale CUDA refs
            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)

            # Move every FSDP DTensor local shard CUDA → pinned RAM
            try:
                shard_cache.offload_to_cpu(model.model.diffusion_model)
            except Exception as e:
                print(f"[FSDPContext {config.local_rank}] Shard cache offload error: {e}")

            # Update device marker so reload_if_needed knows to call hot_load
            if model.model is not None:
                model.model.device = torch.device("cpu")

            if lora_manager:
                lora_manager.clear_tracking()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mem = torch.cuda.memory_allocated(config.device) / 1e9
            print(f"[FSDPContext {config.local_rank}] Post-Soft-Offload VRAM: {mem:.2f} GB")

            return False  # soft offload — model object preserved for hot_load

        else:
            # No shard cache yet (first run was skipped or cache build failed)
            # Fall back to the original hard-offload path.
            print(f"[FSDPContext {config.local_rank}] FSDP Hard-Offload: no shard cache, releasing all resources...")

            if lora_manager:
                lora_manager.clear_gpu_refs(model, config)

            if model is not None and hasattr(model, "release_memory"):
                model.release_memory()

            self._common_offload_cleanup(model, lora_manager, config)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True

    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        """
        Restore FSDP local shards from pinned CPU RAM back to CUDA.
        Called by reload_if_needed when model.current_device != target_device.
        No reshard, no collective communication — pure async H2D memcpy.
        """
        shard_cache = getattr(model, "fsdp_shard_cache", None)
        if shard_cache is not None and shard_cache.built:
            print("[FSDPContext] FSDP Hot-Reload: pinned CPU RAM → CUDA...")
            try:
                shard_cache.reload_to_cuda(model.model.diffusion_model)
                # Restore device marker
                if model.model is not None:
                    model.model.device = device
                model.current_device = device
                print("[FSDPContext] Hot-Reload complete.")
                return
            except Exception as e:
                print(f"[FSDPContext] Hot-Reload failed: {e}, falling back to full reload...")

        # Fallback: full model.load() from whatever state dict is available
        print("[FSDPContext] Hot-Reload: no shard cache or pinned reload failed — full reload.")
        if model is not None and hasattr(model, "load"):
            model.load(device)
            model.current_device = device


class VAEContext(LazyTensorContext):
    """Context for VAE model loading with streaming mmap support."""
    
    def prepare_state_dict(self, sd: Dict[str, torch.Tensor], config: "WorkerConfig") -> Dict[str, torch.Tensor]:
        """VAE: No prefix stripping needed (unlike UNET)."""
        return sd
        
    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None, **kwargs):
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
            
            # Structured CachedState
            if isinstance(cached, CachedState) and cached.checksum:
                if isinstance(model.mmap_cache, dict):
                    verify_model_checksum(model.mmap_cache, cached.checksum, cached.metadata, context_tag="VAEContext")

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

    def offload(self, model: Any, lora_manager: Optional["LoraManager"], worker_mmap_cache: Any, config: "WorkerConfig") -> bool:
        """Zero-copy soft-offload for VAE."""
        
        if model is None:
            return False
        
        
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
                
        return False

class BNBContext(ModelContext):
    """Context for BitsAndBytes (4-bit) model loading."""
    
    def load_state_dict_standard(self, state: ModelState, config: "WorkerConfig") -> Dict:
        # BNB loaders currently handle loading from disk internally so we return a dummy dict
        return {"__bnb_internal__": True}

    def load_state_dict_mmap(self, state: ModelState, config: "WorkerConfig") -> Dict:
         return self.load_state_dict_standard(state, config)

    def instantiate_model(self, sd: Dict, state: ModelState, config: "WorkerConfig", metadata: Any = None, **kwargs):
        unet_path = state.unet_path
        
        if config.is_fsdp:
            # Inline patches for BNB FSDP (ported from RayWorker)
            import comfy.model_patcher as model_patcher
            import comfy.model_management as model_management
            from raylight.comfy_dist.model_management import cleanup_models_gc
            from raylight.comfy_dist.model_patcher import LowVramPatch
            
            model_patcher.LowVramPatch = LowVramPatch
            model_management.cleanup_models_gc = cleanup_models_gc

            from raylight.comfy_dist.sd import fsdp_bnb_load_diffusion_model
            
            model, state_dict = fsdp_bnb_load_diffusion_model(
                unet_path, 
                config.local_rank, 
                config.device_mesh, 
                config.parallel_dict.get("fsdp_cpu_offload", False)
            )
            model.fsdp_state_dict = state_dict
            return model
        else:
            from raylight.comfy_dist.sd import bnb_load_diffusion_model
            return bnb_load_diffusion_model(unet_path)

    def hot_load(self, model: Any, device: torch.device, reload_params: Dict[str, Any], state_cache: Any):
        # BNB models are difficult to hot-load due to 4-bit/CPU offload config
        pass


def get_context(path: str, config: "WorkerConfig", model_type: str = "unet", model_options: Optional[Dict] = None) -> ModelContext:
    """Factory function to select appropriate context based on config and file type.
    
    Args:
        path: Path to model file
        config: Worker configuration
        model_type: "unet" or "vae"
        model_options: Optional model load options (needed for BNB detection)
        
    Returns:
        Appropriate ModelContext subclass instance
    """
    use_mmap = config.parallel_dict.get("use_mmap", True)
    
    if model_type == "vae":
        return VAEContext(use_mmap=use_mmap)
    
    if model_options and ("bnb_4bit" in model_options or "load_in_4bit" in model_options):
        return BNBContext(use_mmap=False) # BNB manages its own memory/loading
        
    if getattr(config, "is_fsdp", False):
        if path.lower().endswith(".gguf"):
            raise ValueError(
                "[Raylight] FSDP is not supported for GGUF models. "
                "GGUF quantization is incompatible with FSDP sharding. "
                "Please use a standard Safetensors model or disable FSDP/Context Parallel to use GGUF."
            )
        return FSDPContext(use_mmap=use_mmap, cache_in_ram=False)
    if path.lower().endswith(".gguf"):
        return GGUFContext(use_mmap=use_mmap)
    return LazyTensorContext(use_mmap=use_mmap)
