"""Lazy tensor wrappers for zero-copy safetensor loading.

Similar to GGMLTensor - wraps mmap'd tensor data and defers materialization
until GPU access is needed. This avoids RAM spikes during model loading.

Supports pointer-swap offload/reload:
- After .to(device), tensor keeps reference to mmap source
- swap_to_mmap() restores pointer to mmap (zero-copy offload)
- Enables fast hot-reload without re-reading from disk
"""
import torch
from typing import Optional, Dict


class MaterializedTensor(torch.Tensor):
    """A regular tensor that remembers its mmap source for pointer-swap offload."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func.__name__ == "_has_compatible_shallow_copy_type":
            return True

        if not all(issubclass(t, (torch.Tensor, MaterializedTensor, LazySafetensor)) for t in types):
            return NotImplemented

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def __new__(cls, data: torch.Tensor, mmap_ref: torch.Tensor) -> 'MaterializedTensor':
        instance = torch.Tensor._make_subclass(cls, data)
        return instance

    def __init__(self, data: torch.Tensor, mmap_ref: torch.Tensor):
        super().__init__()
        # Keep reference to original mmap for pointer-swap
        object.__setattr__(self, '_mmap_ref', mmap_ref)

    def to(self, *args, **kwargs) -> 'MaterializedTensor':
        """Preserve MaterializedTensor type and mmap ref during transfer."""
        device = None
        if args and (isinstance(args[0], (str, torch.device))):
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        if str(device) == "meta":
            meta = torch.empty(self.shape, dtype=self.dtype, device="meta")
            return MaterializedTensor(meta, self.mmap_ref)

        # Standard transfer
        new_data = super().to(*args, **kwargs)
        # Re-wrap to ensure mmap_ref is preserved and type is MaterializedTensor
        return MaterializedTensor(new_data, self.mmap_ref)

    def swap_to_mmap(self) -> 'LazySafetensor':
        """Swap back to mmap reference (zero-copy offload).

        Returns a lazy tensor pointing to the original mmap.
        The GPU memory for this tensor can then be freed.
        """
        mmap_ref = object.__getattribute__(self, '_mmap_ref')
        return LazySafetensor(mmap_ref)

    @property
    def mmap_ref(self) -> torch.Tensor:
        """Access the original mmap reference."""
        return object.__getattribute__(self, '_mmap_ref')

    def detach(self) -> 'MaterializedTensor':
        """Detach while preserving MaterializedTensor type and mmap ref."""
        # Detach the underlying tensor data
        detached_data = super().detach()
        # Create new MaterializedTensor checking out the same mmap ref
        return MaterializedTensor(detached_data, self.mmap_ref)


class LazySafetensor(torch.Tensor):
    """
    Lazy wrapper for mmap'd safetensor data.

    This tensor subclass wraps mmap'd safetensors and defers
    the actual clone/transfer until the tensor is moved to GPU.

    Key behaviors:
    - Contains actual mmap safetensor data (not empty placeholder)
    - .to(device) returns MaterializedTensor that keeps mmap ref
    - Enables pointer-swap offload via MaterializedTensor.swap_to_mmap()
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func.__name__ == "_has_compatible_shallow_copy_type":
            return True

        if not all(issubclass(t, (torch.Tensor, MaterializedTensor, LazySafetensor)) for t in types):
            return NotImplemented

        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            return ret

    def __new__(cls, mmap_tensor: torch.Tensor, patches=None) -> 'LazySafetensor':
        # Use the mmap tensor data directly (like GGMLTensor)
        # This avoids the size mismatch issue with load_state_dict
        instance = torch.Tensor._make_subclass(cls, mmap_tensor)
        return instance

    def __init__(self, mmap_tensor: torch.Tensor, patches=None):
        super().__init__()
        # Mark that this points to mmap (the data IS the mmap)
        object.__setattr__(self, '_is_mmap', True)
        # Store patches for lazy application
        object.__setattr__(self, 'patches', patches if patches is not None else [])

    def __copy__(self):
        new = LazySafetensor(self, patches=self.patches.copy())
        return new

    def __deepcopy__(self, memo):
        new = LazySafetensor(self, patches=[p for p in self.patches])
        return new

    @property
    def is_mmap(self) -> bool:
        return object.__getattribute__(self, '_is_mmap')

    def to(self, *args, **kwargs) -> MaterializedTensor:
        """Materialize and transfer to target device.

        Returns MaterializedTensor that keeps mmap ref for pointer-swap.
        """
        # Optimization: Handle 'meta' device without reading data
        device = None
        if args and (isinstance(args[0], (str, torch.device))):
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        if str(device) == "meta":
            # Create empty meta tensor (Zero-Cost). 
            # Return SAME subclass to allow assignment to parameter.data
            meta = torch.empty(self.shape, dtype=self.dtype, device="meta")
            try:
                p = self.patches
            except AttributeError:
                p = []
            return LazySafetensor(meta, patches=p)
            
        # Optimization: Return self if already on target device
        # Use thorough device comparison to ensure ZERO-COPY on CPU->CPU
        if device is not None:
             target_device = torch.device(device)
             current_device = torch.device(self.device)
             if target_device.type == current_device.type:
                 # If explicit index is provided, check it (unless it's CPU where index 0 is implied)
                 if target_device.index == current_device.index or (target_device.type == 'cpu'):
                     return self
            
        # Use as_subclass to avoid cloning data on CPU (Zero-Copy)
        materialized_data = self.as_subclass(torch.Tensor).to(*args, **kwargs)
        try:
            p = self.patches
        except AttributeError:
            p = []
            
        # Wrap in MaterializedTensor to preserve mmap ref
        # Pass self as the mmap source
        return MaterializedTensor(materialized_data, self)

    def clone(self, *args, **kwargs) -> torch.Tensor:
        # Clone returns a regular tensor (breaks the lazy wrapper)
        return torch.Tensor.clone(self, *args, **kwargs)

    def detach(self) -> 'LazySafetensor':
        new = torch.Tensor._make_subclass(LazySafetensor, torch.Tensor.detach(self))
        object.__setattr__(new, '_is_mmap', True)
        return new

    def __repr__(self) -> str:
        return f"LazySafetensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


def wrap_state_dict_lazy(sd: dict) -> dict:
    """Wrap all tensors in a state dict with LazySafetensor."""
    wrapped = {}
    for key, tensor in sd.items():
        if isinstance(tensor, torch.Tensor) and not isinstance(tensor, LazySafetensor):
            wrapped[key] = LazySafetensor(tensor)
        else:
            wrapped[key] = tensor
    return wrapped


def swap_model_to_mmap(model, mmap_cache: Optional[Dict[str, torch.Tensor]] = None) -> None:
    """
    Swap all parameters in a model back to mmap refs (offload).
    Args:
        model: The model to offload
        mmap_cache: Optional cache of mmap tensors. If provided, uses robust name matching
                    to swap parameters even if they lost MaterializedTensor identity.
    """
    swapped_count = 0
    # Method A: Robust name matching (Preferred)
    if mmap_cache:
        param_map = dict(model.named_parameters())
        # Iterate cache keys to find corresponding model params
        for key, mmap_tensor in mmap_cache.items():
            # Try matching with common prefixes
            matched_param = None
            for prefix in ['', 'diffusion_model.', 'model.diffusion_model.']:
                target_key = f"{prefix}{key}" if prefix else key
                if target_key in param_map:
                    matched_param = param_map[target_key]
                    break
            if matched_param is not None:
                # Found it! Swap to LazySafetensor pointing to mmap
                # This works regardless of whether the current param is on GPU, CPU, or wrapped
                lazy = LazySafetensor(mmap_tensor)
                matched_param.data = lazy
                swapped_count += 1   
        print(f"[LazySafetensor] Offloaded {swapped_count} params via name matching.")
        return

    # Method B: Type-based discovery (Fallback)
    # Only works if parameters preserved MaterializedTensor type
    for name, param in model.named_parameters():
        if isinstance(param.data, MaterializedTensor):
            lazy = param.data.swap_to_mmap()
            param.data = lazy.mmap_ref  # Point to CPU mmap
            swapped_count += 1            
    print(f"[LazySafetensor] Offloaded {swapped_count} params via type checking.")


def restore_model_mmap_refs(model, mmap_cache: Dict[str, torch.Tensor]) -> None:
    """
    Restore mmap references to model parameters from cache.  
    Used for hot-reload after offload.
    """
    param_map = dict(model.named_parameters())
    for key, mmap_tensor in mmap_cache.items():
        # Handle common key prefixes
        for prefix in ['', 'diffusion_model.', 'model.diffusion_model.']:
            target_key = f"{prefix}{key}" if prefix else key
            if target_key in param_map:
                param_map[target_key].data = mmap_tensor
                break

