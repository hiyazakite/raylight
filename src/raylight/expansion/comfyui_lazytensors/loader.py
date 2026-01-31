import torch
from typing import Dict, Optional

class SafetensorMmapWrapper:
    """Wraps mmap'd safetensor state dict for streaming GPU transfer.
    
    Enables per-tensor GPU loading to avoid RAM spikes from full clone.
    """
    
    def __init__(self, mmap_sd: Dict[str, torch.Tensor]):
        self._mmap = mmap_sd
    
    def stream_to_model(self, model, device: torch.device) -> int:
        """Transfer weights per-parameter to avoid RAM spike.
        
        Returns number of parameters + buffers transferred.
        """
        transferred = 0
        unmatched = []
        
        # Build combined map of parameters AND buffers
        # Many VAE models store weights as buffers (e.g., scale_shift_table, timestep_scale_multiplier)
        param_map = {name: param for name, param in model.named_parameters()}
        buffer_map = {name: buf for name, buf in model.named_buffers()}
        combined_map = {**param_map, **buffer_map}
        
        print(f"[SafetensorMmapWrapper] Mmap keys: {len(self._mmap)}, Model params: {len(param_map)}, Model buffers: {len(buffer_map)}")
        
        # Debug: Print first 5 keys from each
        mmap_keys = list(self._mmap.keys())[:5]
        model_param_keys = list(param_map.keys())[:5]
        print(f"[SafetensorMmapWrapper] Sample mmap keys: {mmap_keys}")
        print(f"[SafetensorMmapWrapper] Sample model param keys: {model_param_keys}")
        
        for name, mmap_tensor in self._mmap.items():
            target_name = self._resolve_name(name, combined_map)
            if target_name and target_name in combined_map:
                # Use as_subclass to stream directly from mmap to GPU (Zero-Copy)
                new_data = mmap_tensor.as_subclass(torch.Tensor).to(device)
                
                # For parameters, assign to .data
                # For buffers, we need to use register_buffer or direct assignment
                if target_name in param_map:
                    param_map[target_name].data = new_data
                else:
                    # Buffer - find the parent module and reassign
                    parts = target_name.rsplit('.', 1)
                    if len(parts) == 2:
                        parent_name, attr_name = parts
                        parent = model
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part, None)
                            if parent is None:
                                break
                        if parent is not None:
                            parent.register_buffer(attr_name, new_data, persistent=True)
                    else:
                        model.register_buffer(target_name, new_data, persistent=True)
                        
                transferred += 1
            else:
                unmatched.append(name)
        
        if unmatched:
            print(f"[SafetensorMmapWrapper] WARNING: {len(unmatched)} unmatched keys! First 10: {unmatched[:10]}")
        
        print(f"[SafetensorMmapWrapper] Transferred {transferred}/{len(self._mmap)} tensors to {device}")
        return transferred
    
    def _resolve_name(self, name: str, param_map: Dict) -> Optional[str]:
        """Resolve mmap key to model parameter name."""
        # Direct match
        if name in param_map:
            return name
        # Common prefixes
        prefixes = ["diffusion_model.", "model.diffusion_model.", "first_stage_model.", ""]
        for prefix in prefixes:
            candidate = f"{prefix}{name}" if prefix else name
            if candidate in param_map:
                return candidate
            # Strip prefix if present
            if name.startswith(prefix) and name[len(prefix):] in param_map:
                return name[len(prefix):]
        return None
