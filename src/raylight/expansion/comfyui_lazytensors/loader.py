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
        
        # Build separate maps — avoid merging into a third dict
        param_map = dict(model.named_parameters())
        buffer_map = dict(model.named_buffers())
        
        print(f"[SafetensorMmapWrapper] Mmap keys: {len(self._mmap)}, Model params: {len(param_map)}, Model buffers: {len(buffer_map)}")
        
        # Debug: Print first 5 keys from each
        mmap_keys = list(self._mmap.keys())[:5]
        model_param_keys = list(param_map.keys())[:5]
        print(f"[SafetensorMmapWrapper] Sample mmap keys: {mmap_keys}")
        print(f"[SafetensorMmapWrapper] Sample model param keys: {model_param_keys}")
        
        for name, mmap_tensor in self._mmap.items():
            # Resolve against params first, then buffers (avoids combined_map copy)
            target_name = self._resolve_name(name, param_map)
            is_param = target_name is not None
            if not is_param:
                target_name = self._resolve_name(name, buffer_map)
            
            if target_name is None:
                unmatched.append(name)
                continue
                
            # Stream directly from mmap to GPU
            new_data = mmap_tensor.as_subclass(torch.Tensor).to(device, non_blocking=True)
            
            if is_param:
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
        
        # Single sync after all async transfers
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
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
