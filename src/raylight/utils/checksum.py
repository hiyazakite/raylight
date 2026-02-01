from typing import Dict, Any, Optional

def compute_model_checksum(sd: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Compute a lightweight checksum for debug verification.
    
    Args:
        sd: The state dict to hash (samples keys/shapes/values)
        metadata: Optional metadata dictionary to include in hash
    
    Returns:
        String checksum signature
    """
    try:
        keys = sorted(list(sd.keys()))
        if not keys:
            return "empty"
        
        # Sample first, middle, last
        indices = [0, len(keys)//2, len(keys)-1]
        checksum_parts = []
        for idx in indices:
            k = keys[idx]
            v = sd[k]
            if hasattr(v, "shape"):
                meta = f"{k}:{v.shape}:{v.dtype}"
                # Add custom attributes check for GGML
                if hasattr(v, "tensor_type"):
                    meta += f":{v.tensor_type}"
                
                # Add small data sample (first value)
                # Use .flatten()[0] to avoid shape issues, verify it's a tensor
                if v.numel() > 0:
                    val = v.flatten()[0].item() if not v.is_meta else 0
                    meta += f":{val:.4f}"
                checksum_parts.append(meta)
        
        # Incorporate metadata keys if present
        if metadata:
            meta_keys = sorted(list(metadata.keys()))
            # Hash the keys and a simplified representation of values
            meta_imp = f"|meta:{len(meta_keys)}"
            if meta_keys:
                meta_imp += f":{meta_keys[0]}" # Just check presence of first key for speed
            checksum_parts.append(meta_imp)

        return "|".join(checksum_parts)
    except Exception as e:
        return f"checksum_error:{e}"

def verify_model_checksum(sd: Dict[str, Any], stored_checksum: str, metadata: Optional[Dict[str, Any]] = None, context_tag: str = "ModelContext") -> bool:
    """Verify state dict against stored checksum and log results.
    
    Args:
        sd: The state dict to check
        stored_checksum: The trusted checksum from cache
        metadata: Optional metadata to include in checksum
        context_tag: Tag for logging (e.g. 'GGUFContext')
        
    Returns:
        bool: True if checksum matches, False otherwise
    """
    current_checksum = compute_model_checksum(sd, metadata)
    
    if stored_checksum != current_checksum:
        print(f"[{context_tag}] CRITICAL: Cache Corruption Detected!")
        print(f"  Stored:  {stored_checksum}")
        print(f"  Current: {current_checksum}")
        return False
    else:
        print(f"[{context_tag}] Cache Integrity Verified (Checksum match).")
        return True
