"""DiTFastAttn: Temporal Attention Caching for Diffusion Transformers.

This module implements cross-timestep attention caching to exploit the
temporal redundancy in diffusion models. By reusing attention outputs
from previous denoising steps, we can skip expensive QKV and attention
computations, significantly reducing inference latency.
"""

import logging
import torch

logger = logging.getLogger(__name__)


class DenoisingStepTracker:
    """Tracks the current global denoising step across the sampling loop."""
    
    def __init__(self):
        self.current_step = -1
        self.total_steps = -1
        self.last_timestep = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def update(self, timestep):
        """Update step counter based on changing timestep values."""
        # Handle tensor or float timesteps
        ts_val = timestep.item() if torch.is_tensor(timestep) else float(timestep)
        
        if self.last_timestep != ts_val:
            self.current_step += 1
            self.last_timestep = ts_val

    def reset(self):
        self.current_step = -1
        self.last_timestep = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def should_cache(self, skip_interval: int = 1, start_pct: float = 0.1, end_pct: float = 0.9):
        """Determine if current step should use cached attention.
        
        Args:
            skip_interval: 1 means cache every alternating step (compute 1, cache 1)
                           2 means compute 1, cache 2, etc.
            start_pct: Don't cache in the first X% of steps (high change rate)
            end_pct: Don't cache in the last X% of steps (detail refinement)
        """
        if self.current_step < 0:
            return False
            
        # If we know total steps, respect start/end bounds
        if self.total_steps > 0:
            progress = self.current_step / self.total_steps
            if progress < start_pct or progress > end_pct:
                return False

        # e.g., skip_interval=1 -> compute on even steps, cache on odd steps
        # step 0: 0 % 2 == 0 -> compute
        # step 1: 1 % 2 != 0 -> cache
        # step 2: 2 % 2 == 0 -> compute
        return (self.current_step % (skip_interval + 1)) != 0


class CachedAttentionWrapper:
    """Wraps an attention module to cache and reuse its outputs."""
    
    def __init__(self, original_forward, tracker: DenoisingStepTracker, config: dict, name: str = ""):
        self.original_forward = original_forward
        self.tracker = tracker
        self.config = config
        self.name = name
        self.cache = None
        self.cached_step = -1
        
    def __call__(self, x, context=None, **kwargs):
        # Determine if this is a self-attention call.
        is_self_attn = context is None or context is x
        
        # Check if we should use cache
        use_cache = (
            is_self_attn 
            and self.cache is not None
            and self.tracker.should_cache(
                skip_interval=self.config.get("skip_interval", 1),
                start_pct=self.config.get("start_percent", 0.1),
                end_pct=self.config.get("end_percent", 0.9)
            )
        )
        
        if use_cache:
            self.tracker.cache_hit_count += 1
            return self.cache
            
        # Cache miss - compute and store
        out = self.original_forward(x, context=context, **kwargs)
        
        if is_self_attn:
            self.tracker.cache_miss_count += 1
            self.cache = out
            self.cached_step = self.tracker.current_step
            
        return out


class FluxCachedAttentionHandler:
    """Specialized handler for Flux models which use a functional attention call."""
    
    def __init__(self, original_attention, tracker: DenoisingStepTracker, config: dict):
        self.original_attention = original_attention
        self.tracker = tracker
        self.config = config
        # Individual cache per block
        self.cache = {}
        
    def __call__(self, q, k, v, pe, mask=None, transformer_options={}):
        block_index = transformer_options.get("block_index")
        block_type = transformer_options.get("block_type", "unknown")
        
        if block_index is None:
            return self.original_attention(q, k, v, pe, mask=mask, transformer_options=transformer_options)
            
        key = (block_type, block_index)
        
        use_cache = (
            key in self.cache
            and self.tracker.should_cache(
                skip_interval=self.config.get("skip_interval", 1),
                start_pct=self.config.get("start_percent", 0.1),
                end_pct=self.config.get("end_percent", 0.9)
            )
        )
        
        if use_cache:
            self.tracker.cache_hit_count += 1
            return self.cache[key]
            
        # Cache miss - compute and store
        out = self.original_attention(q, k, v, pe, mask=mask, transformer_options=transformer_options)
        
        self.tracker.cache_miss_count += 1
        self.cache[key] = out
        return out


def apply_temporal_caching(model, config: dict):
    """Recursively patches attention modules in a model with caching wrappers.
    
    Args:
        model: The PyTorch model to patch
        config: Dictionary containing caching parameters (skip_interval, start_percent, etc)
        
    Returns:
        The shared DenoisingStepTracker instance
    """
    tracker = DenoisingStepTracker()
    patched_count = 0
    
    # 1. Specialized patching for Flux models
    model_type = type(model).__name__
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        model_type = type(model.model.diffusion_model).__name__
        
    if model_type == "Flux":
        try:
            import comfy.ldm.flux.math as flux_math
            if not hasattr(flux_math, "_temporal_cache_original_attention"):
                flux_math._temporal_cache_original_attention = flux_math.attention
                handler = FluxCachedAttentionHandler(flux_math._temporal_cache_original_attention, tracker, config)
                flux_math.attention = handler
                print("[TemporalCache] Patched Flux global attention function.")
        except ImportError:
            pass
    
    # 2. Recursive patching for other models (LTX, SDXL, etc)
    def patch_recursive(module, prefix=""):
        nonlocal patched_count
        
        # Direct patching on known attention module types or methods
        if hasattr(module, "forward") and hasattr(module, "to_q") and hasattr(module, "to_k"):
            # This looks like a standard attention module (e.g. comfy/ldm/modules/attention.py)
            if not hasattr(module, "_temporal_cache_original_forward"):
                module._temporal_cache_original_forward = getattr(module, "forward")
                wrapper = CachedAttentionWrapper(module._temporal_cache_original_forward, tracker, config, name=prefix)
                module.forward = wrapper
                patched_count += 1
                
        # Recurse children
        for name, child in module.named_children():
            patch_recursive(child, prefix=f"{prefix}.{name}" if prefix else name)
            
    # Apply patching to the inner diffusion model
    target = model
    if hasattr(target, "model"):
        target = target.model
    if hasattr(target, "diffusion_model"):
        target = target.diffusion_model
        
    patch_recursive(target)
    if patched_count > 0:
        print(f"[TemporalCache] Patched {patched_count} attention modules.")
    
    return tracker
