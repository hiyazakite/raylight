"""ComfyUI Node for DiTFastAttn Temporal Attention Caching."""

import ray

class RayDiTFastAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "enable_caching": ("BOOLEAN", {"default": True}),
                "skip_interval": ("INT", {"default": 1, "min": 1, "max": 10}),
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    FUNCTION = "apply_fast_attn"
    CATEGORY = "Raylight/extra"

    def apply_fast_attn(self, ray_actors, enable_caching, skip_interval, start_percent, end_percent):
        gpu_actors = ray_actors["workers"]
        
        config = {
            "enabled": enable_caching,
            "skip_interval": skip_interval,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        # Apply to all workers
        ray.get([actor.apply_dit_fast_attn.remote(config) for actor in gpu_actors])
        
        return (ray_actors,)

NODE_CLASS_MAPPINGS = {
    "RayDiTFastAttn": RayDiTFastAttn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayDiTFastAttn": "Ray DiTFastAttn (Temporal Cache)",
}
