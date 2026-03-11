import os
import sys
import torch
import time

sys.path.append("/root/ComfyUI")
sys.path.append("/root/ComfyUI/custom_nodes/raylight/src")

# Mock ComfyUI folder paths
import folder_paths
folder_paths.base_path = "/root/ComfyUI"
folder_paths.models_dir = os.path.join(folder_paths.base_path, "models")
folder_paths.folder_names_and_paths["diffusion_models"] = ([os.path.join(folder_paths.models_dir, "diffusion_models")], [".safetensors"])

from raylight.nodes import RayWorkerInit, RayUNETLoader, XFuserKSamplerAdvanced
import ray

def run_test():
    print("Initializing Raylight...")
    init_node = RayWorkerInit()
    # Mock parallel_type
    ray_actors_init = init_node.init_ray(
        parallel_type="Sequence (Standard)",
        degree=1,
        cfg_degree=1,
        attention_backend="XFuser_attention",
        attention_type="TORCH",
        use_kitchen=False,
        kv_cache_quant_enable=False,
        kv_cache_quant_bits=8,
        delta_compression="BINARY"
    )[0]

    print("Loading UNet...")
    # Find any unet
    unet_name = "ltx-video-2b-v0.2.1.safetensors" # Or another accessible one
    loader_node = RayUNETLoader()
    
    try:
        ray_actors = loader_node.load_ray_unet(ray_actors_init, unet_name, "bf16")[0]
    except FileNotFoundError:
        print("Model not found in diffusion_models, trying checkpoints...")
        try:
             ray_actors = loader_node.load_ray_unet(ray_actors_init, "ltxv.safetensors", "bf16")[0]
        except Exception as e:
             print(f"Skipping test, could not load model: {e}")
             return

    # Create dummy latent and args
    latent_image = {"samples": torch.zeros((1, 16, 64, 64), device='cpu'), "noise_mask": None}
    positive = None # Dummy
    negative = None # Dummy

    sampler_node = XFuserKSamplerAdvanced()

    print("Starting loops...")
    for i in range(5):
        print(f"\n--- RUN {i+1} ---")
        start = time.time()
        
        # We need mock conditioning. Raylight expects conditioning as strings or list
        # We will bypass ComfyUI's sample_custom and just pass raw dicts if needed
        # Actually XFuserKSamplerAdvanced takes (ray_actors, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image)
        try:
             res = sampler_node.ray_sample(
                 ray_actors,
                 add_noise="enable",
                 noise_seed=42 + i,
                 steps=4,
                 cfg=1.0,
                 sampler_name="euler",
                 scheduler="normal",
                 positive=[],
                 negative=[],
                 latent_image=latent_image
             )
        except Exception as e:
             print(f"Sampling error: {e}")
             break
        
        torch.cuda.synchronize()
        end = time.time()
        print(f"Run {i+1} took {end - start:.2f} seconds.")

    print("Shutting down Ray...")
    ray.shutdown()

if __name__ == "__main__":
    run_test()
