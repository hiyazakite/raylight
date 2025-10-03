import raylight
import os
import folder_paths

import ray
import torch

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils
from .distributed_worker.ray_worker import (
    make_ray_actor_fn,
    ray_nccl_tester
)


class RayInitializerDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {"default": "local"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
                "GPU": ("INT", {"default": 2}),
                "ulysses_degree": ("INT", {"default": 2}),
                "ring_degree": ("INT", {"default": 1}),
                "FSDP": ("BOOLEAN", {"default": False}),
                "FSDP_CPU_OFFLOAD": ("BOOLEAN", {"default": False}),
                "XFuser_attention": ([
                    "TORCH",
                    "FLASH_ATTN",
                    "FLASH_ATTN_3",
                    "SAGE_AUTO_DETECT",
                    "SAGE_FP16_TRITON",
                    "SAGE_FP16_CUDA",
                    "SAGE_FP8_CUDA",
                    "SAGE_FP8_SM90"
                ], {"default": "TORCH"})
            }
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    def spawn_actor(
        self,
        ray_cluster_address,
        ray_cluster_namespace,
        GPU,
        ulysses_degree,
        ring_degree,
        FSDP,
        FSDP_CPU_OFFLOAD,
        XFuser_attention
    ):
        # THIS IS PYTORCH DIST ADDRESS
        # (TODO) Change so it can be use in cluster of nodes. but it is long waaaaay down in the priority list
        # os.environ['TORCH_CUDA_ARCH_LIST'] = ""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.parallel_dict = dict()

        # Currenty not implementing CFG parallel, since LoRa can enable non cfg run
        world_size = GPU
        max_world_size = torch.cuda.device_count()
        if world_size > max_world_size:
            raise ValueError("To many gpus")

        self.parallel_dict["is_xdit"] = False
        self.parallel_dict["is_fsdp"] = False
        self.parallel_dict["is_dumb_parallel"] = True

        self.parallel_dict["ulysses_degree"] = ulysses_degree
        self.parallel_dict["ring_degree"] = 1

        if ulysses_degree > 0 or ring_degree > 0:
            self.parallel_dict["attention"] = XFuser_attention
            self.parallel_dict["is_xdit"] = True
            self.parallel_dict["ulysses_degree"] = ulysses_degree
            self.parallel_dict["ring_degree"] = ring_degree

        if FSDP:
            self.parallel_dict["fsdp_cpu_offload"] = FSDP_CPU_OFFLOAD
            self.parallel_dict["is_fsdp"] = True

        try:
            # Shut down so if comfy user try another workflow it will not cause error
            ray.shutdown()
            ray.init(
                ray_cluster_address,
                namespace=ray_cluster_namespace,
                runtime_env={"py_modules": [raylight]},
            )
        except Exception as e:
            ray.shutdown()
            ray.init(runtime_env={"py_modules": [raylight]})
            raise RuntimeError(f"Ray connection failed: {e}")

        ray_nccl_tester(world_size)
        ray_actor_fn = make_ray_actor_fn(world_size, self.parallel_dict)
        ray_actors = ray_actor_fn()
        return ([ray_actors, ray_actor_fn],)


class RayLoraLoader2:
    def __init__(self):
        self.loaded_lora_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_lora"
    CATEGORY = "Raylight"

    # Cannot just run ray_patch or lora will not be tracked, if injecting
    # lora directly SerDe on ray.put might get a toll, emphasized on might
    def load_lora(self, ray_actors, lora_name, strength_model):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        if strength_model == 0:
            return (ray_actors,)

        if self.loaded_lora_path is not None:
            if self.loaded_lora_path == lora_path:
                lora_path = self.loaded_lora
            else:
                self.loaded_lora_path = None

        gpu_workers = ray_actors["workers"]
        futures = [actor.load_lora.remote(lora_path, strength_model) for actor in gpu_workers]
        ray.get(futures)

        return (ray_actors,)


NODE_CLASS_MAPPINGS = {
    "RayInitializerDebug": RayInitializerDebug,
    "RayLoraLoader2": RayLoraLoader2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayInitializerDebug": "Ray Init Actor (Debug)",
}
