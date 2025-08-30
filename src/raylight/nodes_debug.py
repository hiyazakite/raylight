import raylight
import os

import ray
import torch

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils
from .distributed_worker.ray_worker import RayWorker


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
                "DEBUG_USP": ("BOOLEAN", {"default": False}),
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
        DEBUG_USP,
    ):
        # THIS IS PYTORCH DIST ADDRESS
        # (TODO) Change so it can be use in cluster of nodes. but it is long down in the priority list
        # os.environ['TORCH_CUDA_ARCH_LIST'] = ""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        self.parallel_dict = dict()

        # Currenty not implementing CFG parallel, since LoRa can enable non cfg run
        world_size = GPU
        max_world_size = torch.cuda.device_count()
        self.parallel_dict["is_xdit"] = False
        self.parallel_dict["is_fsdp"] = False
        self.parallel_dict["is_dumb_parallel"] = True

        if ulysses_degree > 1 or ring_degree > 1:
            self.parallel_dict["is_xdit"] = True
            self.parallel_dict["ulysses_degree"] = ulysses_degree
            self.parallel_dict["ring_degree"] = ring_degree

        if FSDP:
            self.parallel_dict["is_fsdp"] = True
            self.parallel_dict["is_fsdp_wrapped"] = False

        if DEBUG_USP:
            self.parallel_dict["is_xdit"] = True
            self.parallel_dict["ulysses_degree"] = 1

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

        ray_actors = dict()
        gpu_actor = ray.remote(RayWorker)
        gpu_actors = []
        for local_rank in range(world_size):
            gpu_actors.append(
                gpu_actor.options(num_gpus=1, name=f"RayWorker:{local_rank}").remote(
                    local_rank=local_rank,
                    world_size=world_size,
                    device_id=0,
                    parallel_dict=self.parallel_dict,
                )
            )
        ray_actors["workers"] = gpu_actors

        for actor in ray_actors["workers"]:
            ray.get(actor.__ray_ready__.remote())

        return (ray_actors,)


NODE_CLASS_MAPPINGS = {
    "RayInitializerDebug": RayInitializerDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayInitializerDebug": "Ray Init Actor (Debug)",
}
