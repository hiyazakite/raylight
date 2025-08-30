import raylight
import os

import ray
import torch
import comfy
import folder_paths

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils
from .distributed_worker.ray_worker import RayWorker


class RayInitializer:
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
        if world_size > max_world_size:
            raise ValueError("To many gpus")
        if world_size == 0:
            raise ValueError("Num of cuda/cudalike device is 0")
        if world_size < ulysses_degree * ring_degree:
            raise ValueError(
                f"ERROR, num_gpus: {world_size}, is lower than {ulysses_degree=} mul {ring_degree=}"
            )
        if FSDP is True and (world_size == 1):
            raise ValueError(
                "ERROR, FSDP cannot be use in single cuda/cudalike device"
            )

        try:
            import flash_attn
            is_flash_attn_exist = True
        except ImportError:
            is_flash_attn_exist = False

        self.parallel_dict["is_xdit"] = False
        self.parallel_dict["is_fsdp"] = False
        self.parallel_dict["is_dumb_parallel"] = True

        if ulysses_degree > 1 or ring_degree > 1:
            if is_flash_attn_exist is False:
                raise ValueError("ERROR, Flash Attention is not installed, cannot use USP")
            else:
                self.parallel_dict["is_xdit"] = True
                self.parallel_dict["ulysses_degree"] = ulysses_degree
                self.parallel_dict["ring_degree"] = ring_degree

        if FSDP:
            self.parallel_dict["is_fsdp"] = True
            self.parallel_dict["is_fsdp_wrapped"] = False

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


class RayUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
                "ray_actors_init": (
                    "RAY_ACTORS_INIT",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
            },
            "optional": {"lora": ("RAY_LORA", {"default": None})},
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    def load_ray_unet(self, ray_actors_init, unet_name, weight_dtype, lora=None):
        gpu_actors = ray_actors_init["workers"]
        parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        if parallel_dict["is_fsdp"] is True and ("scale" in unet_name.lower()):
            raise ValueError(
                """Pre-FSDP filename check: Non uniform dtype FP8 transformer block is currently
                not supported for Raylight FSDP implementation, especially Ampere architecture and below,
                try non scaled model"""
            )

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        loaded_futures = []
        patched_futures = []

        for actor in gpu_actors:
            loaded_futures.append(
                actor.clear_model.remote()
            )
        ray.get(loaded_futures)
        loaded_futures = []

        for actor in gpu_actors:
            loaded_futures.append(
                actor.set_lora_list.remote(lora)
            )
        ray.get(loaded_futures)
        loaded_futures = []

        if parallel_dict["is_fsdp"] is True:
            worker0 = ray.get_actor("RayWorker:0")
            ray.get(worker0.load_unet.remote(unet_path, model_options=model_options))
            meta_model = ray.get(worker0.get_meta_model.remote())

            for actor in gpu_actors:
                if actor != worker0:
                    loaded_futures.append(actor.set_meta_model.remote(meta_model))

            ray.get(loaded_futures)
            loaded_futures = []
        else:
            for actor in gpu_actors:
                loaded_futures.append(
                    actor.load_unet.remote(unet_path, model_options=model_options)
                )
            ray.get(loaded_futures)
            loaded_futures = []

        for actor in gpu_actors:
            if parallel_dict["is_xdit"]:
                patched_futures.append(actor.patch_usp.remote())

        ray.get(patched_futures)

        return (ray_actors_init,)


class RayLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The name of the LoRA."},
                ),
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
            },
            "optional": {
                "prev_ray_lora": ("RAY_LORA", {"default": None})
            }
        }

    RETURN_TYPES = ("RAY_LORA",)
    RETURN_NAMES = ("ray_lora",)
    FUNCTION = "load_lora"
    CATEGORY = "Raylight"

    def load_lora(self, lora_name, strength_model, prev_ray_lora=None):
        loras_list = []

        if strength_model == 0.0:
            if prev_ray_lora is not None:
                loras_list.extend(prev_ray_lora)
            return (loras_list,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = {
            "path": lora_path,
            "strength_model": strength_model,
        }

        if prev_ray_lora is not None:
            loras_list.extend(prev_ray_lora)

        loras_list.append(lora)
        return (loras_list,)


class XFuserKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "LATENT",
        "LATENT",
        "LATENT",
    )
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        gpu_actors = ray_actors["workers"]
        futures = [
            actor.common_ksampler.remote(
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=force_full_denoise,
            )
            for actor in gpu_actors
        ]

        results = ray.get(futures)
        return tuple(result[0] for result in results[0:4])


NODE_CLASS_MAPPINGS = {
    "XFuserKSamplerAdvanced": XFuserKSamplerAdvanced,
    "RayUNETLoader": RayUNETLoader,
    "RayLoraLoader": RayLoraLoader,
    "RayInitializer": RayInitializer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserKSamplerAdvanced": "XFuser KSampler Advanced",
    "RayUNETLoader": "Load Diffusion Model (Ray)",
    "RayLoraLoader": "Load Lora Model (Ray)",
    "RayInitializer": "Ray Init Actor",
}
