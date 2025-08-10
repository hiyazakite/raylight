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
                "ulysses_degree": ("INT", {"default": 2, "tooltip": "degree of Ulyssess attention, prefer this than ring"}),
                "ring_degree": ("INT", {"default": 1, "tooltip": "degree of ring attention"}),
                "FSDP": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    def spawn_actor(self, ray_cluster_address, ray_cluster_namespace, ulysses_degree, ring_degree, FSDP):
        # THIS IS PYTORCH DIST ADDRESS
        # (TODO) Change so it can be edited
        # os.environ['TORCH_CUDA_ARCH_LIST'] = ""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        self.parallel_dict = dict()

        # Currenty not implementing CFG parallel, since LoRa can enable non cfg run
        world_size = torch.cuda.device_count()
        if (ulysses_degree * ring_degree) > world_size:
            raise ValueError(f"ERROR, num_gpus: {world_size}, is lower than {ulysses_degree=} * {ring_degree=}")

        if ulysses_degree > 1 or ring_degree > 1:
            self.parallel_dict["is_xdit"] = True
            self.parallel_dict["ulysses_degree"] = ulysses_degree
            self.parallel_dict["ring_degree"] = ring_degree
            if FSDP:
                self.parallel_dict["is_fsdp"] = True
            else:
                self.parallel_dict["is_fsdp"] = False
                self.parallel_dict["is_dumb_parallel"] = True
        else:
            self.parallel_dict["is_xdit"] = False
            self.parallel_dict["is_fsdp"] = False

        try:
            # Shut down so if comfy user try another workflow it will not cause
            # error
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

        RemoteActor = ray.remote(RayWorker)
        actors = []
        for local_rank in range(world_size):
            actors.append(
                RemoteActor.options(num_gpus=1, name=f"RayWorker:{local_rank}").remote(
                    local_rank=local_rank, world_size=world_size, device_id=0, parallel_dict=self.parallel_dict
                )
            )

        for actor in actors:
            ray.get(actor.__ray_ready__.remote())

        return (actors,)


class XFuserLoraLoaderModelOnly:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
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
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_lora"
    CATEGORY = "Raylight"

    def load_lora(self, ray_actors, lora_name, strength_model):
        if strength_model == 0:
            return (ray_actors,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        for actor in ray_actors:
            actor.load_lora.remote(lora, strength_model)

        return (ray_actors,)


class XFuserUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    def load_ray_unet(self, ray_actors, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        for actor in ray_actors:
            parallel_dict = ray.get(actor.get_parallel_dict.remote())
            ray.get(actor.load_unet.remote(unet_path, model_options=model_options))

            if parallel_dict["is_xdit"]:
                actor.patch_usp.remote()

            # THIS IS ACTUALLY BAD, komikndr, loading FSDP twice!!!!
            if parallel_dict["is_fsdp"]:
                actor.patch_fsdp.remote()

        return (ray_actors,)


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
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
            }
        }

    RETURN_TYPES = (
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

        final_sample = []
        for additional_noise, actor in enumerate(ray_actors):
            final_sample.append(
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
            )

        return (ray.get(final_sample[0])[0],)


NODE_CLASS_MAPPINGS = {
    "XFuserKSamplerAdvanced": XFuserKSamplerAdvanced,
    "XFuserUNETLoader": XFuserUNETLoader,
    "XFuserLoraLoaderModelOnly": XFuserLoraLoaderModelOnly,
    "RayInitializer": RayInitializer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserKSamplerAdvanced": "XFuser KSampler Advanced",
    "XFuserUNETLoader": "Load Diffusion Model (Ray)",
    "XFuserLoraLoaderModelOnly": "Load Lora Model (Ray)",
    "RayInitializer": "Ray Init Actor",
}
