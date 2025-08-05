import types
import sys

import ray
import torch
import torch.distributed as dist
import comfy
import folder_paths
import time

from comfy import sd, sample, utils

import raylight

## GLOBAL SETTER ##
from typing import Tuple

_USE_XDIT = False


def set_use_xdit(use_dit: bool) -> None:
    """Set whether to use DIT model.

    Args:
        use_dit: Boolean flag indicating whether to use xdur
    """
    global _USE_XDIT
    _USE_XDIT = use_dit
    print(f"The xDiT flag use_xdit={use_dit}")


def is_use_xdit() -> bool:
    return _USE_XDIT


_ULYSSES_DEGREE = None
_RING_DEGREE = None
_CFG_PARALLEL = None


def set_usp_config(ulysses_degree: int, ring_degree: int, cfg_parallel: bool) -> None:
    global _ULYSSES_DEGREE, _RING_DEGREE, _CFG_PARALLEL
    _ULYSSES_DEGREE = ulysses_degree
    _RING_DEGREE = ring_degree
    _CFG_PARALLEL = cfg_parallel
    print(
        f"Now we use xdit with ulysses degree {ulysses_degree}, ring degree {ring_degree}, and CFG parallel {cfg_parallel}"
    )


def get_usp_config() -> Tuple[int, int, bool]:
    return _ULYSSES_DEGREE, _RING_DEGREE, _CFG_PARALLEL


_USE_FSDP = False


def set_use_fsdp(use_fsdp: bool) -> None:
    global _USE_FSDP
    _USE_FSDP = use_fsdp
    print(f"The FSDP flag use_fsdp={use_fsdp}")


def is_use_fsdp() -> bool:
    return _USE_FSDP


class Timer:
    def __init__(self):
        self.times = {}  # Dictionary to store times per stage

    def __call__(self, name):
        print(f"Timing {name}")
        return self.TimerContextManager(self, name)

    def print_stats(self):
        total_time = sum(self.times.values())
        # Print table header
        print("{:<20} {:>10} {:>10}".format("Stage", "Time(s)", "Percent"))
        for name, t in self.times.items():
            percent = (t / total_time) * 100 if total_time > 0 else 0
            print("{:<20} {:>10.2f} {:>9.2f}%".format(name, t, percent))

    class TimerContextManager:
        def __init__(self, outer, name):
            self.outer = outer  # Reference to the Timer instance
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            end_time = time.perf_counter()
            elapsed = end_time - self.start_time
            self.outer.times[self.name] = self.outer.times.get(self.name, 0) + elapsed



class RayWorker:

    master_address = None
    master_port = None
    use_usp = False


    def __init__(self, local_rank):
        self.model = None
        self.local_rank = local_rank
        print("Initializing Local Rank: ", self.local_rank)
        # Unused
        self.device_id = None
        self.world_size = None
        self.decode_type = None
        self.decode_args = None
        self.use_fsdp = None
        self.use_xdit = None
        self.ulysses_degree = None
        self.ring_degree = None
        self.cfg_parallel = None

        t = Timer()
        with t("init_process_group"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=RayWorker.world_size,
                device_id=self.device_id,  # force non-lazy init
            )

    """
    Just Placeholder for now, since without USP it is just
    using both gpu to sample different latent
    """

    def get_sys_path(self):
        return sys.path

    def patch_usp(self):

        # Just place holder so LSP not complaining
        def usp_dit_forward():
            pass

        def usp_self_attn_patch():
            pass

        block_len = len(self.model.model.diffusion_model.blocks)
        model_options = {
            "dtype": torch.float8_e4m3fn,
            "transformer_options": {
                "patches_replace": {
                    "dit": {
                        **{
                            ("self_attn", i): usp_self_attn_patch
                            for i in range(block_len)
                        }
                    }
                }
            },
        }

        self.model.model_options = model_options
        self.model.model.diffusion_model.forward = types.MethodType(
            usp_dit_forward, self.model.model.diffusion_model
        )

    """
    instance_RayWorker.load_unet.remote(*args, **kwargs)
    """

    def load_unet(self, unet_path, model_options):
        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options
        )
        return self.model.size

    """
    instance_RayWorker.load_lora.remote(*args, **kwargs)
    """

    def load_lora(self, lora, strength_model):
        self.model = comfy.sd.load_lora_for_models(
            self.model, None, lora, strength_model, 0
        )[0]
        return self.model

    """
    instance_RayWorker.common_ksampler.remote(*args, **kwargs)
    """

    def common_ksampler(
        self,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
    ):
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)

        if disable_noise:
            noise = torch.zeros(
                latent_image.size(),
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                device="cpu",
            )
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            self.model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        out = latent.copy()
        out["samples"] = samples
        return (out,)


class RayInitializer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {"default": "local"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    def spawn_actor(self, ray_cluster_address, ray_cluster_namespace):
        try:
            ray.init(
                ray_cluster_address,
                namespace=ray_cluster_namespace,
                runtime_env={"py_modules": [raylight]},
            )
        except Exception as e:
            ray.init(runtime_env={"py_modules": [raylight]})
            raise RuntimeError(f"Ray connection failed: {e}")

        world_size = torch.cuda.device_count()
        print(f"current world size: {world_size}")
        RemoteActor = ray.remote(RayWorker)

        actors = []
        for local_rank in range(world_size):
            actors.append(RemoteActor.options(num_gpus=1, name=f"RayWorker:{local_rank}").remote(local_rank=local_rank))

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

    RETURN_TYPES = ("RAY_ACTORS", "STRING")
    RETURN_NAMES = ("ray_actors", "model_size")
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
            actor.load_unet.remote(unet_path, model_options=model_options)
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

    RETURN_TYPES = ("LATENT", "LATENT",)
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
            final_sample.append(actor.common_ksampler.remote(
                noise_seed + additional_noise,
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
            ))

        return (ray.get(final_sample[0])[0], ray.get(final_sample[1])[0],)


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
