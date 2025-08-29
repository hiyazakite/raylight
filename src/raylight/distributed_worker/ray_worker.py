import types
import os
import gc
from datetime import timedelta
import warnings

import torch
import torch.distributed as dist
import ray

import comfy
from comfy import (
    sd,
    sample,
    utils,
)  # Must manually insert comfy package or ray cannot import raylight to cluster
import comfy.patcher_extension as pe
from comfy import model_base

import raylight.distributed_worker.context_parallel as cp
from raylight.comfy_dist.model_patcher import make_ray_patch_weight_to_device
from raylight.comfy_dist.sd import load_lora_for_models as ray_load_lora_for_models

# see comment on init_process_group
warnings.filterwarnings(
    "ignore",
    message="No device id is provided via `init_process_group` or `barrier`.*"
)


def usp_inject_callback(
    model_patcher, device_to, lowvram_model_memory, force_patch_weights, full_load
):
    base_model = model_patcher.model

    if isinstance(base_model, model_base.WAN21) or isinstance(
        base_model, model_base.WAN22
    ):
        from ..wan.distributed.xdit_context_parallel import (
            usp_attn_forward,
            usp_dit_forward,
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    # PlaceHolder For now
    elif isinstance(base_model, model_base.Flux):
        from ..flux.distributed.xdit_context_parallel import (
            usp_dit_forward,
            usp_attn_forward,
        )

        model = base_model.diffusion_model
        dist.barrier()

    elif isinstance(base_model, model_base.QwenImageTransformer2DModel):
        from ..qwen_image.distributed.xdit_context_parallel import (
            usp_dit_forward,
            usp_attn_forward,
        )

        model = base_model.diffusion_model
        dist.barrier()

    else:
        print(
            f"Model: {type(base_model).__name__}, is not yet supported for USP Parallelism"
        )

# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.

class RayWorker:
    def __init__(self, local_rank, world_size, device_id, parallel_dict):
        self.model = None
        self.model_type = None
        self.local_rank = local_rank
        self.world_size = world_size
        self.device_id = device_id
        self.noise_add = 0
        self.state_dict = None

        self.parallel_dict = parallel_dict
        self.parallel_dict["is_fsdp_wrapped"] = False
        self.device = torch.device(f"cuda:{self.device_id}")
        self.device_mesh = None

        if self.model is not None:
            self.is_model_load = True
        else:
            self.is_model_load = False

        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
            # NCCL USP error if we put device into dist.init_process_group
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=self.world_size,
                timeout=timedelta(minutes=1),
            )
            self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(self.world_size,))
            pg = dist.group.WORLD
            cp.set_cp_group(pg, list(range(self.world_size)), local_rank)

            print("Running NCCL COMM pre-run")

            # Each rank contributes rank+1
            x = torch.ones(1, device=self.device) * (self.local_rank + 1)
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            result = x.item()

            # Expected sum = N(N+1)/2
            expected = self.world_size * (self.world_size + 1) // 2

            if abs(result - expected) > 1e-3:
                raise RuntimeError(
                    f"[Rank {self.local_rank}] COMM test failed: "
                    f"got {result}, expected {expected}. "
                    f"world_size may be mismatched!"
                )
            else:
                print(f"[Rank {self.local_rank}] COMM test passed ✅ (result={result})")

        else:
            print(
                f"Running Ray in normal seperate sampler with: {self.world_size} number of workers"
            )
            self.noise_add = self.local_rank

        # From mochi-xdit, xdit, pipelines.py
        # I dont use globals since it does not work as module
        if self.parallel_dict["is_xdit"]:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            cp_rank, cp_size = cp.get_cp_rank_size()
            ulysses_degree = self.parallel_dict["ulysses_degree"]
            ring_degree = self.parallel_dict["ring_degree"]

            print("XDiT is enable")
            init_distributed_environment(rank=cp_rank, world_size=cp_size)

            if ulysses_degree is None and ring_degree is None:
                print(
                    f"No usp config, use default config: ulysses_degree={cp_size}, ring_degree=1"
                )
                initialize_model_parallel(
                    sequence_parallel_degree=self.world_size,
                    ring_degree=1,
                    ulysses_degree=cp_size,
                )
            else:
                print(
                    f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
                )
                initialize_model_parallel(
                    sequence_parallel_degree=self.world_size,
                    ring_degree=ring_degree,
                    ulysses_degree=ulysses_degree,
                )

    def get_parallel_dict(self):
        return self.parallel_dict

    def clear_model(self):
        self.model = None
        gc.collect()
        comfy.model_management.soft_empty_cache()

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict

    def get_local_rank(self):
        return self.local_rank

    def is_model_loaded(self):
        return self.is_model_load

    def patch_usp(self):
        self.model.add_callback(
            pe.CallbacksMP.ON_LOAD,
            usp_inject_callback,
        )
        print("USP registered")

    def set_meta_model(self, model):
        first_param_device = next(model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            self.model = model
        else:
            raise ValueError("Model being set is not meta, can cause OOM in large model")

    def get_meta_model(self):
        first_param_device = next(self.model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            return self.model
        else:
            raise ValueError("Model recieved is not meta, can cause OOM in large model")

    def load_unet(self, unet_path, model_options):
        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options
        )
        if self.lora_list is not None:
            self.load_lora()
        if self.parallel_dict["is_fsdp"] is True:
            if self.local_rank == 0:
                self.state_dict = self.model.model_state_dict()
            self.model.model = self.model.model.to("meta")
            comfy.model_management.soft_empty_cache()
            gc.collect()

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(self,):
        return self.lora_list

    def load_lora(self,):
        for lora in self.lora_list:
            lora_path = lora["path"]
            strength_model = lora["strength_model"]
            lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)

            if self.parallel_dict["is_fsdp"] is True:
                self.model = ray_load_lora_for_models(
                    self.model, lora_model, strength_model
                )
            else:
                self.model = comfy.sd.load_lora_for_models(
                    self.model, None, lora_model, strength_model, 0
                )[0]
            del lora_model

    def patch_fsdp(self,):
        from torch.distributed.fsdp import FSDPModule
        print(f"[Rank {dist.get_rank()}] Applying FSDP to {type(self.model.model.diffusion_model).__name__}")

        if not isinstance(self.model.model.diffusion_model, FSDPModule):
            if isinstance(self.model.model, model_base.WAN21) or isinstance(self.model.model, model_base.WAN22):
                from ..wan.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.device, self.state_dict)

            elif isinstance(self.model.model, model_base.Flux):
                from ..flux.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.device, self.state_dict)

            elif isinstance(self.model.model, model_base.QwenImage):
                from ..qwen_image.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.device, self.state_dict)

            elif isinstance(self.model.model, model_base.HunyuanVideo):
                from ..hunyuan_video.distributed.fsdp import shard_model_fsdp2
                self.model.model = shard_model_fsdp2(self.model.model, self.device, self.state_dict)

            else:
                raise ValueError(f"{type(self.model.model.diffusion_model).__name__} IS CURRENTLY NOT SUPPORTED FOR FSDP")

            self.state_dict = None
            comfy.model_management.soft_empty_cache()
            gc.collect()
            dist.barrier()
            print("FSDP registered")
        else:
            print("FSDP already registered, skip wrapping...")

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
            noise = comfy.sample.prepare_noise(
                latent_image, seed + self.noise_add, batch_inds
            )

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        disable_pbar = comfy.utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        if self.parallel_dict["is_fsdp"] is True:
            self.model.patch_weight_to_device = types.MethodType(
                make_ray_patch_weight_to_device(convert_dtensor=True, device_mesh=self.device_mesh),
                self.model
            )
            self.patch_fsdp()

        with torch.no_grad():
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

        if ray.get_runtime_context().get_accelerator_ids()["GPU"][0] == "0":
            self.model.detach()
        comfy.model_management.soft_empty_cache()
        gc.collect()
        return (out,)
