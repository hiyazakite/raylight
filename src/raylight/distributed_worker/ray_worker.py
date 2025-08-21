import types
import os
import gc
from datetime import timedelta

import torch
import torch.distributed as dist

import comfy
from comfy import (
    sd,
    sample,
    utils,
)  # Must manually insert comfy package or ray cannot import raylight to cluster
import comfy.patcher_extension as pe
from comfy import model_base

import raylight.distributed_worker.context_parallel as cp


# Temp solution, should be init to meta first then load_state_dict, CPU for now
# ERROR: Change scale_weight to a 1D tensor with numel equal to 1.,
# inside transformer of scaled_model there are scale_weight tensors that are 0 dim,
# for now use non scaled model
def fsdp_inject_callback(
    model_patcher, device_to, lowvram_model_memory, force_patch_weights, full_load
):
    print(f"[Rank {dist.get_rank()}] Applying FSDP to {type(model_patcher.model.diffusion_model).__name__}")
    if isinstance(model_patcher.model, model_base.WAN21) or isinstance(model_patcher.model, model_base.WAN22):
        from ..wan.distributed.fsdp import shard_model_fsdp2
        model_patcher.model = shard_model_fsdp2(model_patcher.model, device_to)

    elif isinstance(model_patcher.model, model_base.Flux):
        from ..flux.distributed.fsdp import shard_model_fsdp2
        model_patcher.model = shard_model_fsdp2(model_patcher.model, device_to)

    elif isinstance(model_patcher.model, model_base.QwenImage):
        from ..qwen_image.distributed.fsdp import shard_model_fsdp2
        model_patcher.model = shard_model_fsdp2(model_patcher.model, device_to)

    elif isinstance(model_patcher.model, model_base.HunyuanVideo):
        from ..hunyuan_video.distributed.fsdp import shard_model_fsdp2
        model_patcher.model = shard_model_fsdp2(model_patcher.model, device_to)

    else:
        raise ValueError(f"{type(model_patcher.model.diffusion_model).__name__} IS CURRENTLY NOT SUPPORTED FOR FSDP")

    comfy.model_management.soft_empty_cache()
    gc.collect()
    dist.barrier()


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


class RayWorker:
    def __init__(self, local_rank, world_size, device_id, parallel_dict):
        self.model = None
        self.model_type = None
        self.local_rank = local_rank
        self.world_size = world_size
        self.device_id = device_id
        self.noise_add = 0

        self.parallel_dict = parallel_dict
        self.parallel_dict["is_fsdp_wrapped"] = False
        self.device = torch.device(f"cuda:{self.device_id}")

        if self.model is not None:
            self.is_model_load = True
        else:
            self.is_model_load = False

        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=self.world_size,
                timeout=timedelta(minutes=1)
            )
            pg = dist.group.WORLD
            cp.set_cp_group(pg, list(range(world_size)), local_rank)

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
                f"Running Ray in normal seperate sampler with: {world_size} number of workers"
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
                    sequence_parallel_degree=world_size,
                    ring_degree=1,
                    ulysses_degree=cp_size,
                )
            else:
                print(
                    f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}"
                )
                initialize_model_parallel(
                    sequence_parallel_degree=world_size,
                    ring_degree=ring_degree,
                    ulysses_degree=ulysses_degree,
                )

    def get_parallel_dict(self):
        return self.parallel_dict

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

    def patch_fsdp(self):
        if self.parallel_dict["is_fsdp_wrapped"] is False:
            self.model.add_callback(
                pe.CallbacksMP.ON_LOAD,
                fsdp_inject_callback,
            )
            self.parallel_dict["is_fsdp_wrapped"] = True
            print("FSDP registered")
        else:
            print("FSDP already registered, skipping wrapping")

    def load_unet(self, unet_path, model_options):
        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options
        )
        if self.lora_list is not None:
            self.load_lora()

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(self,):
        return self.lora_list

    def load_lora(self,):
        for lora in self.lora_list:
            lora_path = lora["path"]
            strength_model = lora["strength_model"]
            lora_model = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.model = comfy.sd.load_lora_for_models(
                self.model, None, lora_model, strength_model, 0
            )[0]

            del lora_model

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

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
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
        return (out,)
