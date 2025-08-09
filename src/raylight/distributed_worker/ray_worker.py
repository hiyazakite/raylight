import types

import torch
import torch.distributed as dist
import comfy

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils

from ..wan.distributed.xdit_context_parallel import usp_dit_forward, usp_attn_forward
from ..wan.distributed.fsdp import shard_model

from ..distributed_worker import context_parallel as cp

from functools import partial


from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)


class RayWorker:
    def __init__(self, local_rank, world_size, device_id, parallel_dict):
        self.model = None
        self.model_type = None
        self.local_rank = local_rank
        self.world_size = world_size
        self.device_id = device_id

        self.parallel_dict = parallel_dict

        # TO DO, Actual error checking to determine total rank_nums is equal to world size
        self.device = torch.device(f"cuda:{self.device_id}")
        dist.init_process_group(
            "nccl",
            rank=local_rank,
            world_size=self.world_size,
            device_id=self.device,
        )
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), local_rank)

        # From mochi-xdit, xdit, pipelines.py
        # I dont use globals since it does not work as module
        if self.parallel_dict["is_xdit"]:
            cp_rank, cp_size = cp.get_cp_rank_size()
            ulysses_degree = self.parallel_dict["ulysses_degree"]
            ring_degree = self.parallel_dict["ring_degree"]

            print(f"XDiT is enable, with {ring_degree=}, {ulysses_degree=}")
            init_distributed_environment(rank=cp_rank, world_size=cp_size)

            if ulysses_degree is None and ring_degree is None:
                print(f"No usp config, use default config: ulysses_degree={cp_size}, ring_degree=1, CFG parallel false")
                initialize_model_parallel(
                    sequence_parallel_degree=world_size,
                    ring_degree=1,
                    ulysses_degree=cp_size,
                )
            else:
                if ulysses_degree is None:
                    ulysses_degree = world_size // ring_degree
                if ring_degree is None:
                    ring_degree = world_size // ulysses_degree
                print(f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, CFG parallel false")
                initialize_model_parallel(
                    sequence_parallel_degree=world_size,
                    ring_degree=ring_degree,
                    ulysses_degree=ulysses_degree,
                )

    def get_parallel_dict(self):
        return self.parallel_dict

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict

    def patch_usp(self):
        print("Initializing USP")
        for block in self.model.model.diffusion_model.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.model.model.diffusion_model.forward_orig = types.MethodType(
            usp_dit_forward, self.model.model.diffusion_model
        )
        print("USP APPLIED")

    def patch_fsdp(self):
        print("Initializing FSDP")
        shard_fn = partial(shard_model, device_id=self.device_id)
        self.model.model.diffusion_model = shard_fn(self.model.model.diffusion_model)
        print("FSDP APPLIED")

    def load_unet(self, unet_path, model_options):
        self.model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options
        )
        if self.parallel_dict["is_xdit"]:
            self.patch_usp()
            if self.parallel_dict["is_fsdp"]:
                self.patch_fsdp()

        return None

    def load_lora(self, lora, strength_model):
        self.model = comfy.sd.load_lora_for_models(
            self.model, None, lora, strength_model, 0
        )[0]
        return self.model

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
