from __future__ import annotations

import collections
import logging
import gc

import torch
import torch.distributed as dist

import comfy
from comfy import model_base
from comfy.model_patcher import (get_key_weight,
                                 string_to_seed,
                                 move_weight_functions)

from comfy.patcher_extension import CallbacksMP
from raylight import comfy_dist


class FSDPModelPatcher(comfy.model_patcher.ModelPatcher):

    def patch_fsdp(self, model_state_dict=None, device_to=None):
        if model_state_dict is None:
            model_state_dict = self.model_sd
        if device_to is None:
            self.device_to = "cpu"
        from torch.distributed.fsdp import FSDPModule
        print(f"[Rank {dist.get_rank()}] Applying FSDP to {type(self.model.diffusion_model).__name__}")

        if not isinstance(self.model.diffusion_model, FSDPModule):
            if isinstance(self.model, model_base.WAN21) or isinstance(self.model, model_base.WAN22):
                from ..wan.distributed.fsdp import shard_model_fsdp2
                self.model = shard_model_fsdp2(self.model, device_to, model_state_dict)

            elif isinstance(self.model, model_base.Flux):
                from ..flux.distributed.fsdp import shard_model_fsdp2
                self.model = shard_model_fsdp2(self.model, device_to, model_state_dict)

            elif isinstance(self.model, model_base.QwenImage):
                from ..qwen_image.distributed.fsdp import shard_model_fsdp2
                self.model = shard_model_fsdp2(self.model, device_to, model_state_dict)

            elif isinstance(self.model, model_base.HunyuanVideo):
                from ..hunyuan_video.distributed.fsdp import shard_model_fsdp2
                self.model = shard_model_fsdp2(self.model, device_to, model_state_dict)

            else:
                raise ValueError(f"{type(self.model.diffusion_model).__name__} IS CURRENTLY NOT SUPPORTED FOR FSDP")

            self.state_dict = None
            comfy.model_management.soft_empty_cache()
            gc.collect()
            dist.barrier()
            print("FSDP registered")
        else:
            print("FSDP already registered, skip wrapping...")

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, convert_dtensor=False):
        if key not in self.patches:
            return
        weight, set_func, convert_func = get_key_weight(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy_dist.lora.calculate_weight(self.patches[key], temp_weight, key, device_mesh=self.device_mesh)
        if set_func is None:
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))

            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        if rank == 0:
            self.model_sd = self.model_state_dict()
        else:
            self.model_sd = None
        self.model.to("meta")

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = FSDPModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = FSDPModelPatcher
        self.__class__ = src_cls
        if src_cls != FSDPModelPatcher:
            n.size = 0
        return n
