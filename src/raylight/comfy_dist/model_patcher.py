from __future__ import annotations

import collections

import torch

import comfy
from comfy.model_patcher import (get_key_weight,
                                 string_to_seed,
                                 move_weight_functions)

from raylight import comfy_dist


class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:   # intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return comfy.float.stochastic_rounding(comfy.lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))

        return comfy_dist.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


class FSDPModelPatcher(comfy.model_patcher.ModelPatcher):

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
            out_weight = comfy_dist.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key), device_mesh=self.device_mesh)

            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        self.model.to("meta")

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = FSDPModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = FSDPModelPatcher
        self.__class__ = src_cls

        # inject fsdp-specific defaults if missing
        if not hasattr(n, "device_mesh"):
            n.device_mesh = None
        if not hasattr(n, "rank"):
            n.rank = None

        if src_cls != FSDPModelPatcher:
            n.size = 0
        return n
