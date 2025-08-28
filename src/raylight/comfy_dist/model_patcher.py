"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


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

from raylight import comfy_dist
from comfy.patcher_extension import CallbacksMP


def patch_weight_to_device(self, key, device_to=None, inplace_update=False, convert_dtensor=False, device_mesh=None):
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

    out_weight = comfy_dist.lora.calculate_weight(self.patches[key], temp_weight, key, device_mesh=device_mesh)
    if set_func is None:
        out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    else:
        set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))


def ray_patch_fsdp(self, model_state_dict, device_to=None):
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


def load_fsdp(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, rank=None):
    with self.use_ejected():
        cuda_device = device_to
        if rank == 0:
            self.unpatch_hooks()
            device_to = "cpu"
            mem_counter = 0
            loading = self._load_list()

            load_modules = []
            loading.sort(reverse=True)

            for module_mem, name, module, params in loading:
                weight_key = f"{name}.weight"
                bias_key = f"{name}.bias"

                if hasattr(module, "comfy_cast_weights"):
                    module.prev_comfy_cast_weights = module.comfy_cast_weights
                    module.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    module.weight_function.extend(self.weight_wrapper_patches[weight_key])
                if bias_key in self.weight_wrapper_patches:
                    module.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(module, device_to)
                load_modules.append((module_mem, name, module, params))

            for _, name, module, params in load_modules:
                for param in params:
                    self.patch_weight_to_device(f"{name}.{param}", device_to)
                module.comfy_patched_weights = True

            for _, _, module, _ in load_modules:
                module.to('cpu')

            logging.info(f"Loaded completely on CPU: mem={mem_counter / (1024*1024):.2f} MB")

        if rank == 0:
            model_state_dict = self.model_state_dict()
            self.model.to("meta")
        self.patch_fsdp(device_to, model_state_dict)
        self.model.device = cuda_device
        self.model.model_loaded_weight_memory = mem_counter
        self.model.current_weight_patches_uuid = self.patches_uuid

        for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
            callback(self, cuda_device, lowvram_model_memory, force_patch_weights, full_load)

        self.apply_hooks(self.forced_hooks, force_apply=True)

def make_ray_load_fsdp(rank):
    # Factory function to wrap load_fsdp so it can be injected with rank
    def _ray_load_fsdp(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        return load_fsdp(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            rank=rank
        )
    return _ray_load_fsdp

def make_ray_patch_weight_to_device(convert_dtensor=False, device_mesh=None):
    # Factory function to wrap ray_patch_weight_to_device with custom args.

    def _ray_patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        return patch_weight_to_device(
            self,
            key,
            device_to=device_to,
            inplace_update=inplace_update,
            convert_dtensor=convert_dtensor,
            device_mesh=device_mesh
        )
    return _ray_patch_weight_to_device
