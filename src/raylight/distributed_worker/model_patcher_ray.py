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

# This file only contains swapped function from ComfyUI model_patcher.py into raylight equivalent

from __future__ import annotations

import collections
import logging

import torch
from torch.distributed.tensor import DTensor

import comfy
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import (move_weight_functions,
                                 get_key_weight,
                                 string_to_seed)


import comfy.weight_adapter as weight_adapter
from comfy.lora import pad_tensor_to_shape
from ..wan.distributed.fsdp import inspect_tensor
import sys, traceback

def force_traceback(exctype, value, tb):
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = force_traceback


def calculate_weight(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None, device_mesh=None):
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (calculate_weight(v[1:], v[0][1](comfy.model_management.cast_to_device(v[0][0], weight.device, intermediate_dtype, copy=True), inplace=True), key, intermediate_dtype=intermediate_dtype, device_mesh=device_mesh),)

        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(weight, key, strength, strength_model, offset, function, intermediate_dtype, original_weights, device_mesh)
            if output is None:
                logging.warning("Calculate Weight Failed: {} {}".format(v.name, key))
            else:
                weight = output
                if old_weight is not None:
                    weight = old_weight
            continue

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            # An extra flag to pad the weight if the diff's shape is larger than the weight
            do_pad_weight = len(v) > 1 and v[1]['pad_weight']
            if do_pad_weight and diff.shape != weight.shape:
                logging.info("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    logging.warning("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    if isinstance(weight, DTensor):
                        weight += DTensor.from_local(function(strength * comfy.model_management.cast_to_device(diff, weight.device, weight.dtype)), device_mesh)
                    else:
                        weight += function(strength * comfy.model_management.cast_to_device(diff, weight.device, weight.dtype))
        elif patch_type == "set":
            weight.copy_(v[0])
        elif patch_type == "model_as_lora":
            target_weight: torch.Tensor = v[0]
            diff_weight = comfy.model_management.cast_to_device(target_weight, weight.device, intermediate_dtype) - \
                          comfy.model_management.cast_to_device(original_weights[key][0][0], weight.device, intermediate_dtype)

            if isinstance(weight, DTensor):
                weight += DTensor.from_local(function(strength * comfy.model_management.cast_to_device(diff_weight, weight.device, weight.dtype)), device_mesh)
            else:
                weight += function(strength * comfy.model_management.cast_to_device(diff_weight, weight.device, weight.dtype))
        else:
            logging.warning("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    return weight


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

    out_weight = calculate_weight(self.patches[key], temp_weight, key, device_mesh=device_mesh)
    if set_func is None:
        out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    else:
        set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))


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


# Currently unsused
def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
    """
    Load all model modules to CPU.
    Designed for later FSDP wrapping.
    """
    with self.use_ejected():
        self.unpatch_hooks()
        cuda_device = device_to
        device_to = "cpu"
        mem_counter = 0
        loading = self._load_list()  # list of modules to load

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

        self.model.device = 'cpu'
        self.model.model_loaded_weight_memory = mem_counter
        self.model.current_weight_patches_uuid = self.patches_uuid

        for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
            callback(self, cuda_device, lowvram_model_memory, force_patch_weights, full_load)

        self.apply_hooks(self.forced_hooks, force_apply=True)
