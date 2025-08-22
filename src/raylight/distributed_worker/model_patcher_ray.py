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
import copy
import inspect
import logging
import math
import uuid
from typing import Callable, Optional

import torch

import comfy.float
import comfy.hooks
import comfy.lora
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
from comfy.comfy_types import UnetWrapperFunction
from comfy.patcher_extension import CallbacksMP, PatcherInjection, WrappersMP


def move_weight_functions(m, device):
    if device is None:
        return 0

    memory = 0
    if hasattr(m, "weight_function"):
        for f in m.weight_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)

    if hasattr(m, "bias_function"):
        for f in m.bias_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)
    return memory


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


def string_to_seed(data):
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:  # intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return comfy.float.stochastic_rounding(comfy.lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))

        return comfy.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
    with self.use_ejected():
        self.unpatch_hooks()
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        loading = self._load_list()

        load_completely = []
        loading.sort(reverse=True)
        for x in loading:
            n = x[1]
            m = x[2]
            params = x[3]
            module_mem = x[0]

            lowvram_weight = False

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if not full_load and hasattr(m, "comfy_cast_weights"):
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True
                    lowvram_counter += 1
                    if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                        continue

            cast_weight = self.force_cast_weights
            if lowvram_weight:
                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []

                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = [LowVramPatch(weight_key, self.patches)]
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = [LowVramPatch(bias_key, self.patches)]
                        patch_counter += 1

                cast_weight = True
            else:
                if hasattr(m, "comfy_cast_weights"):
                    wipe_lowvram_weight(m)

                if full_load or mem_counter + module_mem < lowvram_model_memory:
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m, params))

            if cast_weight and hasattr(m, "comfy_cast_weights"):
                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True

            if weight_key in self.weight_wrapper_patches:
                m.weight_function.extend(self.weight_wrapper_patches[weight_key])

            if bias_key in self.weight_wrapper_patches:
                m.bias_function.extend(self.weight_wrapper_patches[bias_key])

            mem_counter += move_weight_functions(m, device_to)

        load_completely.sort(reverse=True)
        for x in load_completely:
            n = x[1]
            m = x[2]
            params = x[3]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights is True:
                    continue

            for param in params:
                self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

            logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
            m.comfy_patched_weights = True

        for x in load_completely:
            x[2].to(device_to)

        if lowvram_counter > 0:
            logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
            self.model.model_lowvram = True
        else:
            logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

        self.model.lowvram_patch_counter += patch_counter
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter
        self.model.current_weight_patches_uuid = self.patches_uuid

        for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
            callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

        self.apply_hooks(self.forced_hooks, force_apply=True)
