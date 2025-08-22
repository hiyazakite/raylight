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

import logging


from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import move_weight_functions


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
