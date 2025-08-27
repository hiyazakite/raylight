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
