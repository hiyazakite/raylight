# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import torch
import logging

import comfy.ops
import comfy.lora
import comfy.model_management
from .dequant import dequantize_tensor, is_quantized
import time

from raylight.comfy_dist.lora import calculate_weight as ray_calculate_weight


def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split("."):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True


# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x

        return noop

    from packaging import version
    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator  # torch too old
    else:
        # CRITICAL: Always use torch.compiler.disable for GGUF paths.
        # Even on newer versions of Torch (2.8+), Dynamo tracing into dequantize/patching
        # logic can trigger warnings about builtins on custom subclasses (e.g. Tensor.add_).
        return torch.compiler.disable


torch_compiler_disable = get_torch_compiler_disable_decorator()


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        if new is self:
            return self
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.size())
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        new = super().clone(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", None)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def detach(self, *args, **kwargs):
        new = super().detach(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", None)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy(),
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    dequant_cache = {}
    cache_patched_weights = False # Strictly optional, disabled by default
    torch_compatible_tensor_types = {
        None,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }

    @staticmethod
    def clear_dequant_cache():
        # Clear the global/class-level cache
        GGMLLayer.dequant_cache = {}
        # Also force a collection to prevent deferred deletion issues
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(
            f"{prefix}bias"
        )
        # CRITICAL: Always use pointer swapping (GGML load) if the incoming weight is a GGMLTensor.
        # This prevents standard load_state_dict from doing a dense copy_ for F16/F32 layers in GGUF.
        from .ops import GGMLTensor
        if isinstance(weight, GGMLTensor) or self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, torch.nn.Linear
        ):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)
        
        # Clear the local dequantization cache on load to avoid stale weights
        if hasattr(self, "dequant_cache") and self.dequant_cache is not GGMLLayer.dequant_cache:
             self.dequant_cache = {}

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + "weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = (
                self.dequant_dtype
                if self.dequant_dtype and self.dequant_dtype != "target"
                else torch.float16
            )
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return

    @torch_compiler_disable()
    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # Strip subclass to bypass __torch_function__ overhead on metadata access
        is_ggml = isinstance(tensor, GGMLTensor)
        native_tensor = tensor.as_subclass(torch.Tensor) if is_ggml else tensor
        
        # We must capture the logical shape defined by the GGUF metadata before stripping!
        target_shape = tensor.shape if is_ggml else native_tensor.shape

        # dequantization cache key (based on tensor data pointer and target dtype)
        ptr = native_tensor.data_ptr()
        cache_key = (ptr, dtype, target_shape)
        if self.cache_patched_weights and cache_key in self.dequant_cache:
            return self.dequant_cache[cache_key]
        
        # Periodic cache health check
        if self.cache_patched_weights:
            if not hasattr(GGMLLayer, "_last_stats_log"):
                GGMLLayer._last_stats_log = 0
            curr_time = time.time()
            if curr_time - GGMLLayer._last_stats_log > 10: # Log every 10 seconds
               GGMLLayer._last_stats_log = curr_time
               print(f"[GGUF DEBUG] Dequant Cache Size: {len(self.dequant_cache)}")
               if len(self.dequant_cache) > 200:
                   print(f"[GGUF WARNING] Cache size suspiciously large! ptr={ptr}")

        # consolidate and load patches to GPU in async
        patch_list = []
        device = native_tensor.device
        key = None
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        # Pass the GGMLTensor (tensor) because dequantize_tensor needs the .tensor_shape/.tensor_type metadata
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # dequantize_tensor should already strip GGMLTensor, but ensure safety
        if isinstance(weight, GGMLTensor):
            weight = weight.as_subclass(torch.Tensor)
            
        # CRITICAL FIX: The dequantized weight might have the shape of the raw 1D tensor buffer
        # if qtype was None (TORCH_COMPATIBLE_QTYPES branch). We must reshape to the logical size.
        if weight.shape != target_shape:
            weight = weight.view(target_shape)

        # apply patches
        if len(patch_list) > 0:
            # Log first 3 times per layer to confirm LoRA is being applied
            if not hasattr(self, "_lora_log_count"):
                self._lora_log_count = 0
            if self._lora_log_count < 3:
                print(f"[GGUF LoRA] Applying {len(patch_list)} patches to key={key}")
                self._lora_log_count += 1
            
            if self.patch_dtype is None:
                weight = ray_calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = (
                    dtype if self.patch_dtype == "target" else self.patch_dtype
                )
                weight = ray_calculate_weight(
                    patch_list, weight, key, patch_dtype
                )
            
            # Cache the patched weight if enabled
            if self.cache_patched_weights:
                self.dequant_cache[cache_key] = weight

        # CRITICAL FIX: Cache the weight even if there were no patches!
        elif self.cache_patched_weights:
            self.dequant_cache[cache_key] = weight

        return weight.to(device)

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        weight = None
        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)

        def match_device(t, dev):
            if t is None or dev is None:
                return False
            return t.device.type == dev.type and (t.device.index or 0) == (dev.index or 0)

        if s.bias is not None:
            # Fast device check
            b_native = s.bias.as_subclass(torch.Tensor) if isinstance(s.bias, GGMLTensor) else s.bias
            b_tensor = s.bias
            if not match_device(b_native, device):
                b_tensor = b_tensor.to(device, non_blocking=non_blocking)
            
            bias = s.get_weight(b_tensor, dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        if s.weight is not None:
            # Fast device check
            w_native = s.weight.as_subclass(torch.Tensor) if isinstance(s.weight, GGMLTensor) else s.weight
            w_tensor = s.weight
            if not match_device(w_native, device):
                w_tensor = w_tensor.to(device, non_blocking=non_blocking)

            weight = s.get_weight(w_tensor, dtype)
            weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)

        return weight, bias

    @torch_compiler_disable()
    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = out.as_subclass(torch.Tensor)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError


class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """

    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(
            self, in_features, out_features, bias=True, device=None, dtype=None
        ):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            expected_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
            if weight.shape != expected_shape:
                weight = weight.view(expected_shape)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(
                self, device=input.device, dtype=out_dtype
            )
            expected_shape = (self.num_embeddings, self.embedding_dim)
            if weight.shape != expected_shape:
                weight = weight.view(expected_shape)
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )


def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item
