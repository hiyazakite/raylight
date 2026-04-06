"""
INT8 + SafetensorLayer hybrid ops for distributed (Ray) loading.

Combines zero-copy mmap assignment from SafetensorOps with INT8 tensorwise
quantization from Int8TensorwiseOps.  Used automatically by LazyTensorContext
and FSDPContext when ``model_options`` contains ``int8_quantize=True``.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import comfy.ops
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    from raylight.expansion.comfyui_lazytensors.ops import SafetensorLayer, SafetensorOps
    from .int8_quant import (
        Int8TensorwiseOps,
        quantize_int8_tensorwise,
        dequantize,
        stochastic_round_int8_delta,
        int8_forward_dynamic,
        int8_forward_dynamic_per_row,
    )
    _AVAILABLE = True
except ImportError as e:
    _AVAILABLE = False
    logger.warning("[Raylight INT8] Cannot build Int8SafetensorOps: %s", e)


if _AVAILABLE and Int8TensorwiseOps is not None:

    class Int8SafetensorOps(SafetensorOps):
        """
        Drop-in replacement for ``SafetensorOps`` that adds INT8 quantization.

        Zero-copy mmap behaviour (``SafetensorLayer``) is preserved for all layer
        types.  Only ``Linear`` gains INT8 awareness: pre-quantized INT8 weights
        are loaded natively; float weights may be quantized on-the-fly.

        Class-level flags mirror ``Int8TensorwiseOps`` and should be configured
        before the model is loaded:

        * ``excluded_names``  – layer-name fragments to keep in float.
        * ``dynamic_quantize`` – if *True*, quantize float weights at load time.
        * ``model_type``       – architecture hint used for exclusion list.
        """

        excluded_names: list[str] = []
        dynamic_quantize: bool = False
        _is_prequantized: bool = False

        class Linear(SafetensorLayer, manual_cast.Linear):
            """Linear layer with zero-copy load **and** INT8 W8A8 inference."""

            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                # Mirror SafetensorOps.Linear – lightweight, no alloc
                nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                self.bias = None

                # INT8-specific state
                self.weight_scale = None
                self._is_quantized = False
                self._is_per_row = False
                self.compute_dtype = torch.bfloat16
                self.lora_A = None
                self.lora_B = None
                self.lora_alpha = None

            def reset_parameters(self):
                return None

            # ---------------------------------------------------------
            # State-dict loading: INT8-aware, zero-copy compatible
            # ---------------------------------------------------------
            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"

                weight_scale = state_dict.pop(scale_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                weight_tensor = state_dict.pop(weight_key, None)
                _ = state_dict.pop(input_scale_key, None)

                if weight_tensor is not None:
                    # Meta/lazy tensors (used by streaming TP Phase 2 to
                    # establish architecture shapes) have no data.  Just
                    # assign them as-is — Phase 4 will stream real values.
                    if getattr(weight_tensor, "is_meta", False):
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        if weight_scale is not None:
                            self.weight_scale = weight_scale
                        bias_tensor = state_dict.pop(bias_key, None)
                        if bias_tensor is not None:
                            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                        return

                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # ---- Pre-quantized INT8 checkpoint ----
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        Int8SafetensorOps._is_prequantized = True

                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                self.weight_scale = weight_scale.float().item()
                                self._is_per_row = False
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.weight_scale = weight_scale.float()
                                self._is_per_row = True
                            else:
                                self.weight_scale = weight_scale.float()
                                self._is_per_row = False
                        else:
                            self.weight_scale = float(weight_scale)
                            self._is_per_row = False

                    else:
                        # ---- Non-INT8 dtype (FP16/BF16/FP32/FP8/…) ----
                        _CASTABLE = (torch.float16, torch.bfloat16, torch.float32)
                        _FP8 = set()
                        for _name in ("float8_e4m3fn", "float8_e5m2",
                                      "float8_e4m3fnuz", "float8_e5m2fnuz"):
                            _dt = getattr(torch, _name, None)
                            if _dt is not None:
                                _FP8.add(_dt)

                        if weight_tensor.dtype in _FP8:
                            weight_tensor = weight_tensor.to(torch.bfloat16)

                        is_excluded = any(
                            ex in prefix
                            for ex in Int8SafetensorOps.excluded_names
                        )
                        is_dim1 = (
                            self.in_features == 1
                            or self.out_features == 1
                            or weight_tensor.ndim == 1
                        )

                        if is_excluded or is_dim1 or not Int8SafetensorOps.dynamic_quantize:
                            # Keep as float – use zero-copy assignment
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        elif weight_tensor.dtype in _CASTABLE:
                            device = (
                                torch.device("cuda")
                                if torch.cuda.is_available()
                                else weight_tensor.device
                            )
                            w_gpu = weight_tensor.to(device, non_blocking=True)
                            q_weight, q_scale = quantize_int8_tensorwise(w_gpu)

                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.weight_scale = (
                                q_scale.cpu()
                                if isinstance(q_scale, torch.Tensor)
                                else q_scale
                            )
                            self._is_quantized = True
                            self._is_per_row = False
                        else:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)

                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

            # ---------------------------------------------------------
            # ComfyUI weight patching hooks (LoRA)
            # ---------------------------------------------------------
            def convert_weight(self, _weight, inplace=False):
                """Dequantize INT8 weight to float for LoRA patching math.

                ``patch_weight_to_device`` passes ``_weight`` as a copy of
                ``self.weight`` already cast to compute-dtype on the target
                device.  For an INT8 parameter that means the int8 integer
                values (−128 … 127) represented in bfloat16/float32.
                Multiplying by the per-tensor (or per-row) scale recovers the
                original floating-point weight so that
                ``comfy.lora.calculate_weight`` can apply LoRA deltas in
                float space.  ``set_weight`` will re-quantise the result.
                """
                if not self._is_quantized:
                    return _weight
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    w_scale = w_scale.to(_weight.device)
                if inplace:
                    return _weight.mul_(w_scale)
                return _weight * w_scale

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight
                    if inplace_update:
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight
                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                new_weight = stochastic_round_int8_delta(out_weight, self.weight_scale, seed)
                if return_weight:
                    return new_weight
                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None:
                    return None
                if return_weight:
                    return out_bias
                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(out_bias)
                else:
                    self.bias = nn.Parameter(out_bias, requires_grad=False)

            # ---------------------------------------------------------
            # Forward: INT8 fast-path or SafetensorLayer fallback
            # ---------------------------------------------------------
            def forward(self, x):
                if not self._is_quantized:
                    # Delegate to SafetensorLayer path (handles lazy + patches)
                    weight, bias = self.cast_bias_weight(x)
                    return F.linear(x, weight, bias)

                # ---- INT8 fast-path ----
                weight = self.weight.to(x.device, non_blocking=True)
                bias = (
                    self.bias.to(x.device, non_blocking=True)
                    if self.bias is not None
                    else None
                )

                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = (
                    x.dtype
                    if x.dtype in (torch.float16, torch.bfloat16)
                    else torch.bfloat16
                )

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])

                if x_2d.shape[0] > 16:
                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(
                            x_2d, weight, w_scale, bias, compute_dtype,
                        )
                    else:
                        y = int8_forward_dynamic(
                            x_2d, weight, w_scale, bias, compute_dtype,
                        )
                else:
                    # Small-batch fallback
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_typed = bias.to(x.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_typed)

                # Dynamic LoRA path
                if self.lora_A is not None and self.lora_B is not None:
                    lA = self.lora_A.to(x.device, non_blocking=True)
                    lB = self.lora_B.to(x.device, non_blocking=True)
                    lora_x = F.linear(x_2d.to(lA.dtype), lA)
                    lora_y = F.linear(lora_x, lB)
                    if self.lora_alpha is not None:
                        lora_y = lora_y * self.lora_alpha
                    y = y + lora_y.to(y.dtype)

                return y.reshape(*x_shape[:-1], y.shape[-1])

        # ---- Non-linear layers: inherit zero-copy from SafetensorOps ----
        class Conv2d(SafetensorOps.Conv2d):
            pass

        class Conv3d(SafetensorLayer, manual_cast.Conv3d):
            pass

        class ConvTranspose2d(SafetensorLayer, manual_cast.ConvTranspose2d):
            pass

        class GroupNorm(SafetensorLayer, manual_cast.GroupNorm):
            pass

        class LayerNorm(SafetensorOps.LayerNorm):
            pass

        class Embedding(SafetensorOps.Embedding):
            pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            else:
                raise ValueError(f"unsupported dimensions: {dims}")

else:
    Int8SafetensorOps = None  # type: ignore[assignment,misc]
