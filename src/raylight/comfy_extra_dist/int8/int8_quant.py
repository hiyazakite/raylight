"""
INT8 tensorwise quantization core: quantization utilities, forward paths,
LoRA adapters, and the Int8TensorwiseOps custom-operations class.

Provides fast W8A8 inference via torch._int_mm with optional Triton fast-path.
"""

import logging
import torch
from torch import Tensor, nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton fast-path (optional – falls back to torch._int_mm)
# ---------------------------------------------------------------------------
try:
    from .int8_fused_kernel import (
        triton_int8_linear,
        triton_int8_linear_per_row,
        triton_quantize_rowwise,
    )
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    logger.info("[Raylight INT8] Triton not found, falling back to torch._int_mm")

# ---------------------------------------------------------------------------
# Quantization Utilities
# ---------------------------------------------------------------------------

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    return q.float() * scale


def stochastic_round_int8_delta(
    x: Tensor, scale: float | Tensor, seed: int = 0,
) -> Tensor:
    """Quantize a delta tensor to INT8 using stochastic rounding (for LoRA)."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    if isinstance(scale, torch.Tensor):
        scale = scale.to(x.device)
    x_scaled = x / scale

    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor

    random_vals = torch.rand(
        x_scaled.shape, generator=generator, device=x.device, dtype=x_scaled.dtype,
    )
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)


# ---------------------------------------------------------------------------
# Forward Paths (W8A8)
# ---------------------------------------------------------------------------

@torch.no_grad()
@torch.compiler.disable
def int8_forward_dynamic(
    x: Tensor,
    weight: Tensor,
    weight_scale: float | Tensor,
    bias: Tensor | None,
    compute_dtype: torch.dtype,
) -> Tensor:
    """Forward with dynamic per-token activation quantization (tensorwise weight)."""
    # Fast path: triton fused kernel
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)

    # Slow path: torch._int_mm
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


@torch.no_grad()
@torch.compiler.disable
def int8_forward_dynamic_per_row(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None,
    compute_dtype: torch.dtype,
) -> Tensor:
    """Forward with per-row weight scales + dynamic per-token activation quantization."""
    if _TRITON_AVAILABLE and x.is_cuda:
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype)

    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


# ---------------------------------------------------------------------------
# INT8 LoRA Adapters
# ---------------------------------------------------------------------------
try:
    from comfy.weight_adapter.lora import LoRAAdapter
    _LORA_ADAPTER_AVAILABLE = True
except ImportError:
    _LORA_ADAPTER_AVAILABLE = False


def _get_effective_scale(weight_scale, delta_f, offset):
    """Slice weight_scale for merged QKV LoRAs or mismatched shapes."""
    ws = weight_scale
    if not isinstance(ws, torch.Tensor) or ws.numel() == 1:
        return ws
    if offset is not None and ws.shape[0] != delta_f.shape[0]:
        dim, start, size = offset
        if dim == 0:
            return ws.narrow(0, start, size)
    if ws.shape[0] != delta_f.shape[0] and ws.shape[0] % delta_f.shape[0] == 0:
        return ws[: delta_f.shape[0]]
    return ws


if _LORA_ADAPTER_AVAILABLE:
    class INT8LoRAPatchAdapter(LoRAAdapter):
        """LoRA adapter that patches INT8 weights in-place using stochastic rounding."""

        def __init__(self, loaded_keys, weights, weight_scale, seed=0):
            super().__init__(loaded_keys, weights)
            self.weight_scale = weight_scale
            self.seed = seed

        def calculate_weight(
            self, weight, key, strength, strength_model, offset, function,
            intermediate_dtype=torch.float32, original_weight=None,
        ):
            v = self.weights
            up, down, alpha = v[0], v[1], v[2]

            rank = down.shape[0] if down.ndim >= 2 else 1
            scale = (alpha / rank) * strength if alpha is not None else strength

            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device

            up_f = up.to(comp_device, dtype=intermediate_dtype)
            down_f = down.to(comp_device, dtype=intermediate_dtype)

            if v[3] is not None:
                mid_f = v[3].to(comp_device, dtype=intermediate_dtype)
                lora_diff = torch.mm(
                    up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1)),
                ).reshape(weight.shape)
            else:
                lora_diff = torch.mm(up_f.flatten(1), down_f.flatten(1)).reshape(weight.shape)

            if weight.dtype == torch.int8:
                delta_f = lora_diff * scale
                eff_scale = _get_effective_scale(self.weight_scale, delta_f, offset)
                delta_int8 = stochastic_round_int8_delta(delta_f, eff_scale, self.seed)
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                return weight + (lora_diff * scale).to(weight.device, weight.dtype)

    class INT8MergedLoRAPatchAdapter(LoRAAdapter):
        """
        Merges multiple LoRAs in float space before a single stochastic
        rounding step – more precise for LoRA stacks.
        """

        def __init__(self, patches, weight_scale, seed=0):
            first_adapter = patches[0][0]
            super().__init__(first_adapter.loaded_keys, first_adapter.weights)
            self.patches = patches
            self.weight_scale = weight_scale
            self.seed = seed

        def calculate_weight(
            self, weight, key, strength, strength_model, offset, function,
            intermediate_dtype=torch.float32, original_weight=None,
        ):
            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device

            total_delta_f = None

            for adapter, lora_strength in self.patches:
                v = adapter.weights
                up, down, alpha = v[0], v[1], v[2]
                rank = down.shape[0] if down.ndim >= 2 else 1
                s = (alpha / rank) * lora_strength if alpha is not None else lora_strength

                up_f = up.to(comp_device, dtype=intermediate_dtype)
                down_f = down.to(comp_device, dtype=intermediate_dtype)

                if v[3] is not None:
                    mid_f = v[3].to(comp_device, dtype=intermediate_dtype)
                    delta = torch.mm(
                        up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1)),
                    ).reshape(weight.shape)
                else:
                    delta = torch.mm(up_f.flatten(1), down_f.flatten(1)).reshape(weight.shape)

                if total_delta_f is None:
                    total_delta_f = delta * s
                else:
                    total_delta_f += delta * s

            if total_delta_f is None:
                return weight

            if weight.dtype == torch.int8:
                eff_scale = _get_effective_scale(self.weight_scale, total_delta_f, offset)
                delta_int8 = stochastic_round_int8_delta(total_delta_f, eff_scale, self.seed)
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                return weight + total_delta_f.to(device, weight.dtype)
else:
    # Stubs so downstream imports don't crash when comfy.weight_adapter is absent
    INT8LoRAPatchAdapter = None  # type: ignore[assignment,misc]
    INT8MergedLoRAPatchAdapter = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Dynamic LoRA Synchronization Hook
# ---------------------------------------------------------------------------

class DynamicLoRAHook:
    """
    Forward pre-hook on the diffusion model that synchronizes dynamic LoRA
    attributes with the current ``ModelPatcher`` context at each forward pass.
    """

    def __init__(self):
        self.current_lora_id = None

    def pre_forward(self, module, input_args, input_kwargs):
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            context = input_args[2] if len(input_args) > 2 else None
            if isinstance(context, dict) and "transformer_options" in context:
                transformer_options = context["transformer_options"]

        dynamic_loras = transformer_options.get("dynamic_loras", [])

        lora_id = (
            hash(tuple((id(d["patches"]), d["strength"]) for d in dynamic_loras))
            if dynamic_loras
            else None
        )
        if lora_id == self.current_lora_id:
            return None

        self._apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    # ------------------------------------------------------------------
    def _apply_composition(self, diffusion_model, dynamic_loras):
        layer_patches: dict = {}
        for entry in dynamic_loras:
            strength = entry["strength"]
            for key, adapter in entry["patches"].items():
                layer_patches.setdefault(key, []).append((adapter, strength))

        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue

            possible_keys = [f"diffusion_model.{name}.weight", f"{name}.weight"]
            patches = None
            for pk in possible_keys:
                if pk in layer_patches:
                    patches = layer_patches[pk]
                    break

            if not patches:
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            all_A, all_B = [], []
            for adapter, strength in patches:
                v = adapter.weights
                up, down, alpha, mid = v[0], v[1], v[2], v[3]
                rank = down.shape[0] if down.ndim >= 2 else 1
                s = (alpha / rank) * strength if alpha is not None else strength
                curr_A = down
                if mid is not None:
                    curr_A = torch.mm(mid.flatten(1), down.flatten(1)).reshape(down.shape)
                all_A.append(curr_A * s)
                all_B.append(up)

            if all_A:
                device = getattr(module, "weight", torch.tensor(0)).device
                module.lora_A = torch.cat(all_A, dim=0).to(device)
                module.lora_B = torch.cat(all_B, dim=1).to(device)
                module.lora_alpha = None
            else:
                module.lora_A = None
                module.lora_B = None

    @classmethod
    def register(cls, diffusion_model):
        if not hasattr(diffusion_model, "_dynamic_lora_hook"):
            hook = cls()
            diffusion_model._dynamic_lora_hook = hook
            diffusion_model.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
        return diffusion_model._dynamic_lora_hook


# =============================================================================
# Int8TensorwiseOps – ComfyUI Custom Operations
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False

if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """Custom ComfyUI operations for INT8 tensorwise quantization."""

        excluded_names: list[str] = []
        dynamic_quantize: bool = False
        _is_prequantized: bool = False

        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self._is_quantized = False
                self._is_per_row = False
                self.compute_dtype = torch.bfloat16
                self.lora_A = None
                self.lora_B = None
                self.lora_alpha = None

            def reset_parameters(self):
                return None

            # -----------------------------------------------------------
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
                _ = state_dict.pop(input_scale_key, None)  # unused

                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Pre-quantized checkpoint
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        Int8TensorwiseOps._is_prequantized = True

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
                        # ---- Handle any non-INT8 dtype (FP16, BF16, FP32, FP8, …) ----
                        # FP8 dtypes can't be used directly; cast to the
                        # target precision first so the rest of the pipeline
                        # (exclusion check, on-the-fly quant, or plain load)
                        # works identically for every source format.
                        _CASTABLE = (
                            torch.float16, torch.bfloat16, torch.float32,
                        )
                        _FP8 = set()
                        for _name in ("float8_e4m3fn", "float8_e5m2",
                                      "float8_e4m3fnuz", "float8_e5m2fnuz"):
                            _dt = getattr(torch, _name, None)
                            if _dt is not None:
                                _FP8.add(_dt)

                        if weight_tensor.dtype in _FP8:
                            # Cast FP8 → BF16 (lossless enough for the
                            # subsequent INT8 quantization step).
                            weight_tensor = weight_tensor.to(torch.bfloat16)

                        is_excluded = any(
                            ex in prefix for ex in Int8TensorwiseOps.excluded_names
                        )
                        is_dim1 = (
                            self.in_features == 1
                            or self.out_features == 1
                            or weight_tensor.ndim == 1
                        )

                        if is_excluded or is_dim1 or not Int8TensorwiseOps.dynamic_quantize:
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
                            # Unknown / exotic dtype – load as-is without quantization
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)

                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None

            # -----------------------------------------------------------
            def convert_weight(self, _weight, inplace=False):
                if not self._is_quantized:
                    return _weight
                return self.weight

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

            # -----------------------------------------------------------
            def forward(self, x: Tensor) -> Tensor:
                if not self._is_quantized:
                    weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                    out = F.linear(x, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    return out

                weight = self.weight.to(x.device, non_blocking=True)
                bias = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None

                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])

                if x_2d.shape[0] > 16:
                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    # Small-batch fallback: dequantize to float
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

        # Pass-through wrappers for non-linear layers
        class GroupNorm(manual_cast.GroupNorm):
            pass

        class LayerNorm(manual_cast.LayerNorm):
            pass

        class Conv2d(manual_cast.Conv2d):
            pass

        class Conv3d(manual_cast.Conv3d):
            pass

        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass

        class Embedding(manual_cast.Embedding):
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
    Int8TensorwiseOps = None  # type: ignore[assignment,misc]
