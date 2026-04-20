"""Fp8AmpereLinear — FP8 weight × BF16 activation linear layer for Ampere GPUs.

This module replaces ``torch.nn.Linear`` when operating with FP8 weights on
Ampere GPUs (SM ≥ 8.0).  Key properties:

  comfy_cast_weights = False
    Tells ComfyUI's ``RaylightModelPatcher`` / ``LowVramPatch`` NOT to attach a
    weight-upcast function.  Without this flag ComfyUI would create a full BF16
    transient copy of the weight on every forward pass, eliminating the VRAM
    savings and compute benefits of keeping weights in FP8.

  Forward path (on Ampere)
    1. Lazily repack weights to Marlin layout (once, cached).
    2. Call ``fp8_ampere_mm`` which runs the Marlin-architecture kernel:
         • Dequants FP8 → BF16 via bitwise ops in registers.
         • Runs m16n8k16 BF16 MMA on tensor cores.
         • Writes BF16 output.
    3. Optionally add bias (pure BF16, tiny cost).

  Fallback path (non-Ampere or kernel not built)
    Falls back to torch.nn.functional.linear after decoding weights to BF16.
    Allows the module to be instantiated and evaluated on any device without a
    hard build dependency.

Weight layout
    weight      : Tensor[N, K]  uint8    (FP8 E4M3 bit-patterns, unpacked)
    weight_scale: Tensor[N]     bfloat16 (per-channel scale, 1.0 for standard FP8)

    N = out_features, K = in_features.

Factory methods
    Fp8AmpereLinear.from_fp8_weight(fp8_weight, weight_scale, bias)
        Build directly from a pre-packed FP8 tensor + scale.

    Fp8AmpereLinear.from_linear(linear)
        Quantise an existing nn.Linear to FP8 in-place.

    Fp8AmpereLinear.from_fp8_checkpoint(fp8_weight_param, bias)
        Accept a float8_e4m3fn parameter (as loaded from a safetensors checkpoint)
        and compute the per-channel scale on-the-fly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import compute_fp8_scales, quantize_to_fp8_e4m3, repack_fp8_for_marlin

__all__ = ["Fp8AmpereLinear"]

# ---------------------------------------------------------------------------
# Lazy extension import (build-optional)
# ---------------------------------------------------------------------------

def _try_load_extension():
    """Return the C extension module, or None if not built / not on Ampere."""
    try:
        from raylight.distributed_modules.fp8_ampere import _C_fp8ampere  # type: ignore[import]
        return _C_fp8ampere
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Runtime Ampere check
# ---------------------------------------------------------------------------

def _is_ampere_or_newer(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device)
    return props.major >= 8  # SM 8.0 = Ampere A100/A10/RTX 30xx


# ---------------------------------------------------------------------------
# Fp8AmpereLinear
# ---------------------------------------------------------------------------

class Fp8AmpereLinear(nn.Module):
    """Linear layer with FP8 weights using BF16 tensor cores on Ampere GPUs.

    See module docstring for full description.
    """

    comfy_cast_weights: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.bfloat16),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.bfloat16),
            )
        else:
            self.register_buffer("bias", None)

        self._ext = None  # loaded lazily on first forward call
        # Cached Marlin-packed weight + padded scale (computed lazily).
        self._weight_packed: torch.Tensor | None = None
        self._scale_padded: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_fp8_weight(
        cls,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> "Fp8AmpereLinear":
        """Construct from a pre-packed FP8 weight + per-channel scale.

        Args:
            weight_fp8  : Tensor[N, K] uint8 (FP8 E4M3 bit-patterns).
            weight_scale: Tensor[N]    bfloat16.
            bias        : Tensor[N]    (any float dtype), optional.
        """
        out_features, in_features = weight_fp8.shape
        obj = cls(in_features, out_features, bias=bias is not None)
        obj.weight.copy_(weight_fp8)
        obj.weight_scale.copy_(weight_scale.to(torch.bfloat16))
        if bias is not None:
            obj.bias = bias.to(dtype=torch.bfloat16).detach().clone()
        return obj

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Fp8AmpereLinear":
        """Quantise a standard ``nn.Linear`` to FP8 and wrap it."""
        w = linear.weight.data  # [N, K]
        weight_fp8, weight_scale = quantize_to_fp8_e4m3(
            w.to(dtype=torch.bfloat16) if w.dtype != torch.bfloat16 else w
        )
        bias = linear.bias.data if linear.bias is not None else None
        return cls.from_fp8_weight(weight_fp8, weight_scale, bias)

    @classmethod
    def from_fp8_checkpoint(
        cls,
        weight_param: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> "Fp8AmpereLinear":
        """Build from a ``float8_e4m3fn`` parameter as loaded from a checkpoint.

        Args:
            weight_param: Tensor[N, K] float8_e4m3fn or uint8.
            bias        : optional bias tensor.
        """
        if weight_param.dtype == torch.float8_e4m3fn:
            weight_u8 = weight_param.view(torch.uint8)
        elif weight_param.dtype == torch.uint8:
            weight_u8 = weight_param
        else:
            raise TypeError(
                f"from_fp8_checkpoint expects float8_e4m3fn or uint8, "
                f"got {weight_param.dtype}"
            )
        weight_scale = compute_fp8_scales(weight_u8)
        return cls.from_fp8_weight(weight_u8, weight_scale, bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _load_ext(self) -> object | None:
        if self._ext is None:
            self._ext = _try_load_extension()
        return self._ext

    def _prepare_packed(self):
        """Lazily repack weight to Marlin format with alignment padding."""
        if self._weight_packed is not None:
            if self._weight_packed.device == self.weight.device:
                return
            # Device changed (e.g. .to(device)); invalidate cache.
            self._weight_packed = None
            self._scale_padded = None

        packed, scale_padded, _, _ = repack_fp8_for_marlin(
            self.weight, self.weight_scale,
        )
        self._weight_packed = packed
        self._scale_padded = scale_padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer.

        On Ampere GPUs with the native extension built this uses BF16 tensor
        cores via the Marlin-architecture kernel.  Otherwise it falls back to
        BF16 matmul after decoding the FP8 weights.
        """
        ext = self._load_ext()
        device = x.device

        if ext is not None and _is_ampere_or_newer(device):
            return self._forward_marlin(x, ext)
        else:
            return self._forward_fallback(x)

    def _forward_marlin(self, x: torch.Tensor, ext) -> torch.Tensor:
        """BF16 tensor core path with FP8 weight dequant."""
        if x.dtype != torch.bfloat16:
            x = x.to(dtype=torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()

        orig_shape = x.shape
        if x.dim() != 2:
            x = x.view(-1, orig_shape[-1])

        self._prepare_packed()

        out = ext.fp8_ampere_mm(x, self._weight_packed, self._scale_padded)

        if len(orig_shape) != 2:
            out = out.view(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """BF16 fallback: decode FP8 weights then use cuBLAS."""
        w_bf16 = self.weight.view(torch.float8_e4m3fn).to(dtype=torch.bfloat16)
        return F.linear(x.to(dtype=torch.bfloat16), w_bf16, self.bias)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
