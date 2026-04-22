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

__all__ = ["Fp8KernelMixin", "Fp8AmpereLinear"]

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


def _is_ampere_only(device: torch.device) -> bool:
    """Return True only for Ampere GPUs (SM 8.0–8.8).

    Ada Lovelace (SM 8.9) and Hopper+ (SM 9.x) have native FP8 hardware
    support and should use the platform's native FP8 path, not this
    Marlin-style BF16-tensor-core fallback kernel.
    """
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device)
    return props.major == 8 and props.minor <= 8  # SM 8.0–8.8 = Ampere only


# ---------------------------------------------------------------------------
# Fp8KernelMixin
# ---------------------------------------------------------------------------


class Fp8KernelMixin:
    """Mixin providing the ``on_device_activated`` hook for FP8 Ampere layers.

    Both ``Fp8AmpereLinear`` and ``TPFp8AmpereLinear`` inherit this mixin.
    ``ModelContext._activate_fp8_buffers`` walks every ``Fp8KernelMixin``
    instance in the model after each ``reload_to_cuda`` / ``model.load`` call
    and invokes ``on_device_activated`` to ensure Marlin-packed buffers are
    on the correct device.
    """

    def on_device_activated(self, device: torch.device) -> None:
        """Migrate packed weight buffers to *device* after a pinned-cache reload.

        Called by ``ModelContext._activate_fp8_buffers`` after every
        ``reload_to_cuda`` / ``model.load`` in all model contexts.  Safe to
        call multiple times — no-ops when buffers are already on *device*.
        """
        if not getattr(self, "_packed", False):
            return
        weight_packed = getattr(self, "weight_packed", None)
        if weight_packed is not None and weight_packed.device != device:
            self.weight_packed = weight_packed.to(device)
        scale_padded = getattr(self, "scale_padded", None)
        if scale_padded is not None and scale_padded.device != device:
            self.scale_padded = scale_padded.to(device)


# ---------------------------------------------------------------------------
# Fp8AmpereLinear
# ---------------------------------------------------------------------------


class Fp8AmpereLinear(Fp8KernelMixin, nn.Module):
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
        self.in_features = in_features
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
        # Marlin-packed weight + padded scale, registered as buffers so the
        # pinned cache can offload/reload them with the rest of the model.
        # Populated lazily by _prepare_packed(); _packed tracks whether packing
        # has happened (survives storage resize-to-0 during offload).
        self.register_buffer("weight_packed", None)
        self.register_buffer("scale_padded", None)
        self._packed: bool = False

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
        weight_fp8, weight_scale = quantize_to_fp8_e4m3(w.to(dtype=torch.bfloat16) if w.dtype != torch.bfloat16 else w)
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
            raise TypeError(f"from_fp8_checkpoint expects float8_e4m3fn or uint8, got {weight_param.dtype}")
        weight_scale = compute_fp8_scales(weight_u8)
        return cls.from_fp8_weight(weight_u8, weight_scale, bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _load_ext(self) -> object | None:
        if self._ext is None:
            self._ext = _try_load_extension()
        return self._ext

    def _prepare_packed(self, device: torch.device) -> None:
        """Lazily repack weight to Marlin format with alignment padding.

        After packing the raw FP8 buffer (``self.weight``) is freed — its
        storage is replaced with a 0-element placeholder — so that only the
        Marlin-layout copy lives in VRAM.  This matches vLLM's
        ``replace_parameter`` pattern and eliminates the permanent 2× VRAM
        overhead that would otherwise result from caching both layouts.

        ``weight_packed`` and ``scale_padded`` are registered buffers so the
        pinned cache includes them in ``offload_to_cpu`` / ``reload_to_cuda``
        cycles.  ``_packed`` is a plain Python bool that survives storage
        resize-to-0 and tells us not to re-pack after a reload.
        """
        if self._packed:
            # Already packed.  In partial-offload scenarios the pinned cache may
            # have left weight_packed / scale_padded on CPU while the activation
            # is on CUDA, so migrate lazily here if needed.
            if self.weight_packed is not None and self.weight_packed.device != device:
                self.weight_packed = self.weight_packed.to(device)
            if self.scale_padded is not None and self.scale_padded.device != device:
                self.scale_padded = self.scale_padded.to(device)
            return

        packed, scale_padded, _, _ = repack_fp8_for_marlin(
            self.weight,
            self.weight_scale,
        )
        self.weight_packed = packed
        self.scale_padded = scale_padded
        self._packed = True

        # Keep raw weight for fallback path (don't free it)
        # VRAM tradeoff: keeps 2x weight storage but enables fallback
        # self.weight.data = self.weight.new_empty(0)

    # Threshold for switching to cuBLAS fallback (tuned for RTX 3090)
    # Below this, kernel launch overhead dominates
    CUBLAS_THRESHOLD = 256 * 3072  # M * K > 784K elements (lowered from 1.5M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer.

        On Ampere GPUs with the native extension built this uses BF16 tensor
        cores via the Marlin-architecture kernel.  Otherwise it falls back to
        BF16 matmul after decoding the FP8 weights.

        For small matrices (M*K < threshold), uses cuBLAS fallback which has
        lower launch overhead and better performance.
        """
        ext = self._load_ext()
        device = x.device

        # Calculate effective matrix size
        M = x.shape[0] if x.dim() == 2 else x.view(-1, x.shape[-1]).shape[0]
        K = self.in_features
        use_custom_kernel = ext is not None and _is_ampere_only(device) and (M * K) >= self.CUBLAS_THRESHOLD

        if use_custom_kernel:
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

        self._prepare_packed(x.device)

        out = ext.fp8_ampere_mm(x, self.weight_packed, self.scale_padded)

        # Slice off padding columns if N was rounded up to BLOCK_N
        if out.shape[-1] != self.out_features:
            out = out[:, : self.out_features]

        if len(orig_shape) != 2:
            out = out.view(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)

        return out

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """BF16 fallback: decode FP8 weights then use cuBLAS."""
        if self.weight.numel() == 0:
            raise RuntimeError(
                "Fp8AmpereLinear: raw FP8 weight has been freed after Marlin packing and the CUDA extension is no longer available."
            )
        # Always decode to bfloat16 — this module's compute dtype is always bf16.
        # Casting x to float32 or float16 would misalign with the bfloat16 bias.
        w_bf16 = self.weight.view(torch.float8_e4m3fn).to(torch.bfloat16)
        bias = self.bias  # already bfloat16
        return F.linear(x.to(torch.bfloat16), w_bf16, bias)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
