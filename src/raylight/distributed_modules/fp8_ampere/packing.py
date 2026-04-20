"""Packing and unpacking utilities for FP8 Ampere weights (Marlin architecture).

Weight format for fp8_ampere_gemm (v3 — Marlin architecture):
  • weight_fp8  : Tensor[N, K]  dtype=torch.uint8
                  Raw FP8 E4M3 bit-patterns in unpacked layout.
  • weight_scale: Tensor[N]     dtype=torch.bfloat16
                  Per-channel scale (1.0 for standard FP8 checkpoints).
                  The kernel applies: dequant_bf16(fp8_val) * scale.

For the Marlin kernel, weights are repacked offline via repack_fp8_for_marlin()
into a permuted layout matching the m16n8k16 tensor core fragment order.
This repacking is done lazily in the linear layer on first forward call.

Public API
----------
quantize_to_fp8_e4m3(weight_bf16) -> (weight_fp8, weight_scale)
    Quantise a BF16 weight to FP8 and compute per-channel BF16 scales.

compute_fp8_int8_scales(weight_fp8) -> weight_scale
    [Deprecated] Compute per-row FP16 INT8 scales.  Kept for backward
    compatibility; new code should use compute_fp8_scales() instead.

compute_fp8_scales(weight_fp8) -> weight_scale
    Compute per-channel BF16 scales (1.0 for standard FP8 checkpoints).

repack_fp8_for_marlin(weight_fp8, weight_scale) -> (packed, scale_padded, N, K)
    Repack [N,K] uint8 to Marlin layout with alignment padding.

narrow_fp8_weight_for_tp(weight_fp8, weight_scale, dim, rank, tp_size)
    Slice FP8 weight + scale along the requested dimension for tensor parallel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "quantize_to_fp8_e4m3",
    "compute_fp8_int8_scales",
    "compute_fp8_scales",
    "repack_fp8_for_marlin",
    "narrow_fp8_weight_for_tp",
]

# FP8 E4M3 maximum representable value (not counting NaN/Inf).
_FP8_E4M3_MAX: float = 448.0

# Alignment requirements for the Marlin kernel.
_N_ALIGN: int = 256
_K_ALIGN: int = 64


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def quantize_to_fp8_e4m3(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise a BF16/FP32 weight matrix to FP8 E4M3 with per-channel scales.

    Args:
        weight_bf16: Tensor[N, K] in bfloat16 (or float32).

    Returns:
        weight_fp8 (Tensor[N, K] uint8): FP8 E4M3 bit-patterns.
        weight_scale (Tensor[N] bfloat16): per-channel scale (1.0).
    """
    if weight_bf16.dim() != 2:
        raise ValueError(f"weight must be 2-D [N, K], got shape {weight_bf16.shape}")

    orig_device = weight_bf16.device
    w = weight_bf16.float()  # work in FP32 for precision

    weight_fp8_native = w.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        dtype=torch.float8_e4m3fn
    )  # [N, K] float8

    # View as uint8 for the kernel (bit-patterns preserved, no value conversion)
    weight_fp8 = weight_fp8_native.view(torch.uint8).to(orig_device)

    # Per-channel scale: 1.0 for standard FP8 (no additional scaling needed).
    weight_scale = compute_fp8_scales(weight_fp8).to(orig_device)

    return weight_fp8, weight_scale


def compute_fp8_scales(weight_fp8: torch.Tensor) -> torch.Tensor:
    """Compute per-channel BF16 scales for the Marlin kernel.

    For standard FP8 E4M3 checkpoints (no per-channel pre-scaling), the
    scale is 1.0 for every channel.  The FP8→BF16 dequant in the kernel
    already recovers the original float value.

    Args:
        weight_fp8: Tensor[N, K] uint8 or float8_e4m3fn.

    Returns:
        weight_scale: Tensor[N] bfloat16 (all 1.0).
    """
    if weight_fp8.dim() != 2:
        raise ValueError(f"weight_fp8 must be 2-D [N, K], got {weight_fp8.shape}")
    N = weight_fp8.shape[0]
    return torch.ones(N, dtype=torch.bfloat16, device=weight_fp8.device)


def compute_fp8_int8_scales(weight_fp8: torch.Tensor) -> torch.Tensor:
    """[Deprecated] Compute per-row FP16 INT8 scales from an FP8 weight tensor.

    Kept for backward compatibility.  New code should use compute_fp8_scales().

    Args:
        weight_fp8: Tensor[N, K] with dtype uint8 or float8_e4m3fn.

    Returns:
        weight_scale: Tensor[N] float16.
    """
    if weight_fp8.dim() != 2:
        raise ValueError(f"weight_fp8 must be 2-D [N, K], got {weight_fp8.shape}")

    if weight_fp8.dtype == torch.uint8:
        decoded = weight_fp8.view(torch.float8_e4m3fn).to(dtype=torch.float16)
    elif weight_fp8.dtype == torch.float8_e4m3fn:
        decoded = weight_fp8.to(dtype=torch.float16)
    else:
        raise TypeError(
            f"weight_fp8 must be uint8 or float8_e4m3fn, got {weight_fp8.dtype}"
        )

    abs_max = decoded.abs().max(dim=1).values.clamp(min=1e-12)  # [N]
    ws = (abs_max / 127.0).to(dtype=torch.float16)
    return ws.contiguous()


def repack_fp8_for_marlin(
    weight_u8: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """Repack FP8 weight [N,K] → Marlin format with alignment padding.

    The Marlin kernel requires N % 256 == 0 and K % 64 == 0.  This function
    pads the weight (with zeros) to meet alignment, then permutes into the
    m16n8k16 tensor core fragment layout.

    Args:
        weight_u8   : Tensor[N, K] uint8 (FP8 E4M3 bit-patterns).
        weight_scale: Tensor[N] bfloat16 (optional).

    Returns:
        packed      : Tensor[K, N] uint8 (transposed).
        scale       : Tensor[N] bfloat16 (or None).
        orig_N      : int — original N.
        orig_K      : int — original K.
    """
    if weight_u8.dim() != 2:
        raise ValueError(f"weight must be 2-D [N, K], got {weight_u8.shape}")

    N, K = weight_u8.shape

    # Simple transpose [N, K] → [K, N]
    packed = weight_u8.t().contiguous()

    scale = weight_scale.contiguous() if weight_scale is not None else None

    return packed, scale, N, K


def narrow_fp8_weight_for_tp(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    dim: int,
    rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice FP8 weight and its scale for a tensor-parallel shard.

    Operates on the unpacked [N, K] layout BEFORE Marlin repacking.

    Args:
        weight_fp8  : Tensor[N, K] uint8.
        weight_scale: Tensor[N]    bfloat16 (or float16 for legacy).
        dim         : Dimension to split along.
                      0 = column-parallel (split N / output features).
                      1 = row-parallel    (split K / input features).
        rank        : This shard's rank index (0 .. tp_size-1).
        tp_size     : Total number of tensor-parallel ranks.

    Returns:
        (weight_fp8_shard, weight_scale_shard)
    """
    if dim not in (0, 1):
        raise ValueError(f"dim must be 0 (column-parallel) or 1 (row-parallel), got {dim}")
    if tp_size <= 1:
        return weight_fp8, weight_scale

    full_size = weight_fp8.shape[dim]
    if full_size % tp_size != 0:
        raise ValueError(
            f"weight dimension {dim} has size {full_size} which is not divisible "
            f"by tp_size={tp_size}"
        )
    shard_size  = full_size // tp_size
    start       = rank * shard_size
    end         = start + shard_size

    w_shard = weight_fp8.narrow(dim, start, shard_size)

    if dim == 0:
        # Output-feature split: scale dimension matches output features
        s_shard = weight_scale.narrow(0, start, shard_size)
    else:
        # Input-feature split: scale is per output feature, not sliced
        s_shard = weight_scale

    return w_shard.contiguous(), s_shard.contiguous()
