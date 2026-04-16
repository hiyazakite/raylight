"""TurboQuant-based TP communication compression.

Compresses TPLinear partial sums before all_gather to reduce NCCL bandwidth.
Supports rotation-based decorrelation + N-bit group quantization + optional
step-to-step residual caching.

Phase 1: Signed permutation + group INT4/INT2 (pure PyTorch, no Triton).
Phase 2: Step-to-step residual compression with error feedback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# =============================================================================
# Optimal centroids for Gaussian-distributed rotated coordinates
# =============================================================================

# 2-bit (4 levels): MSE-optimal for N(0, 1) marginals
_CENTROIDS_2BIT = torch.tensor([-1.510, -0.453, 0.453, 1.510])

# Centroid boundaries for 2-bit: values below boundary[i] map to centroid[i]
# Boundaries are midpoints between adjacent centroids
_BOUNDARIES_2BIT = torch.tensor([-0.9815, 0.0, 0.9815])

# 3-bit (8 levels): Lloyd-Max optimal for N(0, 1) marginals
_CENTROIDS_3BIT = torch.tensor([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152])
_BOUNDARIES_3BIT = torch.tensor([-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748])


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class TPCompressConfig:
    """Configuration for TP communication compression.

    Constructed once from ExecutionStrategy and shared across all TPLinear
    instances.
    """

    mode: str = "none"  # "none" | "fp8" | "turboquant"
    bits: int = 4  # 2 | 3 | 4
    group_size: int = 64
    use_residual: bool = False
    rotation: str = "signperm"  # "signperm" | "wht"
    residual_bits: Optional[int] = None  # None = same as bits; set lower for more compression on deltas
    residual_skip_threshold: float = 0.0  # relative delta norm below which layer is skipped (0 = disabled)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def effective_residual_bits(self) -> int:
        """Bit-width used for compressing deltas.  Falls back to ``bits``."""
        if self.residual_bits is not None:
            return self.residual_bits
        return self.bits


# =============================================================================
# Pack / Unpack utilities
# =============================================================================


def _pack_int4(indices: torch.Tensor) -> torch.Tensor:
    """Pack INT4 indices (0..15) into uint8: 2 nibbles per byte.

    Args:
        indices: (..., N) tensor with values in [0, 15]. N must be even.

    Returns:
        (..., N // 2) uint8 tensor.
    """
    # Take pairs along last dim
    even = indices[..., 0::2]  # low nibble
    odd = indices[..., 1::2]  # high nibble
    return (odd.to(torch.uint8) << 4 | even.to(torch.uint8))


def _unpack_int4(packed: torch.Tensor, last_dim_size: int) -> torch.Tensor:
    """Unpack uint8 back to INT4 indices.

    Args:
        packed: (..., N // 2) uint8 tensor.
        last_dim_size: Original size of the last dimension (N).

    Returns:
        (..., N) tensor with values in [0, 15].
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Interleave low and high
    shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    out = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    if last_dim_size != out.shape[-1]:
        out = out[..., :last_dim_size]
    return out


def _pack_int3(indices: torch.Tensor) -> torch.Tensor:
    """Pack INT3 indices (0..7) into uint8: 8 values per 3 bytes.

    Layout per group of 8 values (v0..v7, each 3 bits = 24 bits = 3 bytes):
        byte0 = v0 | (v1 << 3) | ((v2 & 0x3) << 6)
        byte1 = (v2 >> 2) | (v3 << 1) | (v4 << 4) | ((v5 & 0x1) << 7)
        byte2 = (v5 >> 1) | (v6 << 2) | (v7 << 5)

    Args:
        indices: (..., N) tensor with values in [0, 7]. N must be divisible by 8.

    Returns:
        (..., N * 3 // 8) uint8 tensor.
    """
    v0 = indices[..., 0::8].to(torch.uint8)
    v1 = indices[..., 1::8].to(torch.uint8)
    v2 = indices[..., 2::8].to(torch.uint8)
    v3 = indices[..., 3::8].to(torch.uint8)
    v4 = indices[..., 4::8].to(torch.uint8)
    v5 = indices[..., 5::8].to(torch.uint8)
    v6 = indices[..., 6::8].to(torch.uint8)
    v7 = indices[..., 7::8].to(torch.uint8)

    byte0 = v0 | (v1 << 3) | ((v2 & 0x3) << 6)
    byte1 = (v2 >> 2) | (v3 << 1) | (v4 << 4) | ((v5 & 0x1) << 7)
    byte2 = (v5 >> 1) | (v6 << 2) | (v7 << 5)

    # Interleave: [byte0_0, byte1_0, byte2_0, byte0_1, byte1_1, byte2_1, ...]
    n_groups = byte0.shape[-1]
    shape = indices.shape[:-1] + (n_groups * 3,)
    out = torch.empty(shape, dtype=torch.uint8, device=indices.device)
    out[..., 0::3] = byte0
    out[..., 1::3] = byte1
    out[..., 2::3] = byte2
    return out


def _unpack_int3(packed: torch.Tensor, last_dim_size: int) -> torch.Tensor:
    """Unpack uint8 back to INT3 indices.

    Args:
        packed: (..., N * 3 // 8) uint8 tensor.
        last_dim_size: Original size of the last dimension (N).

    Returns:
        (..., N) tensor with values in [0, 7].
    """
    byte0 = packed[..., 0::3]
    byte1 = packed[..., 1::3]
    byte2 = packed[..., 2::3]

    v0 = byte0 & 0x07
    v1 = (byte0 >> 3) & 0x07
    v2 = ((byte0 >> 6) & 0x03) | ((byte1 & 0x01) << 2)
    v3 = (byte1 >> 1) & 0x07
    v4 = (byte1 >> 4) & 0x07
    v5 = ((byte1 >> 7) & 0x01) | ((byte2 & 0x03) << 1)
    v6 = (byte2 >> 2) & 0x07
    v7 = (byte2 >> 5) & 0x07

    n_groups = byte0.shape[-1]
    shape = packed.shape[:-1] + (n_groups * 8,)
    out = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::8] = v0
    out[..., 1::8] = v1
    out[..., 2::8] = v2
    out[..., 3::8] = v3
    out[..., 4::8] = v4
    out[..., 5::8] = v5
    out[..., 6::8] = v6
    out[..., 7::8] = v7
    if last_dim_size != out.shape[-1]:
        out = out[..., :last_dim_size]
    return out


def _pack_int2(indices: torch.Tensor) -> torch.Tensor:
    """Pack INT2 indices (0..3) into uint8: 4 values per byte.

    Args:
        indices: (..., N) tensor with values in [0, 3]. N must be divisible by 4.

    Returns:
        (..., N // 4) uint8 tensor.
    """
    i0 = indices[..., 0::4].to(torch.uint8)
    i1 = indices[..., 1::4].to(torch.uint8)
    i2 = indices[..., 2::4].to(torch.uint8)
    i3 = indices[..., 3::4].to(torch.uint8)
    return i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)


def _unpack_int2(packed: torch.Tensor, last_dim_size: int) -> torch.Tensor:
    """Unpack uint8 back to INT2 indices.

    Args:
        packed: (..., N // 4) uint8 tensor.
        last_dim_size: Original size of the last dimension (N).

    Returns:
        (..., N) tensor with values in [0, 3].
    """
    i0 = packed & 0x03
    i1 = (packed >> 2) & 0x03
    i2 = (packed >> 4) & 0x03
    i3 = (packed >> 6) & 0x03
    shape = packed.shape[:-1] + (packed.shape[-1] * 4,)
    out = torch.empty(shape, dtype=torch.uint8, device=packed.device)
    out[..., 0::4] = i0
    out[..., 1::4] = i1
    out[..., 2::4] = i2
    out[..., 3::4] = i3
    if last_dim_size != out.shape[-1]:
        out = out[..., :last_dim_size]
    return out


# =============================================================================
# Core compress / decompress
# =============================================================================


def _effective_group_size(dim: int, group_size: int) -> int:
    """Find largest group_size that evenly divides dim."""
    if dim % group_size == 0:
        return group_size
    for gs in [128, 64, 32, 16, 8, 4, 2, 1]:
        if gs <= group_size and dim % gs == 0:
            return gs
    return 1


def _turboquant_compress(
    x: torch.Tensor,
    signs: torch.Tensor,
    perm: torch.Tensor,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Compress a tensor using signed-permutation rotation + group quantization.

    Args:
        x: (..., C) tensor to compress.
        signs: (C,) tensor of ±1.0 for sign-flip rotation.
        perm: (C,) long tensor — channel permutation indices.
        group_size: Number of elements per quantization group.
        bits: Quantization bit-width (2 or 4).

    Returns:
        (packed, scales, zero_points, effective_group_size):
            packed: Packed uint8 tensor.
            scales: Per-group scale factors.
            zero_points: Per-group zero points (4-bit only, None for 2-bit).
            effective_group_size: Actual group size used.
    """
    # Apply sign-flip rotation + permutation
    rotated = x * signs
    rotated = rotated[..., perm]
    return _quantize_and_pack(rotated, group_size, bits)


def _quantize_and_pack(
    x: torch.Tensor,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Quantize and pack a (pre-rotated) tensor — no sign/perm applied.

    Used directly by the WHT path to avoid identity multiply overhead.
    """
    orig_shape = x.shape
    C = orig_shape[-1]
    gs = _effective_group_size(C, group_size)
    grouped = x.reshape(*orig_shape[:-1], C // gs, gs)

    if bits == 4:
        # Uniform affine quantization: 16 levels [0..15]
        vmin = grouped.amin(dim=-1, keepdim=True)
        vmax = grouped.amax(dim=-1, keepdim=True)
        scale = (vmax - vmin) / 15.0
        # Avoid division by zero for constant groups
        scale = scale.clamp(min=1e-10)
        indices = ((grouped - vmin) / scale).round().clamp(0, 15).to(torch.uint8)
        # Flatten group dim into last: (..., C)
        indices_flat = indices.reshape(*orig_shape[:-1], C)
        packed = _pack_int4(indices_flat)
        return packed, scale.squeeze(-1), vmin.squeeze(-1), gs
    elif bits == 3:
        # Centroid-based quantization: 8 Lloyd-Max levels, true 3-bit packing
        absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scale = absmax / 2.152  # Normalize so max centroid = absmax
        normalized = grouped / scale
        boundaries = _BOUNDARIES_3BIT.to(x.device, x.dtype)
        indices = torch.bucketize(normalized, boundaries).to(torch.uint8)
        indices_flat = indices.reshape(*orig_shape[:-1], C)
        packed = _pack_int3(indices_flat)
        return packed, scale.squeeze(-1), None, gs
    elif bits == 2:
        # Centroid-based quantization (PolarQuant-style)
        absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scale = absmax / 1.510  # Normalize so max centroid = absmax
        normalized = grouped / scale
        # Map to nearest centroid index: 3 boundaries for 4 centroids
        boundaries = _BOUNDARIES_2BIT.to(x.device, x.dtype)
        # indices: 0 where val < -0.9815, 1 where -0.9815 <= val < 0, etc.
        indices = torch.bucketize(normalized, boundaries).to(torch.uint8)
        indices_flat = indices.reshape(*orig_shape[:-1], C)
        packed = _pack_int2(indices_flat)
        return packed, scale.squeeze(-1), None, gs
    else:
        raise ValueError(f"Unsupported bits={bits}. Use 2, 3, or 4.")


def _turboquant_decompress(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    signs: torch.Tensor,
    perm_inv: torch.Tensor,
    orig_shape: tuple,
    group_size: int,
    bits: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decompress a TurboQuant-compressed tensor.

    Args:
        packed: Packed uint8 tensor from _turboquant_compress.
        scales: Per-group scale factors.
        zero_points: Per-group zero points (4-bit) or None (2-bit).
        signs: (C,) sign-flip tensor.
        perm_inv: (C,) inverse permutation.
        orig_shape: Original tensor shape.
        group_size: Group size used during compression.
        bits: Bit-width (2 or 4).
        dtype: Output dtype.

    Returns:
        Decompressed tensor with shape orig_shape.
    """
    flat = _dequantize_and_unpack(packed, scales, zero_points, orig_shape, group_size, bits, dtype)

    # Inverse permutation + sign-flip
    flat = flat[..., perm_inv]
    flat = flat * signs

    return flat


def _dequantize_and_unpack(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    orig_shape: tuple,
    group_size: int,
    bits: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize and unpack without applying any rotation inverse.

    Used directly by the WHT path to avoid identity permute/multiply overhead.
    """
    C = orig_shape[-1]

    if bits == 4:
        indices = _unpack_int4(packed, C)
        grouped_idx = indices.reshape(*orig_shape[:-1], C // group_size, group_size)
        dequant = grouped_idx.to(dtype) * scales.unsqueeze(-1) + zero_points.unsqueeze(-1)
    elif bits == 3:
        indices = _unpack_int3(packed, C)
        centroids = _CENTROIDS_3BIT.to(device=packed.device, dtype=dtype)
        grouped_idx = indices.reshape(*orig_shape[:-1], C // group_size, group_size)
        dequant = centroids[grouped_idx.long()] * scales.unsqueeze(-1)
    elif bits == 2:
        indices = _unpack_int2(packed, C)
        centroids = _CENTROIDS_2BIT.to(device=packed.device, dtype=dtype)
        grouped_idx = indices.reshape(*orig_shape[:-1], C // group_size, group_size)
        dequant = centroids[grouped_idx.long()] * scales.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported bits={bits}")

    return dequant.reshape(orig_shape).to(dtype)


# =============================================================================
# Serialization: flatten compressed data for NCCL transfer
# =============================================================================


def _compressed_payload_size(C: int, group_size: int, bits: int) -> tuple[int, int, int]:
    """Calculate the byte sizes of packed, scales, and zero_points.

    Returns sizes in number of bf16 elements (since we transfer as bf16).
    """
    n_groups = C // group_size
    if bits == 4:
        packed_bytes = C // 2  # uint8, C//2 bytes
        # Convert to bf16 element count (2 bytes each): ceil(packed_bytes / 2)
        packed_bf16 = (packed_bytes + 1) // 2
        scales_bf16 = n_groups
        zp_bf16 = n_groups
        return packed_bf16, scales_bf16, zp_bf16
    elif bits == 3:
        packed_bytes = C * 3 // 8  # true 3-bit: 8 values per 3 bytes
        packed_bf16 = (packed_bytes + 1) // 2
        scales_bf16 = n_groups
        return packed_bf16, scales_bf16, 0  # no zero_points
    elif bits == 2:
        packed_bytes = C // 4
        packed_bf16 = (packed_bytes + 1) // 2
        scales_bf16 = n_groups
        return packed_bf16, scales_bf16, 0
    else:
        raise ValueError(f"Unsupported bits={bits}")


def _serialize_compressed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    token_count: int,
    C: int,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """Serialize compressed data into a flat bf16 tensor for NCCL transfer.

    We view the packed uint8 data as bf16 (reinterpret) and concatenate
    scales (and zero_points for 4-bit) along the last dimension.
    Each token row is independently serialized.

    Returns:
        flat: (token_count, payload_per_token) bf16 tensor.
    """
    packed_bf16_size, scales_bf16_size, zp_bf16_size = _compressed_payload_size(C, group_size, bits)
    total_bf16 = packed_bf16_size + scales_bf16_size + zp_bf16_size

    flat = torch.empty(token_count, total_bf16, dtype=torch.bfloat16, device=packed.device)

    # Packed data: reinterpret uint8 as bf16
    packed_2d = packed.reshape(token_count, -1)  # (T, C//2 or C//4) uint8
    # Pad to even byte count if needed
    if packed_2d.shape[-1] % 2 != 0:
        packed_2d = torch.nn.functional.pad(packed_2d, (0, 1))
    packed_as_bf16 = packed_2d.view(torch.bfloat16)
    flat[:, :packed_bf16_size] = packed_as_bf16

    # Scales
    scales_2d = scales.reshape(token_count, -1).to(torch.bfloat16)
    flat[:, packed_bf16_size : packed_bf16_size + scales_bf16_size] = scales_2d

    # Zero points (4-bit only)
    if zero_points is not None and zp_bf16_size > 0:
        zp_2d = zero_points.reshape(token_count, -1).to(torch.bfloat16)
        flat[:, packed_bf16_size + scales_bf16_size :] = zp_2d

    return flat


def _deserialize_compressed(
    flat: torch.Tensor,
    C: int,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Deserialize a flat bf16 payload back to packed/scales/zero_points."""
    packed_bf16_size, scales_bf16_size, zp_bf16_size = _compressed_payload_size(C, group_size, bits)
    token_count = flat.shape[0]
    n_groups = C // group_size

    # Packed data
    packed_bf16 = flat[:, :packed_bf16_size]
    packed_uint8 = packed_bf16.contiguous().view(torch.uint8)
    # Trim to actual packed size
    if bits == 4:
        actual_packed_per_token = C // 2
    elif bits == 3:
        actual_packed_per_token = C * 3 // 8
    else:
        actual_packed_per_token = C // 4
    packed = packed_uint8[:, :actual_packed_per_token]

    # Scales — must be contiguous; Triton kernels assume dense row-major layout
    scales = flat[:, packed_bf16_size : packed_bf16_size + scales_bf16_size]
    scales = scales.reshape(token_count, n_groups).contiguous()

    # Zero points
    if zp_bf16_size > 0:
        zp = flat[:, packed_bf16_size + scales_bf16_size :]
        zp = zp.reshape(token_count, n_groups).contiguous()
    else:
        zp = None

    return packed, scales, zp


# =============================================================================
# TPResidualCache: step-to-step delta compression
# =============================================================================

# Global registry so sampler_manager can clear all caches between generations.
_ALL_RESIDUAL_CACHES: list["TPResidualCache"] = []


def clear_all_tp_residual_caches() -> None:
    """Clear every TPResidualCache. Called from sampler_manager before sampling."""
    for cache in _ALL_RESIDUAL_CACHES:
        cache.clear()


class TPResidualCache:
    """Per-TPCompressor cache for step-to-step residual compression.

    Stores the *reconstructed* (lossy) partial sum from the previous diffusion
    step so that at the next step only the delta needs to be sent. Error
    feedback (caching the reconstructed value, not the true value) prevents
    quantization error from accumulating across steps.

    Bases are stored in TurboQuant INT4 group quantization (group_size=64)
    which uses ~12 MB per base for a 6820×3072 activation — 44% less than
    FP8 (~21 MB) — while preserving per-group detail better than a single
    global FP8 scale.
    """

    _BASE_GROUP_SIZE = 64
    _BASE_BITS = 4

    def __init__(self) -> None:
        # key → (packed, scales, zero_points, orig_shape, group_size)
        self._bases: dict[tuple[str, int], tuple] = {}
        _ALL_RESIDUAL_CACHES.append(self)

    # ----- public API -------------------------------------------------------

    def get_base(
        self, key: str, rank: int, expected_shape: tuple, expected_dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Return decompressed base for *key*/*rank*, or ``None`` on cache miss."""
        entry = self._bases.get((key, rank))
        if entry is None:
            return None
        packed, scales, zp, orig_shape, gs = entry
        if orig_shape != expected_shape:
            return None
        return _dequantize_and_unpack(
            packed, scales, zp, orig_shape, gs, self._BASE_BITS, expected_dtype,
        )

    def put_base(
        self, key: str, rank: int, reconstructed: torch.Tensor
    ) -> None:
        """Store *reconstructed* in INT4 group quantization (error-feedback: cache the lossy value)."""
        packed, scales, zp, gs = _quantize_and_pack(
            reconstructed, self._BASE_GROUP_SIZE, self._BASE_BITS,
        )
        self._bases[(key, rank)] = (packed, scales, zp, reconstructed.shape, gs)

    def clear(self) -> None:
        """Call at start of new generation (new prompt / image)."""
        self._bases.clear()

    def __del__(self) -> None:
        # Unregister from the global list on garbage collection.
        try:
            _ALL_RESIDUAL_CACHES.remove(self)
        except ValueError:
            pass


# =============================================================================
# TPCompressor: per-layer compression state
# =============================================================================


class TPCompressor:
    """Per-TPLinear compression state.

    Holds rotation buffers (signs, perm, perm_inv) and performs compressed
    all-reduce via all_gather + local decompress + sum.

    Supports two rotation modes:
    - "signperm": Phase 1 signed permutation — O(d), ~32 KB storage.
    - "wht": Phase 3 Walsh-Hadamard Transform — O(d log d), ~32 KB storage,
      better decorrelation (15-25% MSE improvement at 2-bit).

    Args:
        config: Shared TPCompressConfig.
        hidden_dim: Channel dimension of the TPLinear output (full, unsharded).
        layer_id: Unique integer per TPLinear for deterministic rotation seed.
        device: CUDA device for buffers.
    """

    def __init__(
        self,
        config: TPCompressConfig,
        hidden_dim: int,
        layer_id: int,
        device: torch.device,
    ):
        self.config = config
        self.hidden_dim = hidden_dim
        self.layer_id = layer_id
        self.bits = config.bits
        self._rotation_mode = config.rotation

        seed = layer_id * 1337 + 42

        # --- Rotation setup ---
        self._wht = None
        self._fused_compressor = None
        if config.rotation == "wht":
            try:
                from raylight.distributed_modules.tp_compress_triton import (
                    WHTRotation,
                    FusedWHTCompressor,
                    _HAS_TRITON,
                )
                if _HAS_TRITON:
                    # Use fully fused kernel: WHT + quant + pack in one launch
                    self._fused_compressor = FusedWHTCompressor(
                        hidden_dim, config.group_size, bits=config.bits,
                        seed=seed, device=device,
                    )
                    quant_dim = self._fused_compressor.padded_dim
                    # Also create a standalone WHT rotation for the fallback
                    # path when residual_bits differs from bits (the fused
                    # compressor is compiled for a fixed bit-width).
                    self._wht = WHTRotation(hidden_dim, seed=seed, device=device)
                    logger.debug(
                        "Using fused WHT+compress Triton kernel (%d-bit) for layer %d.",
                        config.bits, layer_id,
                    )
                else:
                    logger.warning(
                        "Triton not available — falling back to PyTorch WHT."
                    )
                    from raylight.distributed_modules.tp_compress_triton import (
                        WHTRotationPyTorch,
                    )
                    self._wht = WHTRotationPyTorch(hidden_dim, seed=seed, device=device)
                    quant_dim = self._wht.padded_dim
            except ImportError:
                logger.warning(
                    "tp_compress_triton not importable — falling back to signperm."
                )
                self._rotation_mode = "signperm"

        if self._rotation_mode == "signperm":
            # Phase 1: signed permutation rotation
            gen = torch.Generator(device="cpu").manual_seed(seed)
            signs_cpu = torch.where(
                torch.rand(hidden_dim, generator=gen) > 0.5,
                torch.ones(hidden_dim),
                -torch.ones(hidden_dim),
            )
            perm_cpu = torch.randperm(hidden_dim, generator=gen)

            self.signs = signs_cpu.to(dtype=torch.bfloat16)
            self.perm = perm_cpu
            self.perm_inv = torch.argsort(self.perm)
            quant_dim = hidden_dim
        else:
            # WHT path — no signs/perm needed
            self.signs = None
            self.perm = None
            self.perm_inv = None

        self.group_size = _effective_group_size(quant_dim, config.group_size)
        self._quant_dim = quant_dim

        self._buffers_device: torch.device = torch.device("cpu")

        # Phase 2: Residual cache for step-to-step delta compression.
        self._residual_cache: Optional[TPResidualCache] = None
        if config.use_residual:
            self._residual_cache = TPResidualCache()
        self._cache_key = f"tp_layer_{layer_id}"

    @torch.no_grad()
    def compressed_all_reduce(
        self,
        x: torch.Tensor,
        group: Optional[dist.ProcessGroup],
    ) -> torch.Tensor:
        """Compress, all_gather, decompress, and sum across TP ranks.

        Args:
            x: (..., hidden_dim) partial sum tensor.
            group: TP process group.

        Returns:
            Summed result with same shape as x, dtype as x.
        """
        if group is None:
            return x

        tp_size = dist.get_world_size(group)
        if tp_size <= 1:
            return x

        orig_shape = x.shape
        orig_dtype = x.dtype
        C = orig_shape[-1]

        # Flatten leading dims: (T_total, C)
        x_2d = x.reshape(-1, C)
        T = x_2d.shape[0]

        if self.config.mode == "fp8":
            return self._fp8_all_reduce(x, group)

        # --- TurboQuant path ---
        my_rank = dist.get_rank(group)
        cache = self._residual_cache
        cache_key = self._cache_key

        # --- Residual: check if we have a warm base ---
        is_delta = False
        my_base = None
        if cache is not None:
            my_base = cache.get_base(cache_key, my_rank, (T, C), orig_dtype)
            if my_base is not None:
                is_delta = True
            else:
                # Warmup: no base cached yet.  Send the full uncompressed
                # activation via all_gather so every rank initialises its
                # base from the exact (lossless) value.  This avoids
                # polluting the base with TurboQuant quantisation noise on
                # the very first step, which would inflate all subsequent
                # deltas.
                warmup_gathered = [torch.empty_like(x_2d) for _ in range(tp_size)]
                dist.all_gather(warmup_gathered, x_2d.contiguous(), group=group)
                result = torch.zeros(T, C, dtype=orig_dtype, device=x.device)
                for rank_i, buf in enumerate(warmup_gathered):
                    result += buf
                    cache.put_base(cache_key, rank_i, buf)
                return result.reshape(orig_shape)

        # --- Skip check: if ALL ranks have negligible deltas, reuse cache ---
        skip_threshold = self.config.residual_skip_threshold
        if is_delta and skip_threshold > 0:
            base_norm = my_base.norm().clamp(min=1e-12)
            relative_norm = (x_2d - my_base).norm() / base_norm
            want_skip = relative_norm < skip_threshold
            # Cheap 1-element all_reduce so all ranks agree (skip only if unanimous)
            skip_vote = torch.tensor(
                [1.0 if want_skip else 0.0], device=x.device, dtype=torch.float32,
            )
            dist.all_reduce(skip_vote, op=dist.ReduceOp.MIN, group=group)
            if skip_vote.item() > 0.5:
                # All ranks' deltas are negligible — reuse cached bases.
                result = torch.zeros(T, C, dtype=orig_dtype, device=x.device)
                for rank_i in range(tp_size):
                    rank_base = cache.get_base(cache_key, rank_i, (T, C), orig_dtype)
                    if rank_base is not None:
                        result += rank_base
                return result.reshape(orig_shape)

        # --- Select bit-width: lower for deltas → smaller payload ---
        effective_bits = self.config.effective_residual_bits if is_delta else self.bits

        use_wht = self._rotation_mode == "wht" and self._wht is not None
        use_fused = self._rotation_mode == "wht" and self._fused_compressor is not None
        # Fused compressor is compiled for self.bits; fall back if bits differ.
        if use_fused and effective_bits != self.bits:
            use_fused = False
            use_wht = True  # self._wht is always set when rotation == "wht"

        if use_fused:
            # Fused path: single Triton kernel does WHT + quant + pack.
            # When is_delta, the kernel subtracts base in-register — no
            # separate delta tensor materialised.
            fc = self._fused_compressor
            Q = self._quant_dim
            gs = fc.group_size
            packed, scales, zp = fc.compress(
                x_2d, base=my_base if is_delta else None,
            )
            payload = _serialize_compressed(packed, scales, zp, T, Q, gs, effective_bits)
        elif use_wht:
            # WHT rotation → quantize directly (no identity sign/perm overhead)
            to_compress = (x_2d - my_base) if is_delta else x_2d
            rotated = self._wht.rotate_forward(to_compress)  # (T, padded_dim)
            Q = self._quant_dim
            packed, scales, zp, gs = _quantize_and_pack(
                rotated, self.group_size, effective_bits
            )
            payload = _serialize_compressed(packed, scales, zp, T, Q, gs, effective_bits)
        else:
            # Signed permutation: rotation embedded in compress
            to_compress = (x_2d - my_base) if is_delta else x_2d
            if self._buffers_device != x.device:
                self.signs = self.signs.to(device=x.device)
                self.perm = self.perm.to(device=x.device)
                self.perm_inv = self.perm_inv.to(device=x.device)
                self._buffers_device = x.device
            signs = self.signs.to(dtype=x.dtype)
            Q = C
            packed, scales, zp, gs = _turboquant_compress(
                to_compress, signs, self.perm, self.group_size, effective_bits
            )
            payload = _serialize_compressed(packed, scales, zp, T, Q, gs, effective_bits)

        # All-gather
        gathered = [torch.empty_like(payload) for _ in range(tp_size)]
        dist.all_gather(gathered, payload.contiguous(), group=group)

        # Decompress the sender's own contribution directly from the local
        # compressed tensors (before serialization) so that the cached base
        # matches exactly what the compress path produced.  Other ranks go
        # through the serialize→deserialize round-trip as before.
        if use_fused:
            # Fused: kernel adds base in-register when is_delta
            my_decompressed = self._fused_compressor.decompress(
                packed, scales, zp, base=my_base if is_delta else None,
            ).to(orig_dtype)
        elif use_wht:
            my_decompressed_rot = _dequantize_and_unpack(
                packed, scales, zp, (T, Q), gs, effective_bits, orig_dtype,
            )
            my_decompressed = self._wht.rotate_inverse(my_decompressed_rot).to(orig_dtype)
        else:
            my_decompressed = _turboquant_decompress(
                packed, scales, zp, signs, self.perm_inv,
                (T, C), gs, effective_bits, orig_dtype,
            )

        result = torch.zeros(T, C, dtype=orig_dtype, device=x.device)
        for rank_i, buf in enumerate(gathered):
            if rank_i == my_rank:
                # Use locally decompressed result to keep sender/receiver
                # bases identical (error-feedback invariant).
                decompressed = my_decompressed
            else:
                if use_fused:
                    fc = self._fused_compressor
                    p, s, z = _deserialize_compressed(buf, Q, gs, effective_bits)
                    # Fused: kernel adds base in-register when is_delta
                    rank_base_fused = cache.get_base(cache_key, rank_i, (T, C), orig_dtype) if (cache is not None and is_delta) else None
                    decompressed = fc.decompress(p, s, z, base=rank_base_fused).to(orig_dtype)
                elif use_wht:
                    p, s, z = _deserialize_compressed(buf, Q, gs, effective_bits)
                    decompressed_rotated = _dequantize_and_unpack(
                        p, s, z, (T, Q), gs, effective_bits, orig_dtype,
                    )
                    decompressed = self._wht.rotate_inverse(decompressed_rotated).to(orig_dtype)
                else:
                    p, s, z = _deserialize_compressed(buf, Q, gs, effective_bits)
                    decompressed = _turboquant_decompress(
                        p, s, z, signs, self.perm_inv,
                        (T, C), gs, effective_bits, orig_dtype,
                    )

            # Non-fused paths: add base explicitly (fused paths already
            # include base via the kernel).
            if not use_fused and cache is not None and is_delta:
                rank_base = cache.get_base(cache_key, rank_i, (T, C), orig_dtype)
                if rank_base is not None:
                    decompressed = decompressed + rank_base
            result += decompressed

            if cache is not None:
                cache.put_base(cache_key, rank_i, decompressed)

        return result.reshape(orig_shape)

    @torch.no_grad()
    def _fp8_all_reduce(
        self,
        x: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """FP8 E4M3 quantized all-reduce (2× bandwidth reduction).

        Falls back to standard bf16 all-reduce with fp8 cast if native
        fp8 collectives are not available.
        """
        orig_dtype = x.dtype
        scale = x.abs().amax() / 448.0
        scale = scale.clamp(min=1e-12)
        x_fp8 = (x / scale).to(torch.float8_e4m3fn)

        # NCCL doesn't support fp8 all-reduce natively in all PyTorch versions.
        # Use all_gather of fp8 + local sum as the reliable path.
        tp_size = dist.get_world_size(group)
        gathered = [torch.empty_like(x_fp8) for _ in range(tp_size)]

        # Gather scales separately (one scalar per rank)
        scale_list = [torch.empty(1, dtype=torch.float32, device=x.device) for _ in range(tp_size)]
        dist.all_gather(scale_list, scale.float().reshape(1), group=group)
        dist.all_gather(gathered, x_fp8.contiguous(), group=group)

        result = torch.zeros_like(x, dtype=orig_dtype)
        for i in range(tp_size):
            result += gathered[i].to(orig_dtype) * scale_list[i].item()

        return result
