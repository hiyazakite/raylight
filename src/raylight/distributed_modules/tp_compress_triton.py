"""Triton Walsh-Hadamard Transform (WHT) kernel for TurboQuant rotation.

Implements a fused sign-flip → WHT → sign-flip rotation using butterfly
operations in Triton.  This replaces the Phase 1 signed-permutation rotation
with a proper orthogonal decorrelation transform, improving quantization
quality especially at 2-bit (15-25% MSE reduction).

The WHT is applied per-row on a 2-D tensor of shape (T, C_padded) where
C_padded is the next power of 2 >= hidden_dim.  Padding columns are zeroed
and stripped after the inverse transform.

Butterfly stages use Triton's reshape → permute → split → operate → join →
reshape pattern to implement in-register element exchange.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# =============================================================================
# Triton WHT butterfly kernel
# =============================================================================

if _HAS_TRITON:

    @triton.jit
    def _wht_butterfly_stage(x, stride: tl.constexpr, C: tl.constexpr):
        """One butterfly stage of the Walsh-Hadamard Transform.

        For stride = 2^k, pairs elements (i, i ^ stride) and produces
        (a+b, a-b).  Implemented via reshape → permute → split → join.

        Args:
            x: 1-D tensor of shape [C] (C must be power of 2).
            stride: Butterfly stride (1, 2, 4, ..., C//2). Must be constexpr.
            C: Total number of elements.  Must be constexpr.

        Returns:
            1-D tensor of shape [C] after one butterfly stage.
        """
        # Number of butterfly groups
        n_groups: tl.constexpr = C // (2 * stride)

        # Reshape to [n_groups, 2, stride] to expose butterfly pairs
        x_3d = tl.reshape(x, [n_groups, 2, stride])
        # Move the pair dim to the end: [n_groups, stride, 2]
        x_3d = tl.permute(x_3d, [0, 2, 1])
        # Split on last dim (size 2) → two tensors of [n_groups, stride]
        a, b = tl.split(x_3d)
        # Butterfly: (a, b) → (a+b, a-b)
        sum_ab = a + b
        diff_ab = a - b
        # Rejoin: [n_groups, stride, 2]
        x_3d = tl.join(sum_ab, diff_ab)
        # Undo the permute: [n_groups, 2, stride]
        x_3d = tl.permute(x_3d, [0, 2, 1])
        # Flatten back to [C]
        return tl.reshape(x_3d, [C])

    # For C=4096 (log2=12), we statically unroll all 12 stages.
    # Each supported C size needs its own kernel variant because the number
    # of stages and the reshape dims must be constexpr.

    @triton.jit
    def _wht_forward_kernel(
        X_ptr,        # (T, C_padded) input
        OUT_ptr,      # (T, C_padded) output
        SIGNS1_ptr,   # (C_padded,) first sign-flip vector
        SIGNS2_ptr,   # (C_padded,) second sign-flip vector
        T,            # number of rows (tokens)
        C: tl.constexpr,           # padded channel dim (must be power of 2)
        LOG2_C: tl.constexpr,      # log2(C)
        NORMALIZE: tl.constexpr,   # whether to multiply by 1/sqrt(C)
    ):
        """Fused forward WHT: y = D2 · H · D1 · x.

        Each program handles one row.  C elements are loaded into registers
        and processed entirely in-register via butterfly stages.
        """
        row = tl.program_id(0)
        if row >= T:
            return

        offs = tl.arange(0, C)

        # Load row
        x = tl.load(X_ptr + row * C + offs).to(tl.float32)

        # Apply first sign-flip: D1 · x
        s1 = tl.load(SIGNS1_ptr + offs).to(tl.float32)
        x = x * s1

        # Butterfly stages — fully unrolled via static_range
        # Each stage k has stride = 2^k
        #
        # We must call the butterfly inline because Triton constexpr
        # propagation requires literal reshape dimensions.  We use
        # tl.static_range to unroll, then compute stride.
        #
        # Unfortunately, tl.reshape requires constexpr dimensions.
        # We handle this by manually unrolling the common sizes.
        # For C=4096 (LOG2_C=12): stages 0..11
        # For C=2048 (LOG2_C=11): stages 0..10
        # etc.
        #
        # The helper _wht_butterfly_stage takes constexpr stride and C,
        # so we can dispatch via static_range.

        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        # Normalize by 1/sqrt(C)
        if NORMALIZE:
            x = x * (1.0 / tl.sqrt(float(C)))

        # Apply second sign-flip: D2 · (H · D1 · x)
        s2 = tl.load(SIGNS2_ptr + offs).to(tl.float32)
        x = x * s2

        # Store
        tl.store(OUT_ptr + row * C + offs, x)

    @triton.jit
    def _wht_inverse_kernel(
        X_ptr,        # (T, C_padded) input (rotated values)
        OUT_ptr,      # (T, C_padded) output (de-rotated values)
        SIGNS1_ptr,   # (C_padded,) first sign-flip vector
        SIGNS2_ptr,   # (C_padded,) second sign-flip vector
        T,            # number of rows
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        NORMALIZE: tl.constexpr,
    ):
        """Inverse WHT: x = D1 · H · D2 · y.

        Since H is its own inverse (H·H = N·I) and D are diagonal,
        the inverse is D1 · (1/sqrt(C)) · H · D2 · y.
        The same butterfly stages are applied, just with signs swapped.
        """
        row = tl.program_id(0)
        if row >= T:
            return

        offs = tl.arange(0, C)

        # Load row
        x = tl.load(X_ptr + row * C + offs).to(tl.float32)

        # Reverse order: first undo D2, then H, then undo D1
        # Inverse of y = D2·H·D1·x is x = D1·H·D2·y
        # (D are self-inverse, H is self-inverse up to scaling)

        # Undo D2
        s2 = tl.load(SIGNS2_ptr + offs).to(tl.float32)
        x = x * s2

        # Butterfly stages (same as forward — H is symmetric & involutory)
        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        # Normalize
        if NORMALIZE:
            x = x * (1.0 / tl.sqrt(float(C)))

        # Undo D1
        s1 = tl.load(SIGNS1_ptr + offs).to(tl.float32)
        x = x * s1

        tl.store(OUT_ptr + row * C + offs, x)


# =============================================================================
# PyTorch wrappers
# =============================================================================


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


class WHTRotation:
    """Walsh-Hadamard Transform rotation for TurboQuant Phase 3.

    Precomputes two random sign-flip vectors (D1, D2) and handles
    padding to the next power of 2.  The butterfly kernel runs entirely
    in registers — no rotation matrix storage needed.

    Memory per instance: 2 × C_padded × 4 bytes ≈ 32 KB for C_padded=4096.
    Compare to ~32 MB for a dense rotation matrix.

    Args:
        hidden_dim: Original channel dimension (e.g. 3072).
        seed: Deterministic seed for sign-flip generation.
        device: CUDA device.
    """

    def __init__(self, hidden_dim: int, seed: int, device: torch.device):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is required for WHTRotation")

        self.hidden_dim = hidden_dim
        self.padded_dim = _next_power_of_2(hidden_dim)
        self.log2_dim = int(math.log2(self.padded_dim))

        if self.padded_dim > 8192:
            raise ValueError(
                f"hidden_dim={hidden_dim} pads to {self.padded_dim} > 8192. "
                "Kernel only supports up to 8192 (LOG2_C=13)."
            )

        # Deterministic sign-flip vectors — all ranks generate identical
        # vectors from the same seed.  Always materialise on CPU so that
        # structure-only (meta-device) construction works; _ensure_device
        # will lazily move them to the target CUDA device on first use.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.signs1 = torch.where(
            torch.rand(self.padded_dim, generator=gen) > 0.5,
            torch.ones(self.padded_dim),
            -torch.ones(self.padded_dim),
        ).to(dtype=torch.float32)
        self.signs2 = torch.where(
            torch.rand(self.padded_dim, generator=gen) > 0.5,
            torch.ones(self.padded_dim),
            -torch.ones(self.padded_dim),
        ).to(dtype=torch.float32)

        self._device = torch.device("cpu")

    def _ensure_device(self, device: torch.device) -> None:
        """Lazily move sign vectors to the correct device."""
        if self._device != device:
            self.signs1 = self.signs1.to(device=device)
            self.signs2 = self.signs2.to(device=device)
            self._device = device

    def rotate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply WHT rotation: y = D2 · H · D1 · x.

        Args:
            x: (..., hidden_dim) tensor.

        Returns:
            (..., padded_dim) tensor with the rotated values.
            Caller must handle the padding columns (they are valid rotated
            values — include them in quantization, strip after inverse).
        """
        self._ensure_device(x.device)
        orig_shape = x.shape
        C = orig_shape[-1]
        x_2d = x.reshape(-1, C)
        T = x_2d.shape[0]

        # Pad to power-of-2 if needed
        if C < self.padded_dim:
            x_2d = torch.nn.functional.pad(x_2d, (0, self.padded_dim - C))

        out = torch.empty_like(x_2d)
        grid = (T,)

        _wht_forward_kernel[grid](
            x_2d, out, self.signs1, self.signs2,
            T,
            C=self.padded_dim,
            LOG2_C=self.log2_dim,
            NORMALIZE=True,
            num_warps=max(1, self.padded_dim // 256),
        )

        return out

    def rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse WHT: x = D1 · H · D2 · y.

        Args:
            y: (..., padded_dim) rotated tensor.

        Returns:
            (..., hidden_dim) tensor with original values (up to float rounding).
        """
        self._ensure_device(y.device)
        T = y.shape[0] if y.dim() == 2 else y.reshape(-1, y.shape[-1]).shape[0]
        y_2d = y.reshape(-1, self.padded_dim)

        out = torch.empty_like(y_2d)
        grid = (T,)

        _wht_inverse_kernel[grid](
            y_2d, out, self.signs1, self.signs2,
            T,
            C=self.padded_dim,
            LOG2_C=self.log2_dim,
            NORMALIZE=True,
            num_warps=max(1, self.padded_dim // 256),
        )

        # Strip padding
        if self.hidden_dim < self.padded_dim:
            out = out[:, :self.hidden_dim]

        return out


# =============================================================================
# Fused WHT + quantize + pack kernel (Phase 4)
# =============================================================================
# Eliminates intermediate tensors and extra kernel launches by doing
# rotation, quantization, and bit-packing in a single Triton program.

if _HAS_TRITON:

    @triton.jit
    def _fused_wht_compress_2bit_kernel(
        X_ptr,          # (T, C_padded) input
        PACKED_ptr,     # (T, C//4) uint8 output
        SCALES_ptr,     # (T, n_groups) bf16 output
        SIGNS1_ptr,     # (C_padded,) sign vector
        SIGNS2_ptr,     # (C_padded,) sign vector
        BASE_ptr,       # (T, HIDDEN_DIM) residual base (only read when HAS_RESIDUAL)
        T,
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        GS: tl.constexpr,          # group size (constexpr for reshape)
        HAS_RESIDUAL: tl.constexpr = False,
        HIDDEN_DIM: tl.constexpr = 0,
    ):
        """Fused: sign1 → WHT butterfly → normalize → sign2 → 2-bit quant → pack.

        One program per row. All C elements stay in registers throughout.
        """
        row = tl.program_id(0)
        if row >= T:
            return

        offs = tl.arange(0, C)

        # --- Load and optionally subtract residual base ---
        x = tl.load(X_ptr + row * C + offs).to(tl.float32)
        if HAS_RESIDUAL:
            base_mask = offs < HIDDEN_DIM
            base = tl.load(BASE_ptr + row * HIDDEN_DIM + offs, mask=base_mask, other=0.0).to(tl.float32)
            x = x - base

        # --- Apply sign1 ---
        s1 = tl.load(SIGNS1_ptr + offs).to(tl.float32)
        x = x * s1

        # --- WHT butterfly stages ---
        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        # --- Normalize ---
        x = x * (1.0 / tl.sqrt(float(C)))

        # --- Apply sign2 ---
        s2 = tl.load(SIGNS2_ptr + offs).to(tl.float32)
        x = x * s2

        # --- 2-bit group quantization ---
        # Reshape to (n_groups, GS)
        N_GROUPS: tl.constexpr = C // GS
        x_g = tl.reshape(x, [N_GROUPS, GS])

        # Per-group absmax
        absmax = tl.max(tl.abs(x_g), axis=1)  # (N_GROUPS,)
        absmax = tl.maximum(absmax, 1e-10)
        scale = absmax / 1.510  # (N_GROUPS,)

        # Normalize within groups
        normalized = x_g / scale[:, None]

        # Quantize to 4 centroids: boundaries at -0.9815, 0.0, 0.9815
        # idx: 0 if < -0.9815, 1 if < 0.0, 2 if < 0.9815, 3 otherwise
        idx = tl.zeros([N_GROUPS, GS], dtype=tl.int32)
        idx = tl.where(normalized >= -0.9815, idx + 1, idx)
        idx = tl.where(normalized >= 0.0, idx + 1, idx)
        idx = tl.where(normalized >= 0.9815, idx + 1, idx)

        # --- Pack 4 x 2-bit into uint8 ---
        # Reshape to (N_GROUPS, GS//4, 4) then split via tl.split
        PACK_GROUPS: tl.constexpr = GS // 4
        idx_p = tl.reshape(idx, [N_GROUPS, PACK_GROUPS, 4])

        # tl.split only works on last dim of size 2, so split twice.
        # split on [2,2] extracts columns: lo=[q0,q2], hi=[q1,q3]
        # so i0=q0, i1=q2, i2=q1, i3=q3 — swap i1/i2 in packing
        # to produce standard order: q0|(q1<<2)|(q2<<4)|(q3<<6)
        idx_lo2, idx_hi2 = tl.split(tl.reshape(idx_p, [N_GROUPS, PACK_GROUPS, 2, 2]))  # each (N_GROUPS, PACK_GROUPS, 2)
        i0, i1 = tl.split(idx_lo2)  # i0=q0, i1=q2
        i2, i3 = tl.split(idx_hi2)  # i2=q1, i3=q3
        packed = (i0 | (i2 << 2) | (i1 << 4) | (i3 << 6)).to(tl.uint8)

        # --- Store packed data ---
        # packed shape: (N_GROUPS, PACK_GROUPS) → flatten to (C//4,)
        PACKED_PER_ROW: tl.constexpr = C // 4
        packed_flat = tl.reshape(packed, [PACKED_PER_ROW])
        packed_offs = tl.arange(0, PACKED_PER_ROW)
        tl.store(PACKED_ptr + row * PACKED_PER_ROW + packed_offs, packed_flat)

        # --- Store scales ---
        scale_offs = tl.arange(0, N_GROUPS)
        tl.store(SCALES_ptr + row * N_GROUPS + scale_offs, scale.to(tl.bfloat16))

    @triton.jit
    def _fused_wht_decompress_2bit_kernel(
        PACKED_ptr,     # (T, C//4) uint8 input
        SCALES_ptr,     # (T, n_groups) bf16 input
        OUT_ptr,        # (T, C_padded) output
        SIGNS1_ptr,     # (C_padded,) sign vector
        SIGNS2_ptr,     # (C_padded,) sign vector
        BASE_ptr,       # (T, OUT_DIM) residual base (only read when HAS_RESIDUAL)
        T,
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        GS: tl.constexpr,
        OUT_DIM: tl.constexpr,   # hidden_dim (for stripping padding on write)
        HAS_RESIDUAL: tl.constexpr = False,
    ):
        """Fused: unpack → dequant → sign2 → WHT butterfly → normalize → sign1.

        Inverse of _fused_wht_compress_2bit_kernel. One program per row.
        """
        row = tl.program_id(0)
        if row >= T:
            return

        N_GROUPS: tl.constexpr = C // GS
        PACKED_PER_ROW: tl.constexpr = C // 4
        PACK_GROUPS: tl.constexpr = GS // 4

        # --- Load packed data ---
        packed_offs = tl.arange(0, PACKED_PER_ROW)
        packed_flat = tl.load(PACKED_ptr + row * PACKED_PER_ROW + packed_offs)

        # --- Unpack 4 x 2-bit from uint8 ---
        packed_g = tl.reshape(packed_flat, [N_GROUPS, PACK_GROUPS])
        i0 = (packed_g & 0x03).to(tl.int32)
        i1 = ((packed_g >> 2) & 0x03).to(tl.int32)
        i2 = ((packed_g >> 4) & 0x03).to(tl.int32)
        i3 = ((packed_g >> 6) & 0x03).to(tl.int32)

        # Reconstruct indices as (N_GROUPS, PACK_GROUPS, 2, 2) then reshape to (N_GROUPS, GS).
        # tl.join adds a new LAST dim, so nested joins produce [2,2] where the
        # outer-join index varies fastest in row-major.  Interleave i0/i2 and
        # i1/i3 so that row-major flatten gives [q0, q1, q2, q3].
        idx_p = tl.join(tl.join(i0, i2), tl.join(i1, i3))
        # idx_p shape: (N_GROUPS, PACK_GROUPS, 2, 2)
        idx = tl.reshape(idx_p, [N_GROUPS, GS])

        # --- Load scales and dequantize ---
        scale_offs = tl.arange(0, N_GROUPS)
        scale = tl.load(SCALES_ptr + row * N_GROUPS + scale_offs).to(tl.float32)

        # Centroid lookup: [-1.510, -0.453, 0.453, 1.510]
        c0: tl.constexpr = -1.510
        c1: tl.constexpr = -0.453
        c2: tl.constexpr = 0.453
        c3: tl.constexpr = 1.510
        # Map indices to centroids
        vals = tl.where(idx == 0, c0, tl.where(idx == 1, c1, tl.where(idx == 2, c2, c3)))
        x_g = vals.to(tl.float32) * scale[:, None]

        # Flatten groups to (C,)
        x = tl.reshape(x_g, [C])

        # --- Inverse WHT: undo sign2, butterfly, normalize, undo sign1 ---
        s2 = tl.load(SIGNS2_ptr + tl.arange(0, C)).to(tl.float32)
        x = x * s2

        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        x = x * (1.0 / tl.sqrt(float(C)))

        s1 = tl.load(SIGNS1_ptr + tl.arange(0, C)).to(tl.float32)
        x = x * s1

        # --- Add residual base and store ---
        out_offs = tl.arange(0, C)
        out_mask = out_offs < OUT_DIM
        if HAS_RESIDUAL:
            base = tl.load(BASE_ptr + row * OUT_DIM + out_offs, mask=out_mask, other=0.0).to(tl.float32)
            x = x + base
        tl.store(OUT_ptr + row * OUT_DIM + out_offs, x, mask=out_mask)

    # -----------------------------------------------------------------
    # 3-bit fused kernels (8 Lloyd-Max centroids, true 3-bit packing)
    # -----------------------------------------------------------------

    @triton.jit
    def _fused_wht_compress_3bit_kernel(
        X_ptr,          # (T, C_padded) input
        PACKED_ptr,     # (T, C*3//8) uint8 output
        SCALES_ptr,     # (T, n_groups) bf16 output
        SIGNS1_ptr,     # (C_padded,) sign vector
        SIGNS2_ptr,     # (C_padded,) sign vector
        BASE_ptr,       # (T, HIDDEN_DIM) residual base (only read when HAS_RESIDUAL)
        T,
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        GS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr = False,
        HIDDEN_DIM: tl.constexpr = 0,
    ):
        """Fused: sign1 → WHT butterfly → normalize → sign2 → 3-bit quant → true 3-bit pack."""
        row = tl.program_id(0)
        if row >= T:
            return

        offs = tl.arange(0, C)

        # --- Load and optionally subtract residual base ---
        x = tl.load(X_ptr + row * C + offs).to(tl.float32)
        if HAS_RESIDUAL:
            base_mask = offs < HIDDEN_DIM
            base = tl.load(BASE_ptr + row * HIDDEN_DIM + offs, mask=base_mask, other=0.0).to(tl.float32)
            x = x - base

        # --- Apply sign1 ---
        s1 = tl.load(SIGNS1_ptr + offs).to(tl.float32)
        x = x * s1

        # --- WHT butterfly stages ---
        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        # --- Normalize ---
        x = x * (1.0 / tl.sqrt(float(C)))

        # --- Apply sign2 ---
        s2 = tl.load(SIGNS2_ptr + offs).to(tl.float32)
        x = x * s2

        # --- 3-bit group quantization (8 Lloyd-Max centroids) ---
        N_GROUPS: tl.constexpr = C // GS
        x_g = tl.reshape(x, [N_GROUPS, GS])

        # Per-group absmax scaling
        absmax = tl.max(tl.abs(x_g), axis=1)  # (N_GROUPS,)
        absmax = tl.maximum(absmax, 1e-10)
        scale = absmax / 2.152  # (N_GROUPS,)

        # Normalize within groups
        normalized = x_g / scale[:, None]

        # Quantize to 8 centroids via 7 boundaries
        idx = tl.zeros([N_GROUPS, GS], dtype=tl.int32)
        idx = tl.where(normalized >= -1.748, idx + 1, idx)
        idx = tl.where(normalized >= -1.050, idx + 1, idx)
        idx = tl.where(normalized >= -0.501, idx + 1, idx)
        idx = tl.where(normalized >= 0.0, idx + 1, idx)
        idx = tl.where(normalized >= 0.501, idx + 1, idx)
        idx = tl.where(normalized >= 1.050, idx + 1, idx)
        idx = tl.where(normalized >= 1.748, idx + 1, idx)

        # --- True 3-bit packing: 8 values → 3 bytes ---
        # Flatten to 1-D then reshape to [PACK8, 2, 2, 2] for triple split
        PACK8: tl.constexpr = C // 8
        idx_flat = tl.reshape(idx, [C])
        idx_222 = tl.reshape(idx_flat, [PACK8, 2, 2, 2])

        # Triple split extracts 8 scalar streams:
        #   split → c0[p,a,b]=even, c1[p,a,b]=odd over last dim
        #   split → c00[p,a]=v0/v4, c01[p,a]=v2/v6, c10[p,a]=v1/v5, c11[p,a]=v3/v7
        #   split → individual v0..v7
        c0, c1 = tl.split(idx_222)
        c00, c01 = tl.split(c0)
        c10, c11 = tl.split(c1)
        v0, v4 = tl.split(c00)
        v2, v6 = tl.split(c01)
        v1, v5 = tl.split(c10)
        v3, v7 = tl.split(c11)

        # Pack 8×3-bit into 24-bit int32, then extract 3 bytes
        packed = (v0 | (v1 << 3) | (v2 << 6) | (v3 << 9)
                 | (v4 << 12) | (v5 << 15) | (v6 << 18) | (v7 << 21))
        byte0 = (packed & 0xFF).to(tl.uint8)
        byte1 = ((packed >> 8) & 0xFF).to(tl.uint8)
        byte2 = ((packed >> 16) & 0xFF).to(tl.uint8)

        # Store with stride-3 pattern: [b0_0, b1_0, b2_0, b0_1, b1_1, b2_1, ...]
        PACKED_PER_ROW: tl.constexpr = C * 3 // 8
        base = row * PACKED_PER_ROW
        stride_offs = tl.arange(0, PACK8)
        tl.store(PACKED_ptr + base + stride_offs * 3, byte0)
        tl.store(PACKED_ptr + base + stride_offs * 3 + 1, byte1)
        tl.store(PACKED_ptr + base + stride_offs * 3 + 2, byte2)

        # --- Store scales ---
        scale_offs = tl.arange(0, N_GROUPS)
        tl.store(SCALES_ptr + row * N_GROUPS + scale_offs, scale.to(tl.bfloat16))

    @triton.jit
    def _fused_wht_decompress_3bit_kernel(
        PACKED_ptr,     # (T, C*3//8) uint8 input
        SCALES_ptr,     # (T, n_groups) bf16 input
        OUT_ptr,        # (T, OUT_DIM) output
        SIGNS1_ptr,     # (C_padded,) sign vector
        SIGNS2_ptr,     # (C_padded,) sign vector
        BASE_ptr,       # (T, OUT_DIM) residual base (only read when HAS_RESIDUAL)
        T,
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        GS: tl.constexpr,
        OUT_DIM: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr = False,
    ):
        """Fused: true 3-bit unpack → centroid dequant → sign2 → WHT → normalize → sign1."""
        row = tl.program_id(0)
        if row >= T:
            return

        N_GROUPS: tl.constexpr = C // GS
        PACK8: tl.constexpr = C // 8
        PACKED_PER_ROW: tl.constexpr = C * 3 // 8

        # --- Load packed data with stride-3 ---
        base = row * PACKED_PER_ROW
        stride_offs = tl.arange(0, PACK8)
        byte0 = tl.load(PACKED_ptr + base + stride_offs * 3)
        byte1 = tl.load(PACKED_ptr + base + stride_offs * 3 + 1)
        byte2 = tl.load(PACKED_ptr + base + stride_offs * 3 + 2)

        # --- Unpack 8 values from 3 bytes via 24-bit int32 ---
        packed = byte0.to(tl.int32) | (byte1.to(tl.int32) << 8) | (byte2.to(tl.int32) << 16)
        v0 = packed & 0x7
        v1 = (packed >> 3) & 0x7
        v2 = (packed >> 6) & 0x7
        v3 = (packed >> 9) & 0x7
        v4 = (packed >> 12) & 0x7
        v5 = (packed >> 15) & 0x7
        v6 = (packed >> 18) & 0x7
        v7 = (packed >> 21) & 0x7

        # Reassemble to flat [C] via inverse of the triple-split
        c00 = tl.join(v0, v4)    # [PACK8, 2]
        c01 = tl.join(v2, v6)
        c10 = tl.join(v1, v5)
        c11 = tl.join(v3, v7)
        c0 = tl.join(c00, c01)   # [PACK8, 2, 2]
        c1 = tl.join(c10, c11)
        idx_222 = tl.join(c0, c1) # [PACK8, 2, 2, 2]
        idx = tl.reshape(idx_222, [N_GROUPS, GS])

        # --- Load scales and dequantize via centroid lookup ---
        scale_offs = tl.arange(0, N_GROUPS)
        scale = tl.load(SCALES_ptr + row * N_GROUPS + scale_offs).to(tl.float32)

        # 8 Lloyd-Max centroids for N(0,1)
        c_0: tl.constexpr = -2.152
        c_1: tl.constexpr = -1.344
        c_2: tl.constexpr = -0.756
        c_3: tl.constexpr = -0.245
        c_4: tl.constexpr = 0.245
        c_5: tl.constexpr = 0.756
        c_6: tl.constexpr = 1.344
        c_7: tl.constexpr = 2.152
        vals = tl.where(idx == 0, c_0,
               tl.where(idx == 1, c_1,
               tl.where(idx == 2, c_2,
               tl.where(idx == 3, c_3,
               tl.where(idx == 4, c_4,
               tl.where(idx == 5, c_5,
               tl.where(idx == 6, c_6, c_7)))))))
        x_g = vals.to(tl.float32) * scale[:, None]

        # Flatten groups to (C,)
        x = tl.reshape(x_g, [C])

        # --- Inverse WHT: undo sign2, butterfly, normalize, undo sign1 ---
        s2 = tl.load(SIGNS2_ptr + tl.arange(0, C)).to(tl.float32)
        x = x * s2

        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        x = x * (1.0 / tl.sqrt(float(C)))

        s1 = tl.load(SIGNS1_ptr + tl.arange(0, C)).to(tl.float32)
        x = x * s1

        # --- Add residual base and store ---
        out_offs = tl.arange(0, C)
        out_mask = out_offs < OUT_DIM
        if HAS_RESIDUAL:
            base = tl.load(BASE_ptr + row * OUT_DIM + out_offs, mask=out_mask, other=0.0).to(tl.float32)
            x = x + base
        tl.store(OUT_ptr + row * OUT_DIM + out_offs, x, mask=out_mask)

    # -----------------------------------------------------------------
    # 4-bit fused kernels
    # -----------------------------------------------------------------

    @triton.jit
    def _fused_wht_compress_4bit_kernel(
        X_ptr,          # (T, C_padded) input
        PACKED_ptr,     # (T, C//2) uint8 output
        SCALES_ptr,     # (T, n_groups) bf16 output
        ZP_ptr,         # (T, n_groups) bf16 output (zero points = vmin)
        SIGNS1_ptr,     # (C_padded,) sign vector
        SIGNS2_ptr,     # (C_padded,) sign vector
        BASE_ptr,       # (T, HIDDEN_DIM) residual base (only read when HAS_RESIDUAL)
        T,
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        GS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr = False,
        HIDDEN_DIM: tl.constexpr = 0,
    ):
        """Fused: sign1 → WHT butterfly → normalize → sign2 → 4-bit quant → pack."""
        row = tl.program_id(0)
        if row >= T:
            return

        offs = tl.arange(0, C)

        # --- Load and optionally subtract residual base ---
        x = tl.load(X_ptr + row * C + offs).to(tl.float32)
        if HAS_RESIDUAL:
            base_mask = offs < HIDDEN_DIM
            base = tl.load(BASE_ptr + row * HIDDEN_DIM + offs, mask=base_mask, other=0.0).to(tl.float32)
            x = x - base

        # --- Apply sign1 ---
        s1 = tl.load(SIGNS1_ptr + offs).to(tl.float32)
        x = x * s1

        # --- WHT butterfly stages ---
        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        x = x * (1.0 / tl.sqrt(float(C)))

        # --- Apply sign2 ---
        s2 = tl.load(SIGNS2_ptr + offs).to(tl.float32)
        x = x * s2

        # --- 4-bit uniform affine group quantization ---
        N_GROUPS: tl.constexpr = C // GS
        x_g = tl.reshape(x, [N_GROUPS, GS])

        vmin = tl.min(x_g, axis=1)   # (N_GROUPS,)
        vmax = tl.max(x_g, axis=1)   # (N_GROUPS,)
        scale = (vmax - vmin) / 15.0
        scale = tl.maximum(scale, 1e-10)

        # Quantize to [0..15]
        idx = ((x_g - vmin[:, None]) / scale[:, None] + 0.5).to(tl.int32)
        idx = tl.maximum(idx, 0)
        idx = tl.minimum(idx, 15)

        # --- Pack 2 x 4-bit into uint8 ---
        PACK_GROUPS: tl.constexpr = GS // 2
        idx_p = tl.reshape(idx, [N_GROUPS, PACK_GROUPS, 2])
        lo, hi = tl.split(idx_p)  # each (N_GROUPS, PACK_GROUPS)
        packed = (lo | (hi << 4)).to(tl.uint8)

        # --- Store packed data ---
        PACKED_PER_ROW: tl.constexpr = C // 2
        packed_flat = tl.reshape(packed, [PACKED_PER_ROW])
        packed_offs = tl.arange(0, PACKED_PER_ROW)
        tl.store(PACKED_ptr + row * PACKED_PER_ROW + packed_offs, packed_flat)

        # --- Store scales and zero points ---
        g_offs = tl.arange(0, N_GROUPS)
        tl.store(SCALES_ptr + row * N_GROUPS + g_offs, scale.to(tl.bfloat16))
        tl.store(ZP_ptr + row * N_GROUPS + g_offs, vmin.to(tl.bfloat16))

    @triton.jit
    def _fused_wht_decompress_4bit_kernel(
        PACKED_ptr,     # (T, C//2) uint8 input
        SCALES_ptr,     # (T, n_groups) bf16 input
        ZP_ptr,         # (T, n_groups) bf16 input (zero points = vmin)
        OUT_ptr,        # (T, OUT_DIM) output
        SIGNS1_ptr,
        SIGNS2_ptr,
        BASE_ptr,       # (T, OUT_DIM) residual base (only read when HAS_RESIDUAL)
        T,
        C: tl.constexpr,
        LOG2_C: tl.constexpr,
        GS: tl.constexpr,
        OUT_DIM: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr = False,
    ):
        """Fused: unpack → dequant → sign2 → WHT butterfly → normalize → sign1."""
        row = tl.program_id(0)
        if row >= T:
            return

        N_GROUPS: tl.constexpr = C // GS
        PACKED_PER_ROW: tl.constexpr = C // 2
        PACK_GROUPS: tl.constexpr = GS // 2

        # --- Load packed data ---
        packed_offs = tl.arange(0, PACKED_PER_ROW)
        packed_flat = tl.load(PACKED_ptr + row * PACKED_PER_ROW + packed_offs)

        # --- Unpack 2 x 4-bit from uint8 ---
        packed_g = tl.reshape(packed_flat, [N_GROUPS, PACK_GROUPS])
        lo = (packed_g & 0x0F).to(tl.int32)
        hi = ((packed_g >> 4) & 0x0F).to(tl.int32)
        idx_p = tl.join(lo, hi)  # (N_GROUPS, PACK_GROUPS, 2)
        idx = tl.reshape(idx_p, [N_GROUPS, GS])

        # --- Load scales/zp and dequantize ---
        g_offs = tl.arange(0, N_GROUPS)
        scale = tl.load(SCALES_ptr + row * N_GROUPS + g_offs).to(tl.float32)
        vmin = tl.load(ZP_ptr + row * N_GROUPS + g_offs).to(tl.float32)
        x_g = idx.to(tl.float32) * scale[:, None] + vmin[:, None]

        # Flatten groups to (C,)
        x = tl.reshape(x_g, [C])

        # --- Inverse WHT ---
        s2 = tl.load(SIGNS2_ptr + tl.arange(0, C)).to(tl.float32)
        x = x * s2

        if LOG2_C >= 1:
            x = _wht_butterfly_stage(x, 1, C)
        if LOG2_C >= 2:
            x = _wht_butterfly_stage(x, 2, C)
        if LOG2_C >= 3:
            x = _wht_butterfly_stage(x, 4, C)
        if LOG2_C >= 4:
            x = _wht_butterfly_stage(x, 8, C)
        if LOG2_C >= 5:
            x = _wht_butterfly_stage(x, 16, C)
        if LOG2_C >= 6:
            x = _wht_butterfly_stage(x, 32, C)
        if LOG2_C >= 7:
            x = _wht_butterfly_stage(x, 64, C)
        if LOG2_C >= 8:
            x = _wht_butterfly_stage(x, 128, C)
        if LOG2_C >= 9:
            x = _wht_butterfly_stage(x, 256, C)
        if LOG2_C >= 10:
            x = _wht_butterfly_stage(x, 512, C)
        if LOG2_C >= 11:
            x = _wht_butterfly_stage(x, 1024, C)
        if LOG2_C >= 12:
            x = _wht_butterfly_stage(x, 2048, C)
        if LOG2_C >= 13:
            x = _wht_butterfly_stage(x, 4096, C)

        x = x * (1.0 / tl.sqrt(float(C)))

        s1 = tl.load(SIGNS1_ptr + tl.arange(0, C)).to(tl.float32)
        x = x * s1

        # --- Add residual base and store ---
        out_offs = tl.arange(0, C)
        out_mask = out_offs < OUT_DIM
        if HAS_RESIDUAL:
            base = tl.load(BASE_ptr + row * OUT_DIM + out_offs, mask=out_mask, other=0.0).to(tl.float32)
            x = x + base
        tl.store(OUT_ptr + row * OUT_DIM + out_offs, x, mask=out_mask)


class FusedWHTCompressor:
    """Fused WHT rotation + quantization + packing in single kernel launches.

    Eliminates intermediate tensors between rotation and quantization.
    Supports 2-bit, 3-bit, and 4-bit modes.
    """

    def __init__(self, hidden_dim: int, group_size: int, bits: int, seed: int, device: torch.device):
        self.hidden_dim = hidden_dim
        self.bits = bits
        self.padded_dim = _next_power_of_2(hidden_dim)
        self.log2_dim = int(math.log2(self.padded_dim))

        if self.padded_dim > 8192:
            raise ValueError(
                f"hidden_dim={hidden_dim} pads to {self.padded_dim} > 8192."
            )

        # Effective group size for padded dim
        from raylight.distributed_modules.tp_compress import _effective_group_size
        self.group_size = _effective_group_size(self.padded_dim, group_size)

        # Sign vectors (same generation as WHTRotation).  Always on CPU;
        # _ensure_device moves to CUDA lazily on first compress/decompress.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.signs1 = torch.where(
            torch.rand(self.padded_dim, generator=gen) > 0.5,
            torch.ones(self.padded_dim),
            -torch.ones(self.padded_dim),
        ).to(dtype=torch.float32)
        self.signs2 = torch.where(
            torch.rand(self.padded_dim, generator=gen) > 0.5,
            torch.ones(self.padded_dim),
            -torch.ones(self.padded_dim),
        ).to(dtype=torch.float32)

        self._device = torch.device("cpu")
        self.n_groups = self.padded_dim // self.group_size
        if bits == 2:
            self.packed_per_row = self.padded_dim // 4   # 4 values per byte
        elif bits == 3:
            self.packed_per_row = self.padded_dim * 3 // 8  # 8 values per 3 bytes
        else:
            self.packed_per_row = self.padded_dim // 2   # 2 values per byte

    def _ensure_device(self, device: torch.device) -> None:
        if self._device != device:
            self.signs1 = self.signs1.to(device=device)
            self.signs2 = self.signs2.to(device=device)
            self._device = device

    def compress(
        self, x: torch.Tensor,
        base: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Fused WHT rotate + quantize + pack.

        Args:
            x: (T, hidden_dim) input tensor.
            base: (T, hidden_dim) optional residual base. When provided,
                the kernel computes ``x - base`` in-register before rotation,
                eliminating a separate subtraction kernel launch.

        Returns:
            2-bit: (packed, scales, None)
            3-bit: (packed, scales, None)
            4-bit: (packed, scales, zero_points)
        """
        self._ensure_device(x.device)
        T = x.shape[0]
        C = x.shape[-1]

        has_residual = base is not None
        # base_ptr: pass x as dummy when no base (never read due to HAS_RESIDUAL=False)
        base_ptr = base if has_residual else x

        if C < self.padded_dim:
            x = torch.nn.functional.pad(x, (0, self.padded_dim - C))

        packed = torch.empty(T, self.packed_per_row, dtype=torch.uint8, device=x.device)
        scales = torch.empty(T, self.n_groups, dtype=torch.bfloat16, device=x.device)

        grid = (T,)
        warps = max(1, self.padded_dim // 256)

        if self.bits == 2:
            _fused_wht_compress_2bit_kernel[grid](
                x, packed, scales, self.signs1, self.signs2,
                base_ptr,
                T,
                C=self.padded_dim,
                LOG2_C=self.log2_dim,
                GS=self.group_size,
                HAS_RESIDUAL=has_residual,
                HIDDEN_DIM=self.hidden_dim,
                num_warps=warps,
            )
            return packed, scales, None
        elif self.bits == 3:
            _fused_wht_compress_3bit_kernel[grid](
                x, packed, scales, self.signs1, self.signs2,
                base_ptr,
                T,
                C=self.padded_dim,
                LOG2_C=self.log2_dim,
                GS=self.group_size,
                HAS_RESIDUAL=has_residual,
                HIDDEN_DIM=self.hidden_dim,
                num_warps=warps,
            )
            return packed, scales, None
        else:
            zp = torch.empty(T, self.n_groups, dtype=torch.bfloat16, device=x.device)
            _fused_wht_compress_4bit_kernel[grid](
                x, packed, scales, zp, self.signs1, self.signs2,
                base_ptr,
                T,
                C=self.padded_dim,
                LOG2_C=self.log2_dim,
                GS=self.group_size,
                HAS_RESIDUAL=has_residual,
                HIDDEN_DIM=self.hidden_dim,
                num_warps=warps,
            )
            return packed, scales, zp

    def decompress(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor] = None,
        base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fused unpack + dequant + inverse WHT.

        Args:
            packed: (T, packed_per_row) uint8
            scales: (T, n_groups) bf16
            zero_points: (T, n_groups) bf16 — required for 4-bit, ignored for 2-bit.
            base: (T, hidden_dim) optional residual base. When provided,
                the kernel adds ``base`` to the decompressed delta in-register
                after inverse WHT, eliminating a separate addition kernel launch.

        Returns:
            (T, hidden_dim) reconstructed tensor.
        """
        self._ensure_device(packed.device)
        T = packed.shape[0]

        has_residual = base is not None
        # Use the base tensor's dtype when available (avoids a redundant fp32
        # allocation that doubles peak VRAM).  Triton's tl.store handles the
        # implicit fp32→bf16/fp16 downcast automatically.
        if has_residual:
            out_dtype = base.dtype
        elif torch.cuda.is_bf16_supported():
            out_dtype = torch.bfloat16
        else:
            out_dtype = torch.float16
        out = torch.empty(T, self.hidden_dim, dtype=out_dtype, device=packed.device)
        base_ptr = base if has_residual else out

        grid = (T,)
        warps = max(1, self.padded_dim // 256)

        if self.bits == 2:
            _fused_wht_decompress_2bit_kernel[grid](
                packed, scales, out, self.signs1, self.signs2,
                base_ptr,
                T,
                C=self.padded_dim,
                LOG2_C=self.log2_dim,
                GS=self.group_size,
                OUT_DIM=self.hidden_dim,
                HAS_RESIDUAL=has_residual,
                num_warps=warps,
            )
        elif self.bits == 3:
            _fused_wht_decompress_3bit_kernel[grid](
                packed, scales, out, self.signs1, self.signs2,
                base_ptr,
                T,
                C=self.padded_dim,
                LOG2_C=self.log2_dim,
                GS=self.group_size,
                OUT_DIM=self.hidden_dim,
                HAS_RESIDUAL=has_residual,
                num_warps=warps,
            )
        else:
            _fused_wht_decompress_4bit_kernel[grid](
                packed, scales, zero_points, out, self.signs1, self.signs2,
                base_ptr,
                T,
                C=self.padded_dim,
                LOG2_C=self.log2_dim,
                GS=self.group_size,
                OUT_DIM=self.hidden_dim,
                HAS_RESIDUAL=has_residual,
                num_warps=warps,
            )

        return out


# =============================================================================
# PyTorch fallback (no Triton)
# =============================================================================


def _wht_pytorch(x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch Walsh-Hadamard Transform (fallback when Triton unavailable).

    Operates on the last dimension of x, which must be a power of 2.
    """
    C = x.shape[-1]
    h = 1
    while h < C:
        # Reshape to (..., C/(2h), 2, h) for butterfly
        shape = x.shape[:-1] + (C // (2 * h), 2, h)
        x = x.reshape(shape)
        a = x[..., 0, :]  # even
        b = x[..., 1, :]  # odd
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.reshape(x.shape[:-3] + (C,))
        h *= 2
    return x / math.sqrt(C)


class WHTRotationPyTorch:
    """Pure-PyTorch WHT rotation fallback (no Triton required).

    Same interface as WHTRotation but uses PyTorch ops instead of a
    Triton kernel.  Slower but works on any device.
    """

    def __init__(self, hidden_dim: int, seed: int, device: torch.device):
        self.hidden_dim = hidden_dim
        self.padded_dim = _next_power_of_2(hidden_dim)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.signs1 = torch.where(
            torch.rand(self.padded_dim, generator=gen) > 0.5,
            torch.ones(self.padded_dim),
            -torch.ones(self.padded_dim),
        )
        self.signs2 = torch.where(
            torch.rand(self.padded_dim, generator=gen) > 0.5,
            torch.ones(self.padded_dim),
            -torch.ones(self.padded_dim),
        )
        self._device = torch.device("cpu")

    def _ensure_device(self, device: torch.device) -> None:
        if self._device != device:
            self.signs1 = self.signs1.to(device=device)
            self.signs2 = self.signs2.to(device=device)
            self._device = device

    def rotate_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_device(x.device)
        C = x.shape[-1]
        x_2d = x.reshape(-1, C).float()
        if C < self.padded_dim:
            x_2d = torch.nn.functional.pad(x_2d, (0, self.padded_dim - C))
        x_2d = x_2d * self.signs1.float()
        x_2d = _wht_pytorch(x_2d)
        x_2d = x_2d * self.signs2.float()
        return x_2d

    def rotate_inverse(self, y: torch.Tensor) -> torch.Tensor:
        self._ensure_device(y.device)
        y_2d = y.reshape(-1, self.padded_dim).float()
        y_2d = y_2d * self.signs2.float()
        y_2d = _wht_pytorch(y_2d)
        y_2d = y_2d * self.signs1.float()
        if self.hidden_dim < self.padded_dim:
            y_2d = y_2d[:, :self.hidden_dim]
        return y_2d
