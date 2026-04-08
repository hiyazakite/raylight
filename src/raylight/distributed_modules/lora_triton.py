"""Triton LoRA kernels for activation-space LoRA application.

Implements fused shrink → expand (``x @ A^T @ B^T * scale``) in a single
Triton launch with fp32 intermediate accumulation (matching vLLM's precision
pattern).  Falls back to two ``F.linear`` calls when Triton is unavailable.

Design rationale (vLLM/sglang pattern):
  - Base model weights are **never modified** — no backup dict, no weight copy.
  - LoRA delta is applied additively in activation space:
      ``out = base_linear(x) + triton_lora(x, A, B, scale)``
  - LoRA A/B matrices are stored as plain attributes on the module.

Usage::

    from raylight.distributed_modules.lora_triton import lora_forward, clear_lora

    # Attach (done once per LoRA apply):
    module.lora_A = A_matrix   # [rank, in_features]
    module.lora_B = B_matrix   # [out_features, rank]
    module.lora_scale = alpha / rank * strength

    # Forward (inside TPLinear.forward):
    lora_delta = lora_forward(x, module.lora_A, module.lora_B, module.lora_scale)
    out = out + lora_delta

    # Clear (on LoRA reset):
    clear_lora(module)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = [
    "lora_forward", "clear_lora", "attach_lora", "attach_delta",
    "apply_dora", "apply_dora_with_delta", "HAS_TRITON_LORA",
]

HAS_TRITON_LORA: bool = _HAS_TRITON


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _lora_shrink_expand_kernel(
        # Pointers
        X_ptr,          # [M, K]  — input activations
        A_ptr,          # [R, K]  — LoRA down (shrink) matrix
        B_ptr,          # [N, R]  — LoRA up (expand) matrix
        OUT_ptr,        # [M, N]  — output (additive delta)
        # Dimensions
        M,              # batch (num tokens × seq)
        K: tl.constexpr,  # in_features
        R: tl.constexpr,  # LoRA rank
        N: tl.constexpr,  # out_features
        # Strides (in elements)
        stride_xm, stride_xk,
        stride_ar, stride_ak,
        stride_bn, stride_br,
        stride_om, stride_on,
        # Scaling
        SCALE,
        # Meta
        BLOCK_M: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused LoRA: OUT[m, n] = scale * Σ_r (Σ_k X[m,k] * A[r,k]) * B[n,r]

        Two-phase matmul fused into one kernel:
          Phase 1 (shrink):  mid[m, r] = X[m, :] @ A[r, :]^T    (fp32 accum)
          Phase 2 (expand):  out[m, n] = mid[m, :] @ B[n, :]^T   (fp32 accum)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Offsets for this tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M

        # Phase 1: shrink — compute mid[BLOCK_M, R] = X[BLOCK_M, K] @ A^T[K, R]
        # We tile over K in BLOCK_K chunks, accumulate full R at once (R is small).
        offs_r = tl.arange(0, BLOCK_R)
        mid = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load X tile [BLOCK_M, BLOCK_K]
            x = tl.load(
                X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            # Load A tile [BLOCK_R, BLOCK_K] — A is [R, K], we want A^T
            a = tl.load(
                A_ptr + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak,
                mask=(offs_r[:, None] < R) & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            # mid += X @ A^T  →  [BLOCK_M, BLOCK_R]
            mid += tl.dot(x, tl.trans(a))

        # Phase 2: expand — out[BLOCK_M, BLOCK_N] = mid[BLOCK_M, R] @ B^T[R, BLOCK_N]
        # B is [N, R], we want B^T = [R, N].  Tile over R in BLOCK_R chunks
        # (typically R fits in one tile since ranks are ≤128).
        mask_n = offs_n < N
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Load B tile [BLOCK_N, BLOCK_R] — B is [N, R]
        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br,
            mask=mask_n[:, None] & (offs_r[None, :] < R),
            other=0.0,
        ).to(tl.float32)

        # acc = mid @ B^T  →  [BLOCK_M, BLOCK_N]
        acc = tl.dot(mid, tl.trans(b))

        # Apply scale and store
        acc = acc * SCALE
        tl.store(
            OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(tl.float16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


# ---------------------------------------------------------------------------
# Python dispatch
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2, minimum 16."""
    n = max(n, 16)
    return 1 << (n - 1).bit_length()


def _triton_lora_forward(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused LoRA via Triton: ``scale * (x @ A^T) @ B^T``.

    Args:
        x: Input activations [M, K] (contiguous, fp16/bf16).
        lora_a: Down-projection [R, K].
        lora_b: Up-projection [N, R].
        scale: Combined scaling factor (alpha / rank * strength).

    Returns:
        LoRA delta [M, N] in x.dtype.
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    M, K = x.shape
    R, K_a = lora_a.shape
    N, R_b = lora_b.shape
    assert K == K_a and R == R_b, (
        f"Shape mismatch: x=[{M},{K}], A=[{R},{K_a}], B=[{N},{R_b}]"
    )

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Block sizes — R is typically small (4–128), fits in one tile.
    BLOCK_M = min(64, _next_power_of_2(M))
    BLOCK_R = _next_power_of_2(R)
    BLOCK_K = min(128, _next_power_of_2(K))
    BLOCK_N = min(64, _next_power_of_2(N))

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _lora_shrink_expand_kernel[grid](
        x, lora_a, lora_b, out,
        M, K, R, N,
        x.stride(0), x.stride(1),
        lora_a.stride(0), lora_a.stride(1),
        lora_b.stride(0), lora_b.stride(1),
        out.stride(0), out.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    return out


def _torch_lora_forward(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference LoRA: ``scale * (x @ A^T) @ B^T`` using two F.linear calls."""
    mid = F.linear(x, lora_a)           # [M, R]
    out = F.linear(mid, lora_b)          # [M, N]
    if scale != 1.0:
        out = out * scale
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lora_forward(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float = 1.0,
    *,
    force_triton: bool = False,
) -> torch.Tensor:
    """Compute LoRA delta ``scale * (x @ A^T) @ B^T``.

    Dispatch strategy:
      - cuBLAS (two ``F.linear`` calls) is used by default — it is faster
        than the fused Triton kernel for typical LoRA shapes (rank ≤ 128).
      - The fused Triton kernel is available via ``force_triton=True`` and
        will become the default path when multi-LoRA batching is added
        (where fusing avoids materialising per-LoRA intermediate buffers).

    Args:
        x: Activations [..., in_features].
        lora_a: Down-projection [rank, in_features].
        lora_b: Up-projection [out_features, rank].
        scale: Combined ``(alpha / rank) * strength``.
        force_triton: Use fused Triton kernel regardless of heuristic.

    Returns:
        LoRA delta [..., out_features] in x.dtype.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()

    use_triton = force_triton and _HAS_TRITON and x_2d.is_cuda and x_2d.shape[0] > 0

    if use_triton:
        a = lora_a.to(device=x_2d.device, dtype=x_2d.dtype, non_blocking=True).contiguous()
        b = lora_b.to(device=x_2d.device, dtype=x_2d.dtype, non_blocking=True).contiguous()
        delta = _triton_lora_forward(x_2d, a, b, scale)
    else:
        a = lora_a.to(device=x_2d.device, dtype=x_2d.dtype, non_blocking=True)
        b = lora_b.to(device=x_2d.device, dtype=x_2d.dtype, non_blocking=True)
        delta = _torch_lora_forward(x_2d, a, b, scale)

    return delta.reshape(*orig_shape[:-1], delta.shape[-1])


def attach_lora(
    module: torch.nn.Module,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    dora_scale: torch.Tensor | None = None,
) -> None:
    """Attach LoRA matrices to a module for activation-space application.

    These attributes are read by ``TPLinear.forward()`` and
    ``lora_forward()`` during the forward pass.

    When *dora_scale* is provided the forward path applies DoRA
    normalization using the module's weight as a read-only reference
    (Option B — no weight copy).
    """
    module.lora_A = lora_a
    module.lora_B = lora_b
    module.lora_scale = scale
    module.lora_dora_scale = dora_scale
    # Clear the old lora_alpha to avoid double-scaling in TPLinear.forward
    module.lora_alpha = None


def clear_lora(module: torch.nn.Module) -> None:
    """Remove LoRA attributes from a module."""
    for attr in (
        "lora_A", "lora_B", "lora_scale", "lora_alpha",
        "lora_dora_scale", "lora_delta",
    ):
        if hasattr(module, attr):
            setattr(module, attr, None)


def attach_delta(
    module: torch.nn.Module,
    delta: torch.Tensor,
    dora_scale: torch.Tensor | None = None,
) -> None:
    """Attach a full-rank weight delta for activation-space application.

    Used for LoHa / LoKr adapters where the weight delta is a full
    ``[out_features, in_features]`` matrix (not low-rank decomposed).

    The forward hook applies this as ``F.linear(x, delta)``.
    """
    module.lora_delta = delta
    module.lora_dora_scale = dora_scale
    # Clear low-rank attrs to avoid conflicts
    module.lora_A = None
    module.lora_B = None
    module.lora_scale = None
    module.lora_alpha = None


# ---------------------------------------------------------------------------
# DoRA (Weight-Decomposed Low-Rank Adaptation)
# ---------------------------------------------------------------------------

def apply_dora(
    base_out: torch.Tensor,
    lora_delta: torch.Tensor,
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply DoRA normalization in activation space.

    DoRA in weight-space:
        ``W_new = dora_scale * (W + α·Δ) / ‖W + α·Δ‖``

    In activation-space (Option B — read-only weight reference):
        ``out = dora_scale / ‖W + α·Δ‖ · (W·x + α·Δ·x)``
             = ``norm_factor · (base_out + lora_delta)``

    Where ``lora_delta`` already has α baked in from ``_extract_lora_ab``.

    When ``strength != 1``:
        ``out = base_out + strength · (dora_out - base_out)``

    Args:
        base_out: Output from the base linear ``W·x`` [..., out_features].
        lora_delta: LoRA delta ``α·Δ·x`` [..., out_features].
        weight: The base weight tensor [out_features, in_features] (read-only).
        dora_scale: DoRA magnitude vector [out_features].
        strength: LoRA application strength (default 1.0).

    Returns:
        DoRA-normalized output [..., out_features].
    """
    dtype = base_out.dtype

    # Compute ‖W + α·Δ‖ per output row.
    # lora_A has scale baked in (alpha/rank * strength), but for norm
    # computation we need the weight-space Δ.  Since lora_delta = Δ·x
    # (activation-space), we can't recover Δ directly.  Instead we compute
    # the LoRA diff in weight space: Δ_w = B @ A (already stored as attrs).
    # But that's expensive.  Cheaper: use the module's A/B directly.
    #
    # Actually, the simplest correct approach: compute the weight-space
    # perturbation norm once.  This is O(out × in) not O(tokens × out × in).
    # For a 4096×4096 weight this is 16M flops — negligible vs the matmul.

    # Reconstruct weight-space delta: Δ_w = B @ A  (same as lora_diff in weight-space code)
    # Note: lora_delta = (A_scaled) @ x through lora_forward, where A_scaled = A * (alpha/rank * strength)
    # So B @ A_scaled = weight-space lora_diff with scale baked in.
    # We need: W + Δ_w  for norm computation.
    # But we don't have A/B here — they're on the module.
    # Instead, take the approach from weight_decompose: compute norm of (W + alpha*lora_diff)
    # We can compute this via the weight + B@A path.
    #
    # Simpler: pass through as a pre-computed column norm.  But that requires
    # the weight-space diff.  Let's just compute B@A here — it's small.
    #
    # REVISED APPROACH: The cleanest correct path is:
    # 1. At attach time, precompute the column norms of (W + B@A).
    # 2. Store as lora_dora_norm on the module.
    # This avoids recomputing B@A on every forward.
    #
    # BUT we don't want to couple attach_lora to the weight.  So let's just
    # compute it here lazily on first call.  The cost is O(out*in) once per
    # forward — same order as the base matmul already performed.
    #
    # ACTUALLY — the simplest and fully correct approach for the first impl:
    # We have `weight` (read-only ref), and `lora_delta` is already the
    # activation-space result.  We need the weight-space norm.
    # But we DON'T need B@A — we need ‖row_i(W + B@A)‖.
    # We can't get B@A without the matrices themselves.
    #
    # FINAL APPROACH: Accept A/B as optional params, or compute norm
    # outside this function.  For cleanliness, let's have the caller
    # (hook/TPLinear.forward) provide the norm or the A/B.
    #
    # Actually, re-reading weight_decompose more carefully:
    #   weight_calc = weight + function(lora_diff)  (function is identity by default)
    #   weight_norm = weight_calc.reshape(weight_calc.shape[0], -1).norm(dim=1, ...)
    #   weight_calc *= (dora_scale / weight_norm)
    #
    # So it's norm of EACH ROW of (W + Δ).  This requires knowing Δ in weight space.
    # We MUST compute B @ A.  Let's just do it here.

    # SKIP all that complexity — this function will be called with the
    # precomputed column norm.  See the revised signature below.

    # NOTE: This function should not be called directly with raw weight.
    # See _apply_dora_with_weight() below.
    raise NotImplementedError("Use _apply_dora_with_weight() instead")


def _compute_dora_norm_factor(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    dora_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute DoRA per-row norm factor: ``dora_scale / ‖W + B@A‖``.

    Cost: O(out × rank) + O(out × in) for norm — negligible vs matmul.
    The result has shape ``[out_features]`` (or ``[1, out_features]`` for
    broadcasting with [..., out_features] activations).

    Args:
        weight: Base weight [out_features, in_features].
        lora_a: Scaled down-projection [rank, in_features] (scale baked in).
        lora_b: Up-projection [out_features, rank].
        dora_scale: DoRA magnitude [out_features].
    """
    # Reconstruct weight-space delta: B @ A  →  [out, in]
    lora_diff = torch.mm(
        lora_b.to(device=weight.device, dtype=torch.float32).flatten(start_dim=1),
        lora_a.to(device=weight.device, dtype=torch.float32).flatten(start_dim=1),
    )

    # W + Δ in float32 for numerical stability
    w_plus_delta = weight.to(dtype=torch.float32).flatten(start_dim=1) + lora_diff

    # Per-row L2 norm: [out_features]
    row_norm = w_plus_delta.norm(dim=1, keepdim=False)
    row_norm = row_norm + torch.finfo(torch.float32).eps

    # dora_scale / norm → [out_features]
    ds = dora_scale.to(device=weight.device, dtype=torch.float32).flatten()
    return (ds / row_norm)


def apply_dora(
    base_out: torch.Tensor,
    lora_delta: torch.Tensor,
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply DoRA normalization in activation space.

    DoRA weight-space:
        ``W_new = dora_scale · (W + Δ) / ‖W + Δ‖``

    Activation-space equivalent:
        ``out = (dora_scale / ‖W + Δ‖) · (base_out + lora_delta)``

    When ``strength != 1``:
        ``out = base_out + strength · (dora_out - base_out)``

    Args:
        base_out: ``W·x`` [..., out_features].
        lora_delta: ``Δ·x`` [..., out_features] (scale already baked in).
        weight: Base weight [out, in] (read-only).
        dora_scale: DoRA magnitude vector [out_features].
        lora_a: Down-projection [rank, in] (needed for norm computation).
        lora_b: Up-projection [out, rank] (needed for norm computation).
        strength: Application strength (default 1.0).
    """
    dtype = base_out.dtype

    # Compute per-row norm factor: dora_scale / ‖W + B@A‖
    norm_factor = _compute_dora_norm_factor(weight, lora_a, lora_b, dora_scale)

    # Broadcast to activation shape: [..., out_features]
    nf = norm_factor.to(dtype=dtype)
    # Reshape for broadcasting: [1, 1, ..., out_features]
    for _ in range(base_out.ndim - 1):
        nf = nf.unsqueeze(0)

    # DoRA output: norm_factor · (base_out + lora_delta)
    combined = base_out + lora_delta.to(dtype=dtype)
    dora_out = nf * combined

    if strength != 1.0:
        return base_out + strength * (dora_out - base_out)
    return dora_out


def _compute_dora_norm_factor_from_delta(
    weight: torch.Tensor,
    lora_delta_ws: torch.Tensor,
    dora_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute DoRA norm factor from a pre-computed weight-space delta.

    Same as ``_compute_dora_norm_factor`` but takes the full ``[out, in]``
    delta directly instead of reconstructing it from ``B @ A``.
    """
    w_plus_delta = (
        weight.to(dtype=torch.float32).flatten(start_dim=1)
        + lora_delta_ws.to(device=weight.device, dtype=torch.float32).flatten(start_dim=1)
    )
    row_norm = w_plus_delta.norm(dim=1, keepdim=False)
    row_norm = row_norm + torch.finfo(torch.float32).eps
    ds = dora_scale.to(device=weight.device, dtype=torch.float32).flatten()
    return ds / row_norm


def apply_dora_with_delta(
    base_out: torch.Tensor,
    act_delta: torch.Tensor,
    weight: torch.Tensor,
    dora_scale: torch.Tensor,
    lora_delta: torch.Tensor | None = None,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply DoRA normalization for full-delta adapters (LoHa / LoKr).

    Like ``apply_dora`` but takes the weight-space delta ``lora_delta``
    directly (no B @ A reconstruction needed).

    Args:
        base_out: ``W·x`` [..., out_features].
        act_delta: ``Δ·x`` [..., out_features] (delta applied in activation space).
        weight: Base weight [out, in] (read-only).
        dora_scale: DoRA magnitude vector [out_features].
        lora_delta: Weight-space delta [out, in] (scale baked in).
        strength: Application strength (default 1.0).
    """
    dtype = base_out.dtype

    norm_factor = _compute_dora_norm_factor_from_delta(weight, lora_delta, dora_scale)

    nf = norm_factor.to(dtype=dtype)
    for _ in range(base_out.ndim - 1):
        nf = nf.unsqueeze(0)

    combined = base_out + act_delta.to(dtype=dtype)
    dora_out = nf * combined

    if strength != 1.0:
        return base_out + strength * (dora_out - base_out)
    return dora_out
