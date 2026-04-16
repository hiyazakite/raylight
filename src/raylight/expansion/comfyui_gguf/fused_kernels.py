"""Python binding for the GGUF fused CUDA kernels.

Provides three functions that map directly to the CUDA extension:

- ``cuda_dequantize(W, qtype, m, n, dtype=None)`` — standalone dequant
- ``cuda_mul_mat_vec(W, X, qtype, rows)`` — fused matvec (mmvq, batch ≤ few)
- ``cuda_mul_mat(W, X, qtype, rows)`` — fused GEMM (mmq, larger batches)

If the CUDA extension is not compiled, all functions are ``None`` and
``HAS_FUSED_GGUF_CUDA`` is ``False``.

Build the extension with:
    cd <raylight-root>
    make build-cuda   # or: python setup.py build_ext --inplace
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import gguf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the compiled extension
# ---------------------------------------------------------------------------
_C = None
try:
    from raylight.expansion.comfyui_gguf import _C_gguf as _C  # type: ignore[import-not-found]
except ImportError:
    # Fallback: maybe it was built in-tree and lives next to this file
    try:
        from . import _C_gguf as _C  # type: ignore[import-not-found,no-redef]
    except ImportError:
        pass

HAS_FUSED_GGUF_CUDA: bool = _C is not None

if HAS_FUSED_GGUF_CUDA:
    logger.info("GGUF fused CUDA kernels loaded successfully")
else:
    logger.debug(
        "GGUF fused CUDA kernels not available — "
        "falling back to PyTorch dequant. "
        "Build with: make build-cuda"
    )

# ---------------------------------------------------------------------------
# Quant types supported by each kernel path
# ---------------------------------------------------------------------------
# mmvq (matvec) supports legacy + K-quants + IQ types
_MMVQ_SUPPORTED: frozenset[int] = frozenset({
    2, 3, 6, 7, 8,           # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
    10, 11, 12, 13, 14,      # Q2_K .. Q6_K
    16, 17, 18, 19, 20,      # IQ2_XXS .. IQ4_NL
    21, 22, 23, 29,          # IQ3_S, IQ2_S, IQ4_XS, IQ1_M
})

# mmq (batched GEMM) supports legacy + K-quants only
_MMQ_SUPPORTED: frozenset[int] = frozenset({
    2, 3, 6, 7, 8,           # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
    10, 11, 12, 13, 14,      # Q2_K .. Q6_K
})

# dequantize-only supports everything mmvq supports (same underlying dequant)
_DEQUANT_SUPPORTED: frozenset[int] = _MMVQ_SUPPORTED


def _qtype_to_int(qtype: gguf.GGMLQuantizationType) -> int:
    """Convert gguf enum to the integer the CUDA kernel expects."""
    return qtype.value if hasattr(qtype, "value") else int(qtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cuda_dequantize(
    W: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    m: int,
    n: int,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Dequantize using CUDA kernel.  Returns None if not available."""
    if _C is None:
        return None
    type_int = _qtype_to_int(qtype)
    if type_int not in _DEQUANT_SUPPORTED:
        return None
    return _C.ggml_dequantize(W, type_int, m, n, dtype)


def cuda_mul_mat_vec(
    W: torch.Tensor,
    X: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    rows: int,
) -> Optional[torch.Tensor]:
    """Fused matvec (mmvq path).  Returns None if not available."""
    if _C is None:
        return None
    type_int = _qtype_to_int(qtype)
    if type_int not in _MMVQ_SUPPORTED:
        return None
    return _C.ggml_mul_mat_vec_a8(W, X, type_int, rows)


def cuda_mul_mat(
    W: torch.Tensor,
    X: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    rows: int,
) -> Optional[torch.Tensor]:
    """Fused batched GEMM (mmq path).  Returns None if not available."""
    if _C is None:
        return None
    type_int = _qtype_to_int(qtype)
    if type_int not in _MMQ_SUPPORTED:
        return None
    return _C.ggml_mul_mat_a8(W, X, type_int, rows)


def fused_ggml_gemm(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    tensor_type: gguf.GGMLQuantizationType,
    shard_shape: torch.Size,
    bias: Optional[torch.Tensor] = None,
    type_int: Optional[int] = None,
    scratch: Optional[torch.Tensor] = None,
    dequant_dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """High-level API: pick the best fused kernel for the given shapes.

    Returns the output tensor, or None if fused kernels are unavailable
    for this quant type / shape combination (caller should fall back).

    Parameters
    ----------
    x : (*, in_features) input activations
    quantized_weight : raw quantised bytes (flat uint8 buffer)
    tensor_type : GGML quantization enum
    shard_shape : logical (out_features, in_features) of the weight shard
    bias : optional bias vector
    type_int : pre-computed integer type code (avoids enum .value per call)
    scratch : optional pre-allocated [rows*cols] buffer for dequant (avoids alloc)
    dequant_dtype : optional dtype for dequantized weights. When set, x is
        temporarily cast so the CUDA kernel allocates DW in this precision
        (e.g. float16 to halve VRAM vs float32). Output is cast back to the
        original input dtype.
    """
    if _C is None:
        return None

    if type_int is None:
        type_int = _qtype_to_int(tensor_type)
    rows = shard_shape[0]  # output features

    # Cast x to dequant_dtype so the CUDA kernel uses it for DW allocation.
    orig_dtype = x.dtype
    if dequant_dtype is not None and dequant_dtype != x.dtype:
        x = x.to(dequant_dtype)
        if bias is not None:
            bias = bias.to(dequant_dtype)

    # Flatten leading dims for the kernel (expects 2-D input)
    needs_reshape = x.ndim != 2
    if needs_reshape:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
    else:
        x_2d = x
    batch = x_2d.shape[0]

    out = None
    if batch <= 4 and type_int in _MMVQ_SUPPORTED:
        # Small batch → matvec path (higher throughput per element)
        out = _C.ggml_mul_mat_vec_a8(quantized_weight, x_2d, type_int, rows)
        if out is not None and bias is not None:
            out = out + bias
    elif type_int in _DEQUANT_SUPPORTED:
        # Large batch → fused C++ dequant + cuBLAS GEMM (+ addmm bias).
        # Single C++ call: no Python between dequant and mm, uses addmm
        # for fused bias, and optionally reuses a scratch buffer.
        cols = shard_shape[1]
        out = _C.ggml_dequant_mm(
            quantized_weight, x_2d, type_int, rows, cols,
            bias, scratch,
        )
        # bias already fused into addmm — don't add again

    if out is None:
        return None

    # Restore leading dims
    if needs_reshape:
        out = out.reshape(*orig_shape[:-1], rows)

    # Cast back to original input dtype if we changed it for dequant
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)

    return out
