"""fp8_ampere — FP8-weight INT8-IMMA linear layer for Ampere GPUs.

Public surface
--------------
Fp8AmpereLinear        : Drop-in replacement for nn.Linear with FP8 weights.
TPFp8AmpereLinear      : TP-parallel FP8 linear (column / row sharding).
quantize_to_fp8_e4m3   : Quantise a BF16 weight to FP8 + compute INT8 scales.
compute_fp8_int8_scales: Compute INT8 scales from an existing FP8 checkpoint weight.
narrow_fp8_weight_for_tp: Slice FP8 weight + scale for tensor-parallel shards.
"""

from .fp8_ampere_linear import Fp8AmpereLinear
from .tp_fp8_ampere_linear import TPFp8AmpereLinear
from .packing import (
    compute_fp8_int8_scales,
    narrow_fp8_weight_for_tp,
    quantize_to_fp8_e4m3,
)

__all__ = [
    "Fp8AmpereLinear",
    "TPFp8AmpereLinear",
    "compute_fp8_int8_scales",
    "narrow_fp8_weight_for_tp",
    "quantize_to_fp8_e4m3",
]
