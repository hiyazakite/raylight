"""TP-parallel linear layer backed by GGML quantized weights.

Stores quantized byte shards and dequantizes on-the-fly during each forward
pass — maintaining the VRAM savings of GGML quantization across TP ranks.

Block-quantisation layout
-------------------------
GGML stores weights as 1-D raw bytes.  The logical weight matrix
``[out_features, in_features]`` is encoded as contiguous *blocks* in
row-major order:

    total_blocks = out_features × (in_features / block_size)
    raw_bytes    = total_blocks × type_size            (1-D)

Reshaping to ``[out_features, blocks_per_row, type_size]`` lets us
narrow either dimension independently (provided shard boundaries fall
on block_size multiples for the input-dim case).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Literal, Optional, TYPE_CHECKING

import gguf

from raylight.expansion.comfyui_gguf.dequant import (
    dequantize,
    dequantize_tensor,
    is_quantized,
    is_torch_compatible,
)
from raylight.distributed_modules.tensor_parallel import (
    TPLinear,
    TensorParallelState,
    all_reduce_tensor,
    gather_tensor_along_dim,
    get_tp_group,
    get_tp_size,
)

if TYPE_CHECKING:
    from raylight.expansion.comfyui_gguf.ops import GGMLTensor

# Map GGML native-float types to their torch dtype.
_GGML_NATIVE_DTYPES = {
    gguf.GGMLQuantizationType.F32: torch.float32,
    gguf.GGMLQuantizationType.F16: torch.float16,
    gguf.GGMLQuantizationType.BF16: torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Quantised byte-level narrowing
# ---------------------------------------------------------------------------

def narrow_ggml_weight(
    data: torch.Tensor,
    tensor_type: gguf.GGMLQuantizationType,
    tensor_shape: torch.Size,
    parallelism: str,
    tp_rank: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Size]:
    """Narrow quantised GGML byte data for one TP shard.

    Column-parallel → shards output dimension (contiguous byte slice).
    Row-parallel    → shards input dimension (block-aligned narrowing).

    Returns ``(shard_bytes, shard_logical_shape)``.
    """
    out_features = tensor_shape[0]
    in_features = tensor_shape[1]
    block_size, type_size = gguf.GGML_QUANT_SIZES[tensor_type]
    blocks_per_row = in_features // block_size

    # Strip GGMLTensor subclass to work with raw bytes only.
    # GGMLTensor overrides .shape → logical dims which breaks byte math.
    if hasattr(data, "as_subclass"):
        data = data.as_subclass(torch.Tensor)
    raw = data.view(torch.uint8).reshape(-1)  # flatten to 1-D bytes

    if parallelism == "column":
        # Output-dim sharding — contiguous byte range.
        shard_rows = out_features // tp_size
        offset_rows = tp_rank * shard_rows
        bytes_per_row = blocks_per_row * type_size
        start = offset_rows * bytes_per_row
        length = shard_rows * bytes_per_row
        shard_data = raw[start : start + length].clone()  # clone detaches from mmap
        shard_shape = torch.Size((shard_rows, in_features))
    else:
        # Input-dim sharding — extract complete blocks for this shard.
        shard_cols = in_features // tp_size
        if shard_cols % block_size != 0:
            raise ValueError(
                f"GGUF row-parallel TP requires in_features/tp_size "
                f"({shard_cols}) to be divisible by block_size "
                f"({block_size}).  in_features={in_features}, "
                f"tp_size={tp_size}"
            )
        shard_blocks = shard_cols // block_size
        offset_blocks = tp_rank * shard_blocks

        # 1-D → [out_features, blocks_per_row, type_size]
        total_blocks = out_features * blocks_per_row
        blocks_3d = raw.reshape(total_blocks, type_size).reshape(
            out_features, blocks_per_row, type_size,
        )
        shard_3d = blocks_3d[:, offset_blocks : offset_blocks + shard_blocks, :]
        shard_data = shard_3d.contiguous().reshape(-1).clone()  # clone detaches from mmap
        shard_shape = torch.Size((out_features, shard_cols))

    return shard_data, shard_shape


# ---------------------------------------------------------------------------
# TPGGMLLinear
# ---------------------------------------------------------------------------

class TPGGMLLinear(nn.Module):
    """TP-parallel linear backed by quantised GGML weights.

    The sharded quantised weight is stored as a raw-byte registered buffer.
    Each ``forward()`` call dequantises the shard into the compute dtype,
    performs the local matmul, and runs the appropriate TP collective
    (gather for column-parallel, all-reduce for row-parallel).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parallelism: Literal["column", "row"] = "column",
        gather_output: bool = False,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        compressor=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallelism = parallelism
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self._tp_group = tp_group
        self.compressor = compressor

        _tp_size = (
            dist.get_world_size(tp_group) if tp_group is not None
            else get_tp_size()
        )

        if parallelism == "column":
            self.local_out_features = out_features // _tp_size
            self.local_in_features = in_features
        else:
            self.local_out_features = out_features
            self.local_in_features = in_features // _tp_size

        # Quantised weight — raw bytes, set by load_quantized_weight().
        self.register_buffer("quantized_weight", None)
        self.tensor_type: Optional[gguf.GGMLQuantizationType] = None
        self.shard_shape: Optional[torch.Size] = None

        # User-selectable dequant precision (mirrors GGMLOps dequant_dtype).
        # None = native quant precision, "target" = match input dtype.
        self.dequant_dtype = None

        # Zero-element weight so state_dict() exposes "*.weight" keys.
        # ComfyUI's model_lora_keys_unet() and add_patches() both require
        # a .weight entry to build the LoRA key_map and register patches.
        # A zero-element tensor costs 0 bytes, avoids module_size inflation,
        # and works with .to() without the meta-device issues.
        # The real data lives in quantized_weight (dequantised per-forward).
        self.weight = nn.Parameter(
            torch.empty(0),
            requires_grad=False,
        )

        # Mark as comfy_cast_weights so RaylightModelPatcher.load() hooks
        # up weight_function / bias_function for lowvram LoRA patching.
        self.comfy_cast_weights = True

        # Bias (typically unquantised)
        if bias:
            if parallelism == "column":
                self.bias = nn.Parameter(
                    torch.empty(self.local_out_features), requires_grad=False,
                )
            else:
                self.bias = nn.Parameter(
                    torch.empty(out_features), requires_grad=False,
                )
        else:
            self.register_parameter("bias", None)

    # ── TP group helpers ─────────────────────────────────────

    @property
    def _resolved_group(self) -> Optional[dist.ProcessGroup]:
        return self._tp_group if self._tp_group is not None else get_tp_group()

    @property
    def _tp_rank(self) -> int:
        grp = self._resolved_group
        return dist.get_rank(grp) if grp is not None else 0

    @property
    def _tp_size_runtime(self) -> int:
        grp = self._resolved_group
        return dist.get_world_size(grp) if grp is not None else 1

    # ── Weight loading ───────────────────────────────────────

    def load_quantized_weight(self, ggml_tensor: "GGMLTensor") -> None:
        """Shard and store a full quantised weight for this rank."""
        tensor_type = ggml_tensor.tensor_type
        tensor_shape = ggml_tensor.tensor_shape

        # Strip GGMLTensor subclass so narrow_ggml_weight receives plain bytes.
        raw_data = ggml_tensor
        if hasattr(ggml_tensor, "as_subclass"):
            raw_data = ggml_tensor.as_subclass(torch.Tensor)

        shard_data, shard_shape = narrow_ggml_weight(
            raw_data,
            tensor_type,
            tensor_shape,
            self.parallelism,
            self._tp_rank,
            self._tp_size_runtime,
        )

        assert shard_data.numel() > 0, (
            f"TPGGMLLinear: empty shard for {tensor_shape} "
            f"tensor_type={tensor_type.name} "
            f"parallelism={self.parallelism} rank={self._tp_rank}/{self._tp_size_runtime}"
        )

        # Use register_buffer to ensure proper buffer tracking.
        self.register_buffer("quantized_weight", shard_data)
        self.tensor_type = tensor_type
        self.shard_shape = shard_shape

    def load_bias(self, bias_tensor: torch.Tensor) -> None:
        """Load and optionally shard bias."""
        if self.bias is None:
            return
        if self.parallelism == "column":
            shard_size = self.local_out_features
            if bias_tensor.shape[0] != shard_size:
                tp_rank = self._tp_rank
                bias_tensor = bias_tensor.narrow(
                    0, tp_rank * shard_size, shard_size,
                )
        self.bias.data.copy_(bias_tensor)

    # ── Forward ──────────────────────────────────────────────

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Empty-input fast path (shape inference, empty microbatch)
        if x.shape[0] == 0:
            out_dim = self.local_out_features if self.parallelism == "column" else self.out_features
            return torch.empty(*x.shape[:-1], out_dim, dtype=x.dtype, device=x.device)

        # 1. Dequantise shard on-the-fly.
        #    Honour user-selected dequant_dtype (from GGUF loader node).
        #    "target" → use input dtype for internal scale arithmetic.
        #    None     → native quant precision (float16 for K-quants).
        #    explicit → that dtype (e.g. torch.float32).
        qw = self.quantized_weight
        if hasattr(qw, "as_subclass"):
            qw = qw.as_subclass(torch.Tensor)
        _dq = self.dequant_dtype
        _internal_dtype = x.dtype if _dq == "target" else _dq
        weight = dequantize(
            qw,
            self.tensor_type,
            self.shard_shape,
            dtype=_internal_dtype,
        )
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)

        # 2. Input / bias selection
        if self.parallelism == "row":
            if not self.input_is_parallel:
                x = x.chunk(self._tp_size_runtime, dim=-1)[self._tp_rank].contiguous()
            bias = None if self._tp_rank > 0 else self.bias
        else:
            bias = self.bias

        if bias is not None:
            bias = bias.to(x.dtype)

        # 3. Apply LoRA patches (if any).
        #    ComfyUI attaches weight_function via LowVramPatch when LoRA
        #    patches are registered.  Apply them to the dequantised weight
        #    before the matmul so the LoRA delta is fused into the output.
        wf = getattr(self, "weight_function", None)
        if wf:
            for fn in wf:
                weight = fn(weight)

        # 4. Local matmul
        out = F.linear(x, weight, bias)

        # 5. TP collective
        if self.parallelism == "column":
            if self.gather_output and self._tp_size_runtime > 1:
                out = gather_tensor_along_dim(
                    out, dim=-1, full_size=self.out_features,
                )
        else:
            if self.reduce_results and self._tp_size_runtime > 1:
                if self.compressor is not None:
                    out = self.compressor.compressed_all_reduce(
                        out, group=self._resolved_group,
                    )
                else:
                    out = all_reduce_tensor(out)

        return out

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        parallelism: Literal["column", "row"] = "column",
        tp_group: Optional[dist.ProcessGroup] = None,
        gather_output: bool = False,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        compressor=None,
    ) -> "TPGGMLLinear":
        """Create a ``TPGGMLLinear`` from an existing GGML-quantised linear."""
        weight = linear.weight
        tensor_shape = weight.tensor_shape
        in_features = tensor_shape[1] if len(tensor_shape) >= 2 else linear.in_features
        out_features = tensor_shape[0] if len(tensor_shape) >= 2 else linear.out_features

        tp_ggml = cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias is not None,
            parallelism=parallelism,
            gather_output=gather_output,
            input_is_parallel=input_is_parallel,
            reduce_results=reduce_results,
            tp_group=tp_group,
            compressor=compressor,
        )

        # Carry over dequant_dtype from the source GGMLOps.Linear
        tp_ggml.dequant_dtype = getattr(linear, "dequant_dtype", None)

        # Shard and store quantised weight
        tp_ggml.load_quantized_weight(weight)

        # Handle bias (may itself be quantised for BF16 GGUF)
        if linear.bias is not None:
            bias_data = linear.bias
            if is_quantized(bias_data):
                bias_data = dequantize_tensor(bias_data, dtype=torch.float32)
            tp_ggml.load_bias(bias_data.data if hasattr(bias_data, "data") else bias_data)

        return tp_ggml


# ---------------------------------------------------------------------------
# Factory: make_tp_linear  –  auto-selects TPLinear vs TPGGMLLinear
# ---------------------------------------------------------------------------

def make_tp_linear(
    src_linear: nn.Module,
    parallelism: Literal["column", "row"] = "column",
    tp_group: Optional[dist.ProcessGroup] = None,
    gather_output: bool = False,
    input_is_parallel: bool = True,
    reduce_results: bool = True,
    structure_only: bool = False,
    compressor=None,
) -> nn.Module:
    """Create a TP-parallel linear from an existing ``nn.Linear``.

    Automatically selects ``TPGGMLLinear`` when the source weight is
    GGML-quantised, otherwise falls back to ``TPLinear``.

    When *structure_only* is True (streaming safetensors path), a
    ``TPLinear`` with meta-device weights is returned regardless of
    quantisation — Phase 4 will stream real weights later.
    """
    weight = getattr(src_linear, "weight", None)
    has_bias = src_linear.bias is not None

    # GGUF quantised weight → TPGGMLLinear  (structure_only is N/A for
    # GGUF because the mmap-backed weights are already present).
    #
    # Block-quantised GGML types (Q4_K, Q6_K, …) require row-aligned
    # blocks: in_features must be a multiple of block_size so that
    # output-dim sharding doesn't split a block.  When in_features <
    # block_size (e.g. patchify_proj [4096, 128] with Q6_K block_size=256),
    # blocks span multiple rows and quantised sharding is impossible.
    # Fall back to dequantise → TPLinear for those layers.
    _can_shard_quantized = False
    if not structure_only and weight is not None and is_quantized(weight):
        _qtype = getattr(weight, "tensor_type", None)
        _tshape = getattr(weight, "tensor_shape", None)
        if _qtype is not None and _tshape is not None and len(_tshape) >= 2:
            _block_size, _ = gguf.GGML_QUANT_SIZES[_qtype]
            _in_f = _tshape[1]
            # Native float types (BF16, F16, F32) have block_size=1 and
            # are already stored in their logical shape by the GGUF loader.
            # They don't benefit from quantised byte-sharding and the 2-D
            # storage layout breaks narrow_ggml_weight's 1-D byte indexing.
            # Route them through the dequantise → TPLinear fallback.
            _is_native_float = _block_size <= 1
            _can_shard_quantized = (not _is_native_float) and (_in_f % _block_size == 0)

    if _can_shard_quantized:
        return TPGGMLLinear.from_linear(
            src_linear,
            parallelism=parallelism,
            gather_output=gather_output,
            input_is_parallel=input_is_parallel,
            reduce_results=reduce_results,
            tp_group=tp_group,
            compressor=compressor,
        )

    # Standard / dequantised-fallback path → TPLinear.
    # If weight is a GGMLTensor (quantised but not block-shardable),
    # dequantise it to a dense float tensor first.
    # Honour user-selected dequant_dtype from the GGUF loader node.
    if weight is not None and is_quantized(weight):
        _fallback_dtype = _GGML_NATIVE_DTYPES.get(_qtype, torch.float16)
        _user_dequant = getattr(src_linear, "dequant_dtype", None)
        if _user_dequant == "target":
            _dq_arg = _fallback_dtype
        elif _user_dequant is not None:
            _dq_arg = _user_dequant
        else:
            _dq_arg = None
        weight = dequantize_tensor(weight, dtype=_fallback_dtype, dequant_dtype=_dq_arg)

    in_f = weight.shape[1] if weight is not None else src_linear.in_features
    out_f = weight.shape[0] if weight is not None else src_linear.out_features

    tp_lin = TPLinear(
        in_features=in_f,
        out_features=out_f,
        bias=has_bias,
        parallelism=parallelism,
        gather_output=gather_output,
        input_is_parallel=input_is_parallel,
        reduce_results=reduce_results,
        tp_group=tp_group,
        dtype=weight.dtype if weight is not None else None,
        device="meta" if structure_only else (
            weight.device if weight is not None else None
        ),
        compressor=compressor,
    )

    if not structure_only and weight is not None:
        tp_lin.weight_loader(tp_lin.weight, weight.data)
        if has_bias and tp_lin.bias is not None:
            bias_data = src_linear.bias
            if is_quantized(bias_data):
                bias_data = dequantize_tensor(bias_data, dtype=torch.float32)
            if parallelism == "column":
                tp_lin.weight_loader(tp_lin.bias, bias_data.data)
            else:
                tp_lin.bias.data.copy_(bias_data.data)

    # Preserve LoRA attributes
    if not structure_only:
        for attr in ("lora_A", "lora_B", "lora_alpha"):
            if hasattr(src_linear, attr):
                setattr(tp_lin, attr, getattr(src_linear, attr))

    # Preserve INT8 scale attributes
    if not structure_only:
        if hasattr(src_linear, "weight_scale"):
            scale = src_linear.weight_scale
            if (
                isinstance(scale, torch.Tensor)
                and parallelism == "column"
                and scale.numel() > tp_lin.local_out_features
            ):
                tp_rank = tp_lin._tp_rank
                scale = scale.narrow(
                    0,
                    tp_rank * tp_lin.local_out_features,
                    tp_lin.local_out_features,
                )
            setattr(tp_lin, "weight_scale", scale)
        if hasattr(src_linear, "scale"):
            setattr(tp_lin, "scale", getattr(src_linear, "scale"))

    return tp_lin
