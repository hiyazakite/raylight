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
import logging

from raylight.expansion.comfyui_gguf.dequant import (
    dequantize,
    dequantize_tensor,
    is_quantized,
    is_torch_compatible,
)
from raylight.expansion.comfyui_gguf.fused_kernels import (
    HAS_FUSED_GGUF_CUDA,
    fused_ggml_gemm,
)

logger = logging.getLogger(__name__)
_fallback_logged = False

# Shared scratch buffer for CUDA dequant — only one layer executes at a time,
# so a single buffer (sized to the largest layer) avoids per-module allocation
# that would duplicate the entire model's weights in VRAM.
_shared_dequant_scratch: Optional[torch.Tensor] = None
_scratch_pinned_dtype: Optional[torch.dtype] = None  # set by warmup; inhibits realloc on dtype drift


def warmup_scratch_buffer(
    model,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> int:
    """Pre-allocate the shared dequant scratch to the largest shard size.

    Call this **once** after model loading and TP patching completes (i.e.
    after quantized weights have been loaded into TPGGMLLinear modules and
    moved to the target device).

    *model* can be an ``nn.Module`` or a ComfyUI ``ModelPatcher`` wrapper
    (the inner ``nn.Module`` is extracted automatically via
    ``model.model.diffusion_model``).

    Returns the buffer size in elements (0 if no TPGGMLLinear modules found).
    """
    global _shared_dequant_scratch, _scratch_pinned_dtype

    # Unwrap ComfyUI ModelPatcher → BaseModel → diffusion_model
    root = model
    if not isinstance(root, nn.Module):
        root = getattr(root, "model", root)
    if not isinstance(root, nn.Module):
        root = getattr(root, "diffusion_model", root)
    if not isinstance(root, nn.Module):
        logger.warning("warmup_scratch_buffer: could not find nn.Module inside %s", type(model).__name__)
        return 0

    max_needed = 0
    for m in root.modules():
        if isinstance(m, TPGGMLLinear) and m.shard_shape is not None:
            max_needed = max(max_needed, m.shard_shape[0] * m.shard_shape[1])
    if max_needed > 0:
        if device is None:
            device = torch.device("cuda")
        _shared_dequant_scratch = torch.empty(max_needed, dtype=dtype, device=device)
        _scratch_pinned_dtype = dtype
        logger.debug(
            "Scratch buffer warmed up: %d elements, %.1f MiB (%s on %s)",
            max_needed,
            max_needed * dtype.itemsize / (1024 * 1024),
            dtype,
            device,
        )
    return max_needed


from raylight.distributed_modules.tensor_parallel import (
    TPLinear,
    TensorParallelState,
    all_reduce_tensor,
    gather_tensor_along_dim,
    get_tp_group,
    get_tp_size,
)

# Lazy-cached import for LoRA extraction (avoids import lock on every forward)
_extract_lora_ab = None

def _get_extract_lora_ab():
    global _extract_lora_ab
    if _extract_lora_ab is None:
        try:
            from raylight.comfy_dist.model_patcher import _extract_lora_ab as fn
            _extract_lora_ab = fn
        except ImportError:
            _extract_lora_ab = False  # sentinel: not available
    return _extract_lora_ab if _extract_lora_ab is not False else None

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
        self._tensor_type_int: Optional[int] = None  # cached for hot-path

        # User-selectable dequant precision (mirrors GGMLOps dequant_dtype).
        # None = native quant precision, "target" = match input dtype.
        self.dequant_dtype = None
        self.fused_cuda_kernel = True  # can be disabled from node UI

        # Pre-allocated scratch buffer for CUDA dequant (avoids malloc per forward).
        # Uses a module-level shared buffer — see _shared_dequant_scratch.

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
        # Cache for hot-path dispatch (avoids enum .value on every forward)
        self._tensor_type_int = tensor_type.value if hasattr(tensor_type, "value") else int(tensor_type)

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

        # 2. Input / bias selection
        if self.parallelism == "row":
            if not self.input_is_parallel:
                x = x.chunk(self._tp_size_runtime, dim=-1)[self._tp_rank].contiguous()
            bias = None if self._tp_rank > 0 else self.bias
        else:
            bias = self.bias

        if bias is not None:
            bias = bias.to(x.dtype)

        # ── Fast path: fused CUDA kernel (no dequant materialisation) ──
        #
        # For standard LoRA (no DoRA / LoHa / LoKr), we can keep the fused
        # path and compute the LoRA contribution as a cheap residual:
        #   out = fused_gemm(W_q, x) + B @ (A @ x)
        # This avoids materialising the full FP16 weight.
        # If the user requested fused kernels but the extension isn't
        # available, log once and fall back to the dequant path.
        global _fallback_logged
        if self.fused_cuda_kernel and not HAS_FUSED_GGUF_CUDA:
            if not _fallback_logged:
                logger.info(
                    "Fused CUDA kernels requested but not available — falling back to dequant path."
                )
                _fallback_logged = True

        if HAS_FUSED_GGUF_CUDA and self.fused_cuda_kernel:
            lora_residual_ok, lora_ab = self._try_extract_lora_ab()

            if lora_residual_ok:
                # Resolve dequant_dtype for the fused path.
                _dq = self.dequant_dtype
                _fused_dq = x.dtype if _dq == "target" else _dq

                # Shared scratch buffer for CUDA dequant (large-batch path).
                # A single buffer is reused across ALL TPGGMLLinear modules
                # since only one layer executes at a time.  If warmup_scratch_buffer()
                # was called at model load, the buffer is already correctly sized
                # and typed.  The fallback allocation here handles the no-warmup
                # case and genuinely undersized buffers.
                global _shared_dequant_scratch
                rows, cols = self.shard_shape
                needed = rows * cols
                # The C++ kernel requires scratch.dtype == X.dtype.
                # After the dequant_dtype cast in fused_ggml_gemm, X will be
                # in _fused_dq (or x.dtype if _fused_dq is None).  Match that.
                _kernel_dtype = _fused_dq if _fused_dq is not None else x.dtype
                s = _shared_dequant_scratch
                if s is None or s.numel() < needed or s.device != x.device:
                    # Cold start or genuinely undersized — allocate.
                    if _scratch_pinned_dtype is not None:
                        logger.warning(
                            "Scratch buffer realloc despite warmup "
                            "(had %s, need %d on %s) — check warmup_scratch_buffer() call.",
                            "None" if s is None else f"{s.numel()} elems on {s.device}",
                            needed,
                            x.device,
                        )
                    _shared_dequant_scratch = torch.empty(
                        needed, dtype=_kernel_dtype, device=x.device,
                    )
                    s = _shared_dequant_scratch
                else:
                    s = _shared_dequant_scratch[:needed]

                # The kernel checks scratch.dtype == X.dtype.  If the warmup
                # dtype doesn't match the runtime dtype, view-cast the buffer
                # (zero-cost: same memory, different interpretation) so the
                # kernel sees the correct dtype.  Both fp16 and bf16 are 2
                # bytes, so numel() is preserved.
                if s.dtype != _kernel_dtype:
                    s = s.view(_kernel_dtype)

                fused_out = fused_ggml_gemm(
                    x, self.quantized_weight, self.tensor_type,
                    self.shard_shape, bias,
                    type_int=self._tensor_type_int,
                    scratch=s,
                    dequant_dtype=_fused_dq,
                )
                if fused_out is not None:
                    # Add LoRA residual: sum of B_i @ (A_i @ x) for each adapter
                    if lora_ab is not None:
                        fused_out = fused_out + self._compute_lora_residual(
                            x, lora_ab,
                        )
                    # TP collective
                    fused_out = self._apply_tp_collective(fused_out)
                    return fused_out
                else:
                    # fused_ggml_gemm returned None — log details for debugging
                    logger.debug(
                        "fused_ggml_gemm fall-through: type=%r rows=%r batch=%d requested_fused=%s",
                        getattr(self.tensor_type, 'name', self.tensor_type),
                        None if self.shard_shape is None else self.shard_shape[0],
                        x.shape[0],
                        self.fused_cuda_kernel,
                    )

        # ── Fallback: dequant → F.linear ──
        # 1. Dequantise shard on-the-fly.
        #    Honour user-selected dequant_dtype (from GGUF loader node).
        #    "target" → use input dtype for internal scale arithmetic.
        #    None     → native quant precision (float16 for K-quants).
        #    explicit → that dtype (e.g. torch.float32).
        _dq = self.dequant_dtype
        _internal_dtype = x.dtype if _dq == "target" else _dq
        weight = dequantize(
            self.quantized_weight,
            self.tensor_type,
            self.shard_shape,
            dtype=_internal_dtype,
        )
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)

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
        out = self._apply_tp_collective(out)
        return out

    # ── LoRA residual helpers ────────────────────────────────

    def _try_extract_lora_ab(self):
        """Check whether active patches can use the residual LoRA path.

        Returns ``(can_use_fused, lora_ab_or_none)`` where:
        - ``(True, None)``       — no patches at all, fused path is fine
        - ``(True, (As, Bs))``   — standard LoRA, residual approach works
        - ``(False, None)``      — DoRA / LoHa / unsupported, need dequant fallback
        """
        wf = getattr(self, "weight_function", None)
        if not wf:
            return True, None  # no LoRA — fused path is fine

        extract_fn = _get_extract_lora_ab()
        if extract_fn is None:
            return False, None

        # Get the raw patch list from LowVramPatch
        for fn in wf:
            if not hasattr(fn, "patches") or not hasattr(fn, "key"):
                return False, None  # unknown callback type
            if fn.key not in fn.patches:
                continue  # no active patches for this key
            patch_list = fn.patches[fn.key]
            try:
                all_A, all_B, dora_scale = extract_fn(patch_list)
            except Exception:
                return False, None

            if all_A is None:
                return False, None  # unsupported patch type (LoHa/LoKr/mid/offset)
            if dora_scale is not None:
                return False, None  # DoRA needs full weight for norm

            return True, (all_A, all_B)

        return True, None  # all patches were inactive

    @torch.no_grad()
    def _compute_lora_residual(self, x, lora_ab):
        """Compute sum_i B_i @ (A_i @ x^T) with TP-aware sharding.

        A_i has shape [rank, full_in], B_i has shape [full_out, rank].
        For column-parallel: shard B on dim 0 (output).
        For row-parallel: shard A on dim 1 (input).
        """
        all_A, all_B = lora_ab
        tp_rank = self._tp_rank
        residual = None

        for A, B in zip(all_A, all_B):
            A = A.to(device=x.device, dtype=x.dtype)
            B = B.to(device=x.device, dtype=x.dtype)

            # TP shard the LoRA matrices (stored at full size)
            if self._tp_size_runtime > 1:
                if self.parallelism == "column":
                    # Output dim is sharded — narrow B on dim 0
                    local_out = self.local_out_features
                    if B.shape[0] != local_out:
                        B = B.narrow(0, tp_rank * local_out, local_out)
                else:
                    # Input dim is sharded — narrow A on dim 1
                    local_in = self.local_in_features
                    if A.shape[1] != local_in:
                        A = A.narrow(1, tp_rank * local_in, local_in)

            # x: (*, in) → x_2d: (batch, in)
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])

            # LoRA residual: B @ (A @ x^T) → (out, batch) → transpose → (batch, out)
            Ax = torch.mm(A, x_2d.t())   # [rank, batch]
            BAx = torch.mm(B, Ax)          # [local_out, batch]
            delta = BAx.t()                # [batch, local_out]

            # Restore leading dims
            delta = delta.reshape(*orig_shape[:-1], delta.shape[-1])

            if residual is None:
                residual = delta
            else:
                residual = residual + delta

        return residual

    def _apply_tp_collective(self, out):
        """Apply the TP collective (gather or all-reduce) to the output.

        When :func:`defer_tp_collectives` context is active, returns *out*
        unchanged — the caller handles the collective manually.
        """
        from raylight.distributed_modules.tensor_parallel import _is_collective_deferred
        if _is_collective_deferred():
            return out
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

        # Carry over settings from the source GGMLOps.Linear
        tp_ggml.dequant_dtype = getattr(linear, "dequant_dtype", None)
        tp_ggml.fused_cuda_kernel = getattr(linear, "fused_cuda_kernel", True)

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
