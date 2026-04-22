"""TPFp8AmpereLinear — TP-parallel linear layer backed by FP8 Ampere weights.

Wraps ``Fp8AmpereLinear`` to add tensor-parallel sharding and collectives.
The architecture mirrors ``TPGGMLLinear`` in ``tp_linear_factory.py``:

  Column-parallel  (parallelism="column")
    Shards output dimension N across TP ranks.
    Each rank holds weight_u8[local_N, K] and weight_scale[local_N].
    After the local IMMA GEMM an all-gather across dim=-1 reconstructs the
    full [*, N] output (when gather_output=True).

  Row-parallel  (parallelism="row")
    Shards input dimension K across TP ranks.
    Each rank holds weight_u8[N, local_K] and weight_scale[N].
    After the local IMMA GEMM an all-reduce sums partial [*, N] results.
    The per-row weight_scale must reflect the scale of the local K-shard
    (see ``narrow_fp8_weight_for_tp``).

LoRA
    comfy_cast_weights = True so RaylightModelPatcher attaches weight_function /
    bias_function for LoRA patching.  The forward uses the residual approach:
      out = imma_gemm(W_fp8, x) + Σ B_i @ (A_i @ x)
    This avoids materialising the full BF16 weight for standard rank-decomposition
    LoRAs.  DoRA / LoHa / LoKr fall back to the dequant path.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Literal, Optional

from raylight.distributed_modules.tensor_parallel import (
    all_reduce_tensor,
    gather_tensor_along_dim,
    get_tp_group,
    get_tp_size,
)
from .fp8_ampere_linear import Fp8AmpereLinear, Fp8KernelMixin, _try_load_extension, _is_ampere_only
from .packing import compute_fp8_scales, narrow_fp8_weight_for_tp, quantize_to_fp8_e4m3, repack_fp8_for_marlin

__all__ = ["TPFp8AmpereLinear"]

logger = logging.getLogger(__name__)

# Lazy-cached import for LoRA A/B extraction (same pattern as TPGGMLLinear).
_extract_lora_ab = None


def _get_extract_lora_ab():
    global _extract_lora_ab
    if _extract_lora_ab is None:
        try:
            from raylight.comfy_dist.model_patcher import _extract_lora_ab as fn
            _extract_lora_ab = fn
        except ImportError:
            _extract_lora_ab = False
    return _extract_lora_ab if _extract_lora_ab is not False else None


# ---------------------------------------------------------------------------
# TPFp8AmpereLinear
# ---------------------------------------------------------------------------

class TPFp8AmpereLinear(Fp8KernelMixin, nn.Module):
    """TP-parallel FP8 Ampere linear layer.

    See module docstring for description.  The layer is intentionally
    structured in the same way as ``TPGGMLLinear`` so that existing TP
    infrastructure (warmup, LoRA, compressor, collective deferral) works
    without modification.
    """

    # Enables ComfyUI weight_function hook for LoRA patching.
    comfy_cast_weights: bool = True

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
    ) -> None:
        super().__init__()
        self.in_features     = in_features
        self.out_features    = out_features
        self.parallelism     = parallelism
        self.gather_output   = gather_output
        self.input_is_parallel = input_is_parallel
        self.reduce_results  = reduce_results
        self._tp_group       = tp_group
        self.compressor      = compressor

        _tp_sz = (
            dist.get_world_size(tp_group) if tp_group is not None
            else get_tp_size()
        )
        if parallelism == "column":
            self.local_out_features = out_features // _tp_sz
            self.local_in_features  = in_features
        else:
            self.local_out_features = out_features
            self.local_in_features  = in_features // _tp_sz

        # FP8 weight shard [local_N, K] or [N, local_K] as uint8.
        self.register_buffer("weight_u8",    None)
        # Per-channel scale [local_N] or [N] as bfloat16.
        self.register_buffer("weight_scale", None)

        # Zero-element weight stub so state_dict() exposes "*weight" keys
        # required by ComfyUI LoRA key construction.
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)

        # Bias
        if bias:
            _bias_sz = (
                self.local_out_features if parallelism == "column"
                else out_features
            )
            self.bias = nn.Parameter(
                torch.zeros(_bias_sz, dtype=torch.bfloat16), requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self._ext = None  # lazily loaded CUDA extension
        # Marlin-packed weight + padded scale, registered as buffers so the
        # pinned cache can offload/reload them with the rest of the model.
        # Populated lazily by _prepare_packed(); _packed tracks whether packing
        # has happened (survives storage resize-to-0 during offload).
        self.register_buffer("weight_packed", None)
        self.register_buffer("scale_padded",  None)
        self._packed: bool = False
        self._local_n: int | None = None  # cached before weight_u8 is freed

    # ── TP group helpers ──────────────────────────────────────────────

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

    # ── Extension loading ─────────────────────────────────────────────

    def _load_ext(self):
        if self._ext is None:
            self._ext = _try_load_extension()
        return self._ext

    # ── Forward ───────────────────────────────────────────────────────

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Empty-input fast path (shape inference, empty microbatch)
        if x.numel() == 0:
            out_dim = (
                self.local_out_features
                if self.parallelism == "column"
                else self.out_features
            )
            return torch.empty(*x.shape[:-1], out_dim, dtype=x.dtype, device=x.device)

        # Row-parallel: optionally select this rank's input shard.
        if self.parallelism == "row" and not self.input_is_parallel:
            x = x.chunk(self._tp_size_runtime, dim=-1)[self._tp_rank].contiguous()

        # Bias: only rank 0 adds bias in row-parallel to avoid double-counting.
        if self.parallelism == "row" and self._tp_rank > 0:
            bias = None
        else:
            bias = self.bias

        # ── IMMA path (with optional LoRA residual) ─────────────────
        ext = self._load_ext()
        device = x.device
        if ext is not None and _is_ampere_only(device):
            can_residual, lora_ab = self._try_extract_lora_ab()
            if can_residual:
                out = self._forward_imma_2d(x, ext)
                if lora_ab is not None:
                    out = out + self._compute_lora_residual(x, lora_ab)
                if bias is not None:
                    out = out + bias.to(device=out.device, dtype=out.dtype)
                return self._apply_tp_collective(out)

        # ── Fallback: dequant → BF16 F.linear ───────────────────────
        if self.weight_u8.numel() == 0:
            raise RuntimeError(
                "TPFp8AmpereLinear: raw FP8 weight has been freed after Marlin "
                "packing and the CUDA extension is no longer available."
            )
        w_bf16 = self.weight_u8.view(torch.float8_e4m3fn).to(dtype=torch.bfloat16)

        # Apply LoRA patches via weight_function (ComfyUI LowVramPatch).
        wf = getattr(self, "weight_function", None)
        if wf:
            for fn in wf:
                w_bf16 = fn(w_bf16)

        out = F.linear(x.to(torch.bfloat16), w_bf16, bias.to(device=x.device, dtype=torch.bfloat16) if bias is not None else None)
        return self._apply_tp_collective(out)

    def _prepare_packed(self, device: torch.device | None = None) -> None:
        """Repack weight_u8 to Marlin format and free the raw FP8 buffer.

        After packing succeeds ``weight_u8`` is replaced with a 0-element
        placeholder so only the Marlin-layout copy lives in VRAM (same
        pattern as ``Fp8AmpereLinear``).

        ``weight_packed`` and ``scale_padded`` are registered buffers so the
        pinned cache includes them in ``offload_to_cpu`` / ``reload_to_cuda``
        cycles.  ``_packed`` is a plain Python bool that survives storage
        resize-to-0 and tells us not to re-pack after a reload.

        Args:
            device: target device for the packed tensor.  If *None* the
                    device of ``weight_u8`` is used (lazy-forward path).
        """
        if self._packed:
            # Device migration is handled by on_device_activated (called by
            # ModelContext._activate_fp8_buffers after every reload_to_cuda).
            return

        target = device or self.weight_u8.device

        packed, scale_padded, _, _ = repack_fp8_for_marlin(
            self.weight_u8, self.weight_scale,
        )
        self.weight_packed = packed.to(target)
        self.scale_padded  = scale_padded.to(target) if scale_padded is not None else None
        # Cache local_N before freeing (needed by _forward_imma_2d).
        self._local_n = self.weight_u8.shape[0]
        self._packed = True

        # Free raw FP8 storage — keeps only the Marlin-layout copy in VRAM.
        self.weight_u8.data = self.weight_u8.new_empty(0)

    def _forward_imma_2d(self, x: torch.Tensor, ext) -> torch.Tensor:
        """Run the BF16 tensor core kernel, handling arbitrary leading dims."""
        if x.dtype != torch.bfloat16:
            x = x.to(dtype=torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        orig_shape = x.shape
        if x.dim() != 2:
            x = x.view(-1, orig_shape[-1])

        self._prepare_packed(x.device)

        out = ext.fp8_ampere_mm(x, self.weight_packed, self.scale_padded)

        local_n = self._local_n
        # Slice off padding columns if N was rounded up to BLOCK_N
        if out.shape[-1] != local_n:
            out = out[:, :local_n]
        if len(orig_shape) != 2:
            out = out.view(*orig_shape[:-1], local_n)
        return out

    # ── LoRA residual helpers (mirrored from TPGGMLLinear) ────────────

    def _try_extract_lora_ab(self):
        """Return (can_use_imma, lora_ab_or_None).

        (True, None)     — no patches; IMMA is fine.
        (True, (As, Bs)) — standard rank-decomposition LoRA; IMMA + residual.
        (False, None)    — DoRA / LoHa / unsupported; fall back to dequant.
        """
        wf = getattr(self, "weight_function", None)
        if not wf:
            return True, None

        extract_fn = _get_extract_lora_ab()
        if extract_fn is None:
            return False, None

        for fn in wf:
            if not (hasattr(fn, "patches") and hasattr(fn, "key")):
                return False, None
            if fn.key not in fn.patches:
                continue
            try:
                all_A, all_B, dora_scale = extract_fn(fn.patches[fn.key])
            except Exception:
                return False, None
            if all_A is None or dora_scale is not None:
                return False, None
            return True, (all_A, all_B)

        return True, None

    @torch.no_grad()
    def _compute_lora_residual(self, x: torch.Tensor, lora_ab) -> torch.Tensor:
        """Compute Σ_i B_i @ (A_i @ x) with TP-aware shard narrowing."""
        all_A, all_B = lora_ab
        tp_rank = self._tp_rank
        residual = None

        for A, B in zip(all_A, all_B):
            A = A.to(device=x.device, dtype=x.dtype)
            B = B.to(device=x.device, dtype=x.dtype)

            if self._tp_size_runtime > 1:
                if self.parallelism == "column":
                    local_out = self.local_out_features
                    if B.shape[0] != local_out:
                        B = B.narrow(0, tp_rank * local_out, local_out)
                else:
                    local_in = self.local_in_features
                    if A.shape[1] != local_in:
                        A = A.narrow(1, tp_rank * local_in, local_in)

            orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            Ax = torch.mm(A, x_2d.t())   # [rank, batch]
            BAx = torch.mm(B, Ax)         # [local_out, batch]
            delta = BAx.t().reshape(*orig_shape[:-1], BAx.shape[0])

            residual = delta if residual is None else residual + delta

        return residual

    def _apply_tp_collective(self, out: torch.Tensor) -> torch.Tensor:
        """All-gather (column) or all-reduce (row) with optional deferral."""
        from raylight.distributed_modules.tensor_parallel import _is_collective_deferred
        if _is_collective_deferred():
            return out
        if self.parallelism == "column":
            if self.gather_output and self._tp_size_runtime > 1:
                out = gather_tensor_along_dim(out, dim=-1, full_size=self.out_features)
        else:
            if self.reduce_results and self._tp_size_runtime > 1:
                if self.compressor is not None:
                    out = self.compressor.compressed_all_reduce(
                        out, group=self._resolved_group,
                    )
                else:
                    out = all_reduce_tensor(out)
        return out

    # ── Factory ───────────────────────────────────────────────────────

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
    ) -> "TPFp8AmpereLinear":
        """Create a ``TPFp8AmpereLinear`` from any linear with FP8 or float weights.

        Accepts:
          - ``nn.Linear`` (BF16/FP32 weights) → quantises to FP8 first.
          - ``Fp8AmpereLinear`` → reuses existing weight_u8 / weight_scale.
          - Any module whose ``.weight`` is ``float8_e4m3fn`` or ``uint8``.
        """
        w = getattr(linear, "weight", None)
        if w is None:
            raise TypeError(f"from_linear: module {type(linear)!r} has no 'weight'")

        in_features  = linear.in_features
        out_features = linear.out_features
        has_bias     = getattr(linear, "bias", None) is not None

        # --- Obtain full (unsharded) weight_u8 + weight_scale ---

        if isinstance(linear, Fp8AmpereLinear):
            # Already packed — reuse buffers directly.
            full_u8    = linear.weight.data         # [N, K] uint8
            full_scale = linear.weight_scale.data   # [N] bfloat16
        elif w.dtype in (torch.float8_e4m3fn, torch.uint8):
            # FP8 checkpoint weight.
            full_u8 = w.view(torch.uint8) if w.dtype == torch.float8_e4m3fn else w
            full_scale = compute_fp8_scales(full_u8)
        else:
            # Dense float weight — quantise to FP8.
            full_u8, full_scale = quantize_to_fp8_e4m3(
                w.to(dtype=torch.bfloat16) if w.dtype != torch.bfloat16 else w
            )

        # --- Determine tp_rank for sharding ---
        _tmp_tp_sz = (
            dist.get_world_size(tp_group) if tp_group is not None
            else get_tp_size()
        )
        _tmp_rank = (
            dist.get_rank(tp_group) if tp_group is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )

        # --- Shard the FP8 weight ---
        _shard_dim = 0 if parallelism == "column" else 1
        shard_u8, shard_scale = narrow_fp8_weight_for_tp(
            full_u8.contiguous(),
            full_scale.contiguous(),
            dim=_shard_dim,
            rank=_tmp_rank,
            tp_size=_tmp_tp_sz,
        )

        obj = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            parallelism=parallelism,
            gather_output=gather_output,
            input_is_parallel=input_is_parallel,
            reduce_results=reduce_results,
            tp_group=tp_group,
            compressor=compressor,
        )
        obj.register_buffer("weight_u8",    shard_u8.contiguous())
        obj.register_buffer("weight_scale", shard_scale.contiguous())

        # Bias — column-parallel shards bias; row-parallel keeps full bias.
        if has_bias:
            bias_src = (
                linear.bias.data
                if hasattr(linear.bias, "data")
                else linear.bias
            )
            if parallelism == "column" and bias_src.shape[0] != obj.local_out_features:
                bias_src = bias_src.narrow(
                    0, _tmp_rank * obj.local_out_features, obj.local_out_features,
                )
            obj.bias.data.copy_(bias_src.to(dtype=torch.bfloat16))

        return obj

    @classmethod
    def from_tp_linear(
        cls,
        tp_linear: "TPLinear",
        tp_group: Optional[dist.ProcessGroup] = None,
        compressor=None,
    ) -> "TPFp8AmpereLinear":
        """Create a ``TPFp8AmpereLinear`` from a ``TPLinear`` whose weight is
        already a sharded FP8 tensor (``float8_e4m3fn`` or ``uint8``).

        This is the factory used by ``_apply_tp_fp8_ampere_swap`` in
        ``tp.py``.  The weight is already the per-rank shard — no further
        sharding is performed.  Scales are computed from the shard directly.

        Args:
            tp_linear:  A ``TPLinear`` with ``weight.dtype == float8_e4m3fn``
                        (as loaded by the TP streaming loader).
            tp_group:   Optional explicit process group.  Defaults to the
                        same group as *tp_linear*.
            compressor: Optional gradient compressor (passed through).
        """
        from raylight.distributed_modules.tensor_parallel import TPLinear  # local to avoid circular

        if not isinstance(tp_linear, TPLinear):
            raise TypeError(
                f"from_tp_linear expects a TPLinear, got {type(tp_linear)!r}"
            )

        w = tp_linear.weight
        if w.dtype not in (torch.float8_e4m3fn, torch.uint8):
            raise TypeError(
                f"from_tp_linear: TPLinear.weight must be float8_e4m3fn or uint8, "
                f"got {w.dtype}"
            )

        shard_u8 = w.view(torch.uint8).contiguous()

        # Prefer the checkpoint scale already loaded by the streaming loader
        # (tp_linear.weight_scale is set from the safetensors file during Phase 5).
        # Fall back to all-ones only when no checkpoint scale is present.
        existing_scale = getattr(tp_linear, "weight_scale", None)
        if (
            isinstance(existing_scale, torch.Tensor)
            and existing_scale.shape == (shard_u8.shape[0],)
        ):
            shard_scale = existing_scale.to(dtype=torch.bfloat16)
        else:
            shard_scale = compute_fp8_scales(shard_u8)

        resolved_group = tp_group if tp_group is not None else tp_linear._tp_group

        obj = cls(
            in_features=tp_linear.in_features,
            out_features=tp_linear.out_features,
            bias=tp_linear.bias is not None,
            parallelism=tp_linear.parallelism,
            gather_output=tp_linear.gather_output,
            input_is_parallel=tp_linear.input_is_parallel,
            reduce_results=tp_linear.reduce_results,
            tp_group=resolved_group,
            compressor=compressor,
        )
        obj.register_buffer("weight_u8",    shard_u8)
        obj.register_buffer("weight_scale", shard_scale)

        if tp_linear.bias is not None:
            obj.bias.data.copy_(tp_linear.bias.data.to(dtype=torch.bfloat16))

        return obj

    # ── String representation ─────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"parallelism={self.parallelism!r}, "
            f"local_out={self.local_out_features}, "
            f"bias={self.bias is not None}"
        )
