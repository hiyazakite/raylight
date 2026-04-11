"""Tensor Parallel for Lumina models (true head-sharded TP).

True head-sharded TP: each rank holds a shard of the attention heads and
MLP intermediate dimension.  This gives **compute, activation-memory, and
weight-memory savings**.

Lumina's ``JointAttention`` uses a fused QKV projection with optional GQA
and per-head-dim QK norms (``RMSNorm(head_dim)``).  Since the norm is
per-head, each rank normalises its local shard independently — no
cross-rank communication is needed (unlike LTXAV's full-dim norm).

RoPE positional embeddings broadcast across heads (the head dimension is
size 1), so no per-rank RoPE slicing is required.

Attribute map (ComfyUI ``comfy.ldm.lumina.model``):

  JointAttention
  ──────────────
    qkv           Linear(D, (H+2*KVH)*d)  → TPLinear column, gather=False
    out           Linear(D, D)             → TPLinear row, all-reduce
    q_norm/k_norm RMSNorm(head_dim)        → kept as-is (per-head, no sharding)

  FeedForward (SwiGLU)
  ────────────────────
    w1            Linear(D, FFN)           → TPLinear column, gather=False
    w3            Linear(D, FFN)           → TPLinear column, gather=False
    w2            Linear(FFN, D)           → TPLinear row, all-reduce

  JointTransformerBlock
  ─────────────────────
    adaLN_modulation[-1]  Linear(…, 4D)   → TPLinear column, gather=True (replicated output)

  Boundary projections                     → TPLinear column, gather=True (replicated output)
"""

import types
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from raylight.distributed_modules.tensor_parallel import (
    TPLinear,
    TensorParallelState,
    get_tp_group,
)
from raylight.distributed_modules.tp_linear_factory import make_tp_linear
from raylight.distributed_modules.attention.dispatcher import get_local_attn_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_linear(
    parent: nn.Module,
    attr_name: str,
    parallelism: Literal["column", "row"] = "column",
    gather_output: bool = True,
    input_is_parallel: bool = False,
    reduce_results: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """Replace ``parent.<attr_name>`` (an ``nn.Linear``) with a TP variant.

    Supports sequential-style indexed access (``parent[idx]``) when
    *attr_name* is a digit string and *parent* is an ``nn.Sequential``.
    """
    if attr_name.isdigit():
        linear = parent[int(attr_name)]
    else:
        linear = getattr(parent, attr_name, None)

    if linear is None or not isinstance(linear, nn.Linear):
        return

    tp_lin = make_tp_linear(
        linear,
        parallelism=parallelism,
        gather_output=gather_output,
        input_is_parallel=input_is_parallel,
        reduce_results=reduce_results,
        tp_group=tp_group,
        structure_only=structure_only,
    )

    if attr_name.isdigit():
        parent[int(attr_name)] = tp_lin
    else:
        setattr(parent, attr_name, tp_lin)


# ---------------------------------------------------------------------------
# Fused-QKV sharding helper
# ---------------------------------------------------------------------------

def _replace_qkv_linear(
    attn: nn.Module,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """Replace ``attn.qkv`` with a column-parallel TPLinear whose weight is
    sharded per-head across Q, K, V sections independently.

    The original fused weight layout is ``[Q_all | K_all | V_all]`` along
    dim 0.  Standard column-parallel ``narrow`` would cut across the Q/K/V
    boundary.  This helper installs a custom ``weight_loader`` that extracts
    each rank's Q, K, V head-slices and concatenates them as
    ``[Q_local | K_local | V_local]``.
    """
    old_qkv = attn.qkv
    if old_qkv is None or not isinstance(old_qkv, nn.Linear):
        return

    tp_size = TensorParallelState.get_size()
    tp_rank = TensorParallelState.get_rank()

    q_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim
    full_out = q_size + 2 * kv_size

    local_q_size = (n_heads // tp_size) * head_dim
    local_kv_size = (n_kv_heads // tp_size) * head_dim

    in_f = old_qkv.in_features
    has_bias = old_qkv.bias is not None

    # Dequantise GGML weights when needed
    try:
        from raylight.expansion.comfyui_gguf.dequant import is_quantized, dequantize_tensor
    except ImportError:
        is_quantized = lambda w: False
        dequantize_tensor = None

    weight = old_qkv.weight
    if is_quantized(weight):
        weight = dequantize_tensor(weight, dtype=torch.float16)

    tp_qkv = TPLinear(
        in_features=in_f,
        out_features=full_out,
        bias=has_bias,
        parallelism="column",
        gather_output=False,
        tp_group=tp_group,
        dtype=weight.dtype,
        device="meta" if structure_only else weight.device,
    )

    # --- QKV-aware weight_loader ---
    _q_size = q_size
    _kv_size = kv_size
    _local_q = local_q_size
    _local_kv = local_kv_size

    def _qkv_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Shard Q, K, V sections independently along output dim."""
        _rank = (
            dist.get_rank(tp_group) if tp_group is not None else TensorParallelState.get_rank()
        )
        if loaded_weight.shape[0] == param.data.shape[0]:
            # Already sharded — direct copy.
            param.data.copy_(loaded_weight)
            return
        # Full weight: [Q_all | K_all | V_all] along dim 0
        q_full = loaded_weight.narrow(0, 0, _q_size)
        k_full = loaded_weight.narrow(0, _q_size, _kv_size)
        v_full = loaded_weight.narrow(0, _q_size + _kv_size, _kv_size)
        q_local = q_full.narrow(0, _rank * _local_q, _local_q)
        k_local = k_full.narrow(0, _rank * _local_kv, _local_kv)
        v_local = v_full.narrow(0, _rank * _local_kv, _local_kv)
        param.data.copy_(torch.cat([q_local, k_local, v_local], dim=0))

    def _qkv_scale_loader(scale: torch.Tensor) -> torch.Tensor:
        """Shard per-row weight_scale with QKV-aware slicing.

        Per-row scales have the same fused layout as the weight's dim 0:
        ``[Q_scales | K_scales | V_scales]``.  Must be sliced per-head
        the same way as the weight itself.

        Scalar scales (Python float or 0-dim tensor) are returned as-is.
        """
        if not isinstance(scale, torch.Tensor):
            return scale  # Python float — no sharding needed
        _rank = (
            dist.get_rank(tp_group) if tp_group is not None else TensorParallelState.get_rank()
        )
        if scale.numel() <= 1:
            return scale  # scalar per-tensor scale — no sharding
        expected_full = _q_size + 2 * _kv_size
        if scale.shape[0] != expected_full:
            return scale  # unexpected shape — don't shard
        q_s = scale.narrow(0, 0, _q_size)
        k_s = scale.narrow(0, _q_size, _kv_size)
        v_s = scale.narrow(0, _q_size + _kv_size, _kv_size)
        return torch.cat([
            q_s.narrow(0, _rank * _local_q, _local_q),
            k_s.narrow(0, _rank * _local_kv, _local_kv),
            v_s.narrow(0, _rank * _local_kv, _local_kv),
        ], dim=0).contiguous()

    tp_qkv.weight_loader = _qkv_weight_loader  # type: ignore[assignment]
    tp_qkv._qkv_scale_loader = _qkv_scale_loader  # type: ignore[assignment]

    # Load weight immediately if not structure_only
    if not structure_only:
        _qkv_weight_loader(tp_qkv.weight, weight.data)
        if has_bias and tp_qkv.bias is not None:
            bias_data = old_qkv.bias.data
            if is_quantized(bias_data):
                bias_data = dequantize_tensor(bias_data, dtype=torch.float32)
            # Bias uses same QKV-aware sharding (dim 0)
            q_b = bias_data[:_q_size]
            k_b = bias_data[_q_size:_q_size + _kv_size]
            v_b = bias_data[_q_size + _kv_size:]
            q_b_local = q_b[tp_rank * local_q_size:(tp_rank + 1) * local_q_size]
            k_b_local = k_b[tp_rank * local_kv_size:(tp_rank + 1) * local_kv_size]
            v_b_local = v_b[tp_rank * local_kv_size:(tp_rank + 1) * local_kv_size]
            tp_qkv.bias.data.copy_(torch.cat([q_b_local, k_b_local, v_b_local], dim=0))

        # INT8 per-row weight_scale: shard with QKV-aware slicing
        if hasattr(old_qkv, "weight_scale"):
            setattr(tp_qkv, "weight_scale", _qkv_scale_loader(old_qkv.weight_scale))

    attn.qkv = tp_qkv


# ---------------------------------------------------------------------------
# JointAttention — true head-sharded TP
# ---------------------------------------------------------------------------

def apply_tp_to_lumina_attention(
    attn: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a Lumina ``JointAttention`` module.

    QKV projection is column-parallel (heads sharded across ranks).
    Output projection is row-parallel with all-reduce.
    QK norms are per-head-dim and stay unsharded.
    A custom forward is installed that uses local heads.

    The fused QKV weight is laid out as ``[Q_all | K_all | V_all]`` along
    dim 0.  Plain column-parallel ``narrow`` would shard *across* the Q/K/V
    boundary (giving rank 0 all-Q + some-K, rank 1 some-K + all-V, etc.).
    We therefore install a QKV-aware ``weight_loader`` that extracts this
    rank's Q, K, V head-slices independently and concatenates them as
    ``[Q_local | K_local | V_local]``.
    """
    tp_size = TensorParallelState.get_size()
    tp_rank = TensorParallelState.get_rank()

    # Compute local head counts
    n_heads = attn.n_local_heads
    n_kv_heads = attn.n_local_kv_heads
    head_dim = attn.head_dim
    local_n_heads = n_heads // tp_size
    local_n_kv_heads = n_kv_heads // tp_size

    # 1. QKV: column-parallel with QKV-aware sharding
    #    Standard column-parallel would slice a contiguous block from
    #    [Q|K|V]; instead we shard Q, K, V heads independently.
    _replace_qkv_linear(
        attn, n_heads, n_kv_heads, head_dim,
        tp_group=tp_group, structure_only=structure_only,
    )

    # 2. Output: row-parallel with all-reduce
    _replace_linear(
        attn, "out", parallelism="row",
        input_is_parallel=True, reduce_results=True,
        tp_group=tp_group, structure_only=structure_only,
    )

    # 3. Store local head info for the TP forward
    attn._tp_local_heads = local_n_heads
    attn._tp_local_kv_heads = local_n_kv_heads
    attn._tp_rank = tp_rank
    attn._tp_size = tp_size

    # 4. Install TP-aware forward
    @torch.compiler.disable
    def _tp_forward(self, x, x_mask, freqs_cis, transformer_options={}):
        from comfy.ldm.flux.math import apply_rope
        from raylight.distributed_modules.attention.dispatcher import get_local_attn_fn
        bsz, seqlen, _ = x.shape

        # QKV with local head counts
        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self._tp_local_heads * self.head_dim,
                self._tp_local_kv_heads * self.head_dim,
                self._tp_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self._tp_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self._tp_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self._tp_local_kv_heads, self.head_dim)

        # Per-head-dim QK norms — no cross-rank communication needed
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # RoPE (broadcasts across heads, no slicing needed)
        xq, xk = apply_rope(xq, xk, freqs_cis)

        # GQA expansion for local heads
        n_rep = self._tp_local_heads // self._tp_local_kv_heads
        if n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # Use Raylight's local attention dispatcher
        if not hasattr(self, "_raylight_attn_fn"):
            self._raylight_attn_fn = get_local_attn_fn()

        # Dispatcher expects [B, T, H*D] format
        q_flat = xq.flatten(2)  # [B, S, local_heads * head_dim]
        k_flat = xk.flatten(2)
        v_flat = xv.flatten(2)

        output = self._raylight_attn_fn(
            q_flat, k_flat, v_flat, self._tp_local_heads,
            mask=x_mask,
            skip_reshape=False,
            transformer_options=transformer_options,
        )

        # Row-parallel output projection with all-reduce
        return self.out(output)

    attn.forward = types.MethodType(_tp_forward, attn)


# ---------------------------------------------------------------------------
# FeedForward (SwiGLU) — true head-sharded TP
# ---------------------------------------------------------------------------

def apply_tp_to_lumina_feedforward(
    ff: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a Lumina ``FeedForward`` module.

    w1 (gate) and w3 (up) are column-parallel (intermediate dim sharded).
    w2 (down) is row-parallel with all-reduce.
    """
    # w1, w3: column-parallel, sharded intermediate output
    _replace_linear(
        ff, "w1", parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    _replace_linear(
        ff, "w3", parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    # w2: row-parallel, takes sharded input, all-reduces output
    _replace_linear(
        ff, "w2", parallelism="row",
        input_is_parallel=True, reduce_results=True,
        tp_group=tp_group, structure_only=structure_only,
    )


# ---------------------------------------------------------------------------
# JointTransformerBlock
# ---------------------------------------------------------------------------

def apply_tp_to_lumina_block(
    block: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a Lumina ``JointTransformerBlock``.

    Attention and feedforward use true head-sharded TP.
    AdaLN modulation is kept **replicated** (full weight on every rank,
    matching sglang's ``ReplicatedLinear`` pattern).  Its output must be
    identical on every rank since it feeds into modulate/gate paths that
    operate on replicated activations.
    """
    if hasattr(block, "attention"):
        apply_tp_to_lumina_attention(
            block.attention, tp_group=tp_group, structure_only=structure_only,
        )

    if hasattr(block, "feed_forward"):
        apply_tp_to_lumina_feedforward(
            block.feed_forward, tp_group=tp_group, structure_only=structure_only,
        )

    # AdaLN modulation is intentionally NOT TP-patched.
    # sglang keeps this as ReplicatedLinear — every rank holds the full
    # weight and produces identical output, avoiding any gather.

    block._tp_patched = True


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def apply_tp_to_lumina_model(
    model: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """Full in-place TP patching of a Lumina NextDiT diffusion model.

    Follows sglang's ``ZImageTransformer2DModel`` TP patterns:

    **TP-sharded** (compute + memory savings):
      - Attention QKV: column-parallel (QKV-aware head sharding)
      - Attention output: row-parallel (all-reduce)
      - FeedForward w1/w3: column-parallel
      - FeedForward w2: row-parallel (all-reduce)
      - x_embedder: column-parallel, gather (boundary)
      - final_layer.linear: column-parallel, gather (boundary)

    **Replicated** (full weight on every rank, no communication):
      - adaLN_modulation (block + final_layer)
      - t_embedder, cap_embedder, clip_text_pooled_proj
      - time_text_embed, siglip_embedder
      - QK norms (per-head-dim, no sharding needed)

    Args:
        model: The inner diffusion model (``comfy_model.diffusion_model``).
        tp_group: TP process group (defaults to global TP group).
        structure_only: When True, create TPLinear with correct shapes but
            skip weight copies (for streaming weight population).
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_group = tp_group or get_tp_group()

    # --- Main transformer blocks ---
    for block in getattr(model, "layers", []):
        apply_tp_to_lumina_block(block, tp_group=tp_group, structure_only=structure_only)

    # --- Refiner blocks ---
    for block in getattr(model, "noise_refiner", []):
        apply_tp_to_lumina_block(block, tp_group=tp_group, structure_only=structure_only)

    for block in getattr(model, "context_refiner", []):
        apply_tp_to_lumina_block(block, tp_group=tp_group, structure_only=structure_only)

    for block in getattr(model, "siglip_refiner", []) or []:
        apply_tp_to_lumina_block(block, tp_group=tp_group, structure_only=structure_only)

    # --- Boundary projections ---
    # x_embedder: column-parallel with gather (matches sglang)
    _replace_linear(
        model, "x_embedder", parallelism="column", gather_output=True,
        tp_group=tp_group, structure_only=structure_only,
    )

    # cap_embedder, clip_text_pooled_proj, time_text_embed, siglip_embedder:
    # Kept replicated (sglang uses ReplicatedLinear for all of these).

    # final_layer: only the output linear is TP'd; adaLN is replicated
    if hasattr(model, "final_layer"):
        fl = model.final_layer
        _replace_linear(
            fl, "linear", parallelism="column", gather_output=True,
            tp_group=tp_group, structure_only=structure_only,
        )
