"""Tensor Parallelism for Qwen Image edit models (Phase 2.5 — true head-sharded TP).

Architecture: ``QwenImageTransformer2DModel`` (60 ``QwenImageTransformerBlock``s)
Each block is dual-stream (img + txt) with:

  Attention
  ─────────
    to_q / to_k / to_v            Linear(D, H*d)   img stream
    add_q_proj / add_k_proj / add_v_proj             txt stream
    norm_q / norm_k / norm_added_q / norm_added_k    RMSNorm(head_dim) — per-head, no sharding
    to_out[0]                      Linear(D, D)      img output
    to_add_out                     Linear(D, D)      txt output

  FeedForward (img_mlp, txt_mlp)  GELU 2-layer
  ──────────────────────────────
    net[0].proj                    Linear(D, FFN)    GELU gate
    net[2]                         Linear(FFN, D)    output

  AdaLN modulation
  ────────────────
    img_mod[1] / txt_mod[1]        Linear(D, 6*D)   shift/scale/gate

TP strategy
───────────
  to_q/k/v, add_q/k/v_proj → column, gather_output=False   (head-sharded)
  to_out[0], to_add_out      → row, reduce_results=True     (all-reduce)
  norm_q/k, norm_added_q/k  → kept as-is (per-head-dim, no cross-rank comm needed)
  net[0].proj                → column, gather_output=False   (intermediate sharded)
  net[2]                     → row, reduce_results=True      (all-reduce)
  img_mod[1] / txt_mod[1]   → column, gather_output=True    (replicated full output)

Boundary layers (img_in, txt_in, proj_out, norm_out.linear) are kept replicated.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from raylight.distributed_modules.tensor_parallel import (
    TPLinear,
    TensorParallelState,
    get_tp_group,
)
from raylight.distributed_modules.tp_linear_factory import make_tp_linear


# =============================================================================
# _replace_linear helper (same pattern as lumina/tp.py)
# =============================================================================

def _replace_linear(
    parent: nn.Module,
    attr_name: str,
    parallelism: str = "column",
    gather_output: bool = True,
    input_is_parallel: bool = False,
    reduce_results: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """Replace ``parent.<attr_name>`` (an ``nn.Linear``) with a TP variant.

    Supports ``nn.Sequential`` indexed access when *attr_name* is a digit string.
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


# =============================================================================
# Attention — dual-stream, separate Q/K/V
# =============================================================================

def apply_tp_to_qwen_attention(
    attn: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a Qwen ``Attention`` module.

    Q/K/V projections for both img and txt streams are column-parallel
    (heads sharded across ranks, gather_output=False).
    Output projections are row-parallel with all-reduce.
    QK norms are per-head-dim RMSNorm — kept unsharded on each rank
    (each rank normalises its local head shard independently, which is
    mathematically identical to normalising on the full head set since
    RMSNorm operates per-element along the last dim = head_dim).
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    n_heads = attn.heads
    head_dim = attn.dim_head
    local_heads = n_heads // tp_size

    # 1. Image stream — Q/K/V column-parallel (heads sharded)
    for attr in ("to_q", "to_k", "to_v"):
        _replace_linear(
            attn, attr, parallelism="column", gather_output=False,
            tp_group=tp_group, structure_only=structure_only,
        )

    # 2. Text stream — same
    for attr in ("add_q_proj", "add_k_proj", "add_v_proj"):
        _replace_linear(
            attn, attr, parallelism="column", gather_output=False,
            tp_group=tp_group, structure_only=structure_only,
        )

    # 3. Image output projection — row-parallel, all-reduce
    #    to_out is nn.ModuleList([Linear, Dropout]); replace index 0
    _replace_linear(
        attn.to_out, "0", parallelism="row",
        input_is_parallel=True, reduce_results=True,
        tp_group=tp_group, structure_only=structure_only,
    )

    # 4. Text output projection — row-parallel, all-reduce
    _replace_linear(
        attn, "to_add_out", parallelism="row",
        input_is_parallel=True, reduce_results=True,
        tp_group=tp_group, structure_only=structure_only,
    )

    # 5. Store local head info for the custom forward
    attn._tp_local_heads = local_heads
    attn._tp_head_dim = head_dim
    attn._tp_size = tp_size

    # 6. Install TP-aware forward
    @torch.compiler.disable
    def _tp_attn_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        transformer_options={},
    ):
        from comfy.ldm.flux.math import apply_rope1
        from comfy.ldm.modules.attention import optimized_attention_masked

        batch_size = hidden_states.shape[0]
        seq_img = hidden_states.shape[1]
        seq_txt = encoder_hidden_states.shape[1]
        local_heads = self._tp_local_heads
        head_dim = self._tp_head_dim

        # Project and reshape to [B, local_H, S, D]
        img_query = self.to_q(hidden_states).view(batch_size, seq_img, local_heads, head_dim).transpose(1, 2).contiguous()
        img_key   = self.to_k(hidden_states).view(batch_size, seq_img, local_heads, head_dim).transpose(1, 2).contiguous()
        img_value = self.to_v(hidden_states).view(batch_size, seq_img, local_heads, head_dim).transpose(1, 2)

        txt_query = self.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, local_heads, head_dim).transpose(1, 2).contiguous()
        txt_key   = self.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, local_heads, head_dim).transpose(1, 2).contiguous()
        txt_value = self.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, local_heads, head_dim).transpose(1, 2)

        # Per-head QK norms (each rank normalises its own head shard — correct)
        img_query = self.norm_q(img_query)
        img_key   = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key   = self.norm_added_k(txt_key)

        # Joint [B, local_H, S_txt+S_img, D]
        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key   = torch.cat([txt_key,   img_key],   dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        if encoder_hidden_states_mask is not None:
            attn_mask = torch.zeros(
                (batch_size, 1, seq_txt + seq_img),
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            attn_mask[:, 0, :seq_txt] = encoder_hidden_states_mask
        else:
            attn_mask = None

        # RoPE — image_rotary_emb broadcasts across heads (no per-rank slicing)
        joint_query = apply_rope1(joint_query, image_rotary_emb)
        joint_key   = apply_rope1(joint_key,   image_rotary_emb)

        # Attention with local head count
        joint_hidden_states = optimized_attention_masked(
            joint_query, joint_key, joint_value,
            local_heads,
            attn_mask,
            transformer_options=transformer_options,
            skip_reshape=True,
        )

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Row-parallel output projections → each all-reduces internally
        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)   # Dropout (no-op in eval)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    attn.forward = types.MethodType(_tp_attn_forward, attn)


# =============================================================================
# FeedForward — GELU 2-layer MLP
# =============================================================================

def apply_tp_to_qwen_feedforward(
    ff: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a Qwen ``FeedForward`` module.

    ``net[0]`` is a ``GELU`` wrapper whose ``.proj`` is the gate linear.
    ``net[2]`` is the output ``nn.Linear``.
    GELU and Dropout (net[1]) are element-wise — they work unchanged on
    sharded activations.

    net[0].proj → column, gather_output=False  (intermediate sharded)
    net[2]      → row, reduce_results=True     (all-reduce)
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    # net[0] is a GELU module; its .proj is the Linear
    gelu_module = ff.net[0]
    _replace_linear(
        gelu_module, "proj", parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )

    # net[2] is the output Linear
    _replace_linear(
        ff.net, "2", parallelism="row",
        input_is_parallel=True, reduce_results=True,
        tp_group=tp_group, structure_only=structure_only,
    )


# =============================================================================
# Full block
# =============================================================================

def apply_tp_to_qwen_block(
    block: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a ``QwenImageTransformerBlock``.

    Attention and both feedforwards (img + txt) use true head-sharded TP.
    AdaLN modulation linears (``img_mod[1]``, ``txt_mod[1]``) use
    column-parallel with ``gather_output=True`` so every rank holds the
    full 6×D output needed for shift/scale/gate decomposition.
    """
    if TensorParallelState.get_size() <= 1:
        return

    # Attention
    if hasattr(block, "attn"):
        apply_tp_to_qwen_attention(block.attn, tp_group=tp_group, structure_only=structure_only)

    # Image FeedForward
    if hasattr(block, "img_mlp"):
        apply_tp_to_qwen_feedforward(block.img_mlp, tp_group=tp_group, structure_only=structure_only)

    # Text FeedForward
    if hasattr(block, "txt_mlp"):
        apply_tp_to_qwen_feedforward(block.txt_mlp, tp_group=tp_group, structure_only=structure_only)

    # AdaLN mod projections — replicate output (gather_output=True)
    # img_mod / txt_mod are nn.Sequential([SiLU, Linear(D, 6*D)])
    if hasattr(block, "img_mod"):
        _replace_linear(
            block.img_mod, "1", parallelism="column", gather_output=True,
            tp_group=tp_group, structure_only=structure_only,
        )
    if hasattr(block, "txt_mod"):
        _replace_linear(
            block.txt_mod, "1", parallelism="column", gather_output=True,
            tp_group=tp_group, structure_only=structure_only,
        )

    block._tp_patched = True


# =============================================================================
# Full model
# =============================================================================

def apply_tp_to_qwen_image_model(
    model: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    structure_only: bool = False,
) -> None:
    """Full in-place TP patching of a ``QwenImageTransformer2DModel``.

    TP-sharded (compute + memory savings):
      - Attention Q/K/V per stream: column-parallel
      - Attention output per stream: row-parallel (all-reduce)
      - FeedForward gate (net[0].proj): column-parallel
      - FeedForward output (net[2]): row-parallel (all-reduce)
      - AdaLN mod[1]: column-parallel, gather (replicated output)

    Replicated (full weight on every rank, no communication):
      - img_in, txt_in, txt_norm  (boundary embeddings)
      - time_text_embed, pe_embedder
      - norm_q/k, norm_added_q/k  (per-head-dim RMSNorm)
      - norm_out (LastLayer), proj_out

    Args:
        model: Inner diffusion model (``comfy_model.diffusion_model``).
        tp_group: TP process group (defaults to global TP group).
        structure_only: Create TPLinear with correct shapes but skip weight
            copies — used by streaming weight loading.
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_group = tp_group or get_tp_group()

    for block in getattr(model, "transformer_blocks", []):
        apply_tp_to_qwen_block(block, tp_group=tp_group, structure_only=structure_only)
