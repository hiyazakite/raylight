"""Tensor Parallel for Flux model (weight-only / "fake TP").

Uses ``gather_output=True`` on ALL projections so that every TP rank
materialises full-size activations.  This gives **weight-memory savings
only** — no compute or activation-memory savings — but the forward path
is *identical* to the non-TP path (all shapes are unchanged), which
avoids any need for a custom TP forward or head-slicing.

This follows sglang's approach for Flux (``runtime/models/dits/flux.py``).
Flux uses joint attention (text + image tokens concatenated) which makes
true head-sharded TP complex; weight-only TP is the pragmatic choice.

Attribute map (ComfyUI ``comfy.ldm.flux.layers``):

  DoubleStreamBlock
  ─────────────────
    img_attn.qkv    Linear(D, 3D)        → TPLinear column, gather=True
    img_attn.proj   Linear(D, D)         → TPLinear column, gather=True
    txt_attn.qkv    Linear(D, 3D)        → same
    txt_attn.proj   Linear(D, D)         → same
    img_mlp          Sequential | YakMLP  → replace all linears, gather=True
    txt_mlp          Sequential | YakMLP  → same

  SingleStreamBlock
  ─────────────────
    linear1   Linear(D, 3D+mlp)          → TPLinear column, gather=True
    linear2   Linear(D+mlp, D)           → TPLinear column, gather=True
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

import torch.distributed as dist

from raylight.distributed_modules.tensor_parallel import (
    TPLinear,
    TensorParallelState,
    get_tp_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_linear(
    parent: nn.Module,
    attr_name: str,
    parallelism: Literal["column", "row"] = "column",
    gather_output: bool = True,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Replace ``parent.<attr_name>`` (an ``nn.Linear``) with a ``TPLinear``.

    Supports sequential-style indexed access (``parent[idx]``) when
    *attr_name* is a digit string and *parent* is an ``nn.Sequential``.
    """
    if attr_name.isdigit():
        linear = parent[int(attr_name)]  # type: ignore
    else:
        linear = getattr(parent, attr_name)

    if not isinstance(linear, nn.Linear):
        return  # nothing to replace

    tp_lin = TPLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        parallelism=parallelism,
        gather_output=gather_output,
        tp_group=tp_group,
        dtype=linear.weight.dtype,
        device=linear.weight.device,
    )

    # Copy existing weights into the TP shard.  When the model has already
    # been loaded (pure-TP path) the old nn.Linear holds the full checkpoint
    # tensor; weight_loader() slices this rank's shard via narrow().
    tp_lin.weight_loader(tp_lin.weight, linear.weight.data)
    if linear.bias is not None and tp_lin.bias is not None:
        if parallelism == "column":
            tp_lin.weight_loader(tp_lin.bias, linear.bias.data)
        else:
            tp_lin.bias.data.copy_(linear.bias.data)

    # Preserve INT8 quantization scale (sharded to match weight)
    if hasattr(linear, "weight_scale"):
        scale = linear.weight_scale
        if isinstance(scale, torch.Tensor):
            if parallelism == "column" and scale.numel() > tp_lin.local_out_features:
                tp_rank = tp_lin._tp_rank
                scale = scale.narrow(0, tp_rank * tp_lin.local_out_features, tp_lin.local_out_features)
            tp_lin.register_buffer("weight_scale", scale)
        else:
            tp_lin.weight_scale = scale

    if attr_name.isdigit():
        parent[int(attr_name)] = tp_lin  # type: ignore
    else:
        setattr(parent, attr_name, tp_lin)


def _replace_mlp_linears(
    mlp: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Replace all ``nn.Linear`` layers inside a Flux MLP module.

    Handles three MLP variants:
      - ``nn.Sequential(Linear, GELU, Linear)``            (default)
      - ``nn.Sequential(Linear, SiLU, Linear)``            (mlp_silu_act)
      - ``YakMLP`` with ``gate_proj / up_proj / down_proj`` (yak_mlp)
    """
    if isinstance(mlp, nn.Sequential):
        for idx, child in enumerate(mlp):
            if isinstance(child, nn.Linear):
                _replace_linear(mlp, str(idx), tp_group=tp_group)
    else:
        # YakMLP (or similar): replace named Linear children.
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp, name) and isinstance(getattr(mlp, name), nn.Linear):
                _replace_linear(mlp, name, tp_group=tp_group)


# ---------------------------------------------------------------------------
# Double-stream block
# ---------------------------------------------------------------------------

def apply_tp_to_flux_double_block(
    block: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """In-place TP patching of a Flux ``DoubleStreamBlock``.

    Replaces:
      img_attn.qkv / img_attn.proj  → TPLinear (column, gather=True)
      txt_attn.qkv / txt_attn.proj  → TPLinear (column, gather=True)
      img_mlp / txt_mlp linears      → TPLinear (column, gather=True)
    """
    # Attention projections (both streams)
    for attn_name in ("img_attn", "txt_attn"):
        attn = getattr(block, attn_name, None)
        if attn is None:
            continue
        _replace_linear(attn, "qkv", tp_group=tp_group)
        _replace_linear(attn, "proj", tp_group=tp_group)

    # MLP (both streams)
    for mlp_name in ("img_mlp", "txt_mlp"):
        mlp = getattr(block, mlp_name, None)
        if mlp is None:
            continue
        _replace_mlp_linears(mlp, tp_group=tp_group)


# ---------------------------------------------------------------------------
# Single-stream block
# ---------------------------------------------------------------------------

def apply_tp_to_flux_single_block(
    block: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """In-place TP patching of a Flux ``SingleStreamBlock``.

    ``linear1`` is a fused QKV+MLP projection; ``linear2`` is a fused
    attn+MLP output projection.  Both become column-parallel with
    ``gather_output=True`` so the split/concat logic in ``forward()``
    sees unchanged shapes.
    """
    _replace_linear(block, "linear1", tp_group=tp_group)
    _replace_linear(block, "linear2", tp_group=tp_group)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def apply_tp_to_flux_model(
    model: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Full in-place TP patching of the Flux diffusion model.

    Patches every ``DoubleStreamBlock`` and ``SingleStreamBlock`` inside
    *model*.  Boundary projections (``img_in``, ``txt_in``,
    ``final_layer.linear``) are optionally patchable for extra weight savings
    but are left untouched here to minimise surface area.

    Args:
        model: The inner diffusion model (``comfy_model.diffusion_model``).
        tp_group: TP process group (defaults to global TP group).
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_group = tp_group or get_tp_group()

    # Double-stream blocks
    for block in getattr(model, "double_blocks", []):
        apply_tp_to_flux_double_block(block, tp_group=tp_group)

    # Single-stream blocks
    for block in getattr(model, "single_blocks", []):
        apply_tp_to_flux_single_block(block, tp_group=tp_group)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_tp_flux_weights(
    model: nn.Module,
    state_dict: dict[str, "torch.Tensor"],
    tp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Load a full (un-sharded) checkpoint into a TP-patched Flux model.

    Walks all modules.  For each ``TPLinear`` found, calls its
    ``weight_loader()`` which slices the correct rank's shard via
    ``narrow()``.  Non-TP parameters are copied directly.
    """
    import torch

    for name, module in model.named_modules():
        if isinstance(module, TPLinear):
            weight_key = name + ".weight"
            if weight_key in state_dict:
                module.weight_loader(module.weight, state_dict[weight_key])

            bias_key = name + ".bias"
            if bias_key in state_dict and module.bias is not None:
                if module.parallelism == "column":
                    module.weight_loader(module.bias, state_dict[bias_key])
                else:
                    module.bias.data.copy_(state_dict[bias_key])
