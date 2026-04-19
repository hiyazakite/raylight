"""Tensor Parallelism for WAN models (Phase 2.2 — true head-sharded TP).

Uses column-parallel QKV projections with ``gather_output=False`` so that
attention runs on local head shards only, giving both weight-memory and
compute savings.  Output projections are row-parallel with all-reduce.

Supports all WAN attention variants:
  - WanSelfAttention          (self-attention with RoPE)
  - WanT2VCrossAttention      (text→video cross-attention)
  - WanI2VCrossAttention      (image→video cross-attention, extra k_img/v_img)
  - WanT2VCrossAttentionGather (HuMo audio cross-attention)

References:
  - LTXAV TP:  diffusion_models/lightricks/tp.py
  - sglang:    runtime/layers/linear.py
  - Megatron:  megatron/core/tensor_parallel/layers.py
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from raylight.distributed_modules.tensor_parallel import (
    TPFusedQKNorm,
    TPLinear,
    TPRMSNormAcrossHeads,
    TensorParallelState,
    get_tp_group,
)
from raylight.distributed_modules.tp_compress import TPCompressConfig, TPCompressor
from raylight.distributed_modules.tp_linear_factory import make_tp_linear


# =============================================================================
# RoPE helper
# =============================================================================


def _wan_rope_needs_slicing(freqs: torch.Tensor) -> bool:
    """Check whether WAN RoPE frequencies need per-head slicing.

    WAN RoPE has shape ``[B, L, 1, D/2, 2, 2]`` where the head dimension
    is broadcast (=1).  In this case no slicing is needed — each rank can
    use the full freqs tensor.  If a future WAN variant expands per-head
    RoPE, this function detects it.
    """
    if freqs is None:
        return False
    # Standard WAN: dim-2 is 1 (broadcast)
    if freqs.ndim >= 3 and freqs.shape[2] == 1:
        return False
    return True


# =============================================================================
# Self-attention
# =============================================================================


def apply_tp_to_wan_self_attention(
    attn: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    compress_config: Optional[TPCompressConfig] = None,
    block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a ``WanSelfAttention`` module.

    Replaces:
      q/k/v   → TPLinear (column, gather_output=False)
      norm_q/norm_k → TPFusedQKNorm (single all-reduce)
      o       → TPLinear (row, all-reduce)
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_rank = TensorParallelState.get_rank()

    dim = attn.q.in_features
    local_dim = dim // tp_size
    local_heads = attn.num_heads // tp_size

    # 1. Replace QKV projections — column-parallel, output stays sharded
    attn.q = make_tp_linear(
        attn.q, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    attn.k = make_tp_linear(
        attn.k, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    attn.v = make_tp_linear(
        attn.v, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )

    # 2. Replace QK norms with fused distributed variant
    #    (single all-reduce for both Q and K sum-of-squares)
    qk_norm_active = attn.qk_norm if hasattr(attn, "qk_norm") else True
    if qk_norm_active and not isinstance(attn.norm_q, nn.Identity):
        eps = attn.norm_q.eps if hasattr(attn.norm_q, "eps") else 1e-5
        fused_qk_norm = TPFusedQKNorm(
            full_hidden_size=dim,
            local_hidden_size=local_dim,
            tp_group=tp_group,
            eps=eps,
        )
        if not structure_only:
            fused_qk_norm.weight_loader(fused_qk_norm.q_weight, attn.norm_q.weight.data)
            fused_qk_norm.weight_loader(fused_qk_norm.k_weight, attn.norm_k.weight.data)

        attn._fused_qk_norm = fused_qk_norm
        attn.norm_q = None
        attn.norm_k = None
    else:
        attn._fused_qk_norm = None

    # 3. Replace output projection — row-parallel with all-reduce
    _compressor = None
    if compress_config is not None and compress_config.enabled:
        _compressor = TPCompressor(
            config=compress_config,
            hidden_dim=dim,
            layer_id=block_idx * 10,
            device=attn.o.weight.device,
        )
    attn.o = make_tp_linear(
        attn.o, parallelism="row", tp_group=tp_group,
        input_is_parallel=True, reduce_results=True,
        structure_only=structure_only, compressor=_compressor,
    )

    # 4. Store TP metadata
    attn._tp_local_heads = local_heads
    attn._tp_rank = tp_rank
    attn._tp_size = tp_size

    # 5. Replace forward
    @torch.compiler.disable
    def _tp_self_attn_forward(self, x, freqs, transformer_options={}):
        from comfy.ldm.modules.attention import optimized_attention
        from comfy.ldm.wan.model import apply_rope1

        b, s, n, d = *x.shape[:2], self._tp_local_heads, self.head_dim
        input_dtype = x.dtype

        q = self.q(x)
        k = self.k(x)

        # Fused QK-norm (returns fp32 for precision; cast back to input dtype)
        if self._fused_qk_norm is not None:
            q, k = self._fused_qk_norm(q, k)
            q = q.to(input_dtype)
            k = k.to(input_dtype)

        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)

        # RoPE — WAN freqs are broadcast (head dim = 1), no slicing needed
        q = apply_rope1(q, freqs)
        k = apply_rope1(k, freqs)

        v = self.v(x).view(b, s, n * d)

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            v,
            heads=self._tp_local_heads,
            transformer_options=transformer_options,
        )

        x = self.o(x)  # row-parallel → all-reduce
        return x

    attn.forward = types.MethodType(_tp_self_attn_forward, attn)


# =============================================================================
# T2V Cross-attention
# =============================================================================


def apply_tp_to_wan_t2v_cross_attention(
    attn: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    compress_config: Optional[TPCompressConfig] = None,
    block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of ``WanT2VCrossAttention``.

    Same Q/K/V/O/norm pattern as self-attention but no RoPE.
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_rank = TensorParallelState.get_rank()
    dim = attn.q.in_features
    local_dim = dim // tp_size
    local_heads = attn.num_heads // tp_size

    # QKV — column-parallel
    attn.q = make_tp_linear(
        attn.q, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    attn.k = make_tp_linear(
        attn.k, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    attn.v = make_tp_linear(
        attn.v, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )

    # Fused QK-norm
    qk_norm_active = attn.qk_norm if hasattr(attn, "qk_norm") else True
    if qk_norm_active and not isinstance(attn.norm_q, nn.Identity):
        eps = attn.norm_q.eps if hasattr(attn.norm_q, "eps") else 1e-5
        fused_qk_norm = TPFusedQKNorm(
            full_hidden_size=dim,
            local_hidden_size=local_dim,
            tp_group=tp_group,
            eps=eps,
        )
        if not structure_only:
            fused_qk_norm.weight_loader(fused_qk_norm.q_weight, attn.norm_q.weight.data)
            fused_qk_norm.weight_loader(fused_qk_norm.k_weight, attn.norm_k.weight.data)

        attn._fused_qk_norm = fused_qk_norm
        attn.norm_q = None
        attn.norm_k = None
    else:
        attn._fused_qk_norm = None

    # Output — row-parallel
    _compressor = None
    if compress_config is not None and compress_config.enabled:
        _compressor = TPCompressor(
            config=compress_config,
            hidden_dim=dim,
            layer_id=block_idx * 10 + 1,
            device=attn.o.weight.device,
        )
    attn.o = make_tp_linear(
        attn.o, parallelism="row", tp_group=tp_group,
        input_is_parallel=True, reduce_results=True,
        structure_only=structure_only, compressor=_compressor,
    )

    # TP metadata
    attn._tp_local_heads = local_heads
    attn._tp_rank = tp_rank
    attn._tp_size = tp_size

    # Forward
    @torch.compiler.disable
    def _tp_t2v_cross_attn_forward(self, x, context, transformer_options={}, **kwargs):
        from comfy.ldm.modules.attention import optimized_attention

        input_dtype = x.dtype
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)

        if self._fused_qk_norm is not None:
            q, k = self._fused_qk_norm(q, k)
            q = q.to(input_dtype)
            k = k.to(input_dtype)

        x = optimized_attention(
            q, k, v,
            heads=self._tp_local_heads,
            transformer_options=transformer_options,
        )

        x = self.o(x)
        return x

    attn.forward = types.MethodType(_tp_t2v_cross_attn_forward, attn)


# =============================================================================
# I2V Cross-attention
# =============================================================================


def apply_tp_to_wan_i2v_cross_attention(
    attn: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    compress_config: Optional[TPCompressConfig] = None,
    block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of ``WanI2VCrossAttention``.

    Like T2V cross-attention but adds ``k_img``, ``v_img``, ``norm_k_img``
    for CLIP image features.
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_rank = TensorParallelState.get_rank()
    dim = attn.q.in_features
    local_dim = dim // tp_size
    local_heads = attn.num_heads // tp_size

    # QKV — column-parallel
    for attr in ("q", "k", "v", "k_img", "v_img"):
        if hasattr(attn, attr):
            setattr(attn, attr, make_tp_linear(
                getattr(attn, attr), parallelism="column", gather_output=False,
                tp_group=tp_group, structure_only=structure_only,
            ))

    # Fused QK-norm for text Q and K
    qk_norm_active = attn.qk_norm if hasattr(attn, "qk_norm") else True
    if qk_norm_active and not isinstance(attn.norm_q, nn.Identity):
        eps = attn.norm_q.eps if hasattr(attn.norm_q, "eps") else 1e-5

        fused_qk_norm = TPFusedQKNorm(
            full_hidden_size=dim,
            local_hidden_size=local_dim,
            tp_group=tp_group,
            eps=eps,
        )
        if not structure_only:
            fused_qk_norm.weight_loader(fused_qk_norm.q_weight, attn.norm_q.weight.data)
            fused_qk_norm.weight_loader(fused_qk_norm.k_weight, attn.norm_k.weight.data)
        attn._fused_qk_norm = fused_qk_norm
        attn.norm_q = None
        attn.norm_k = None

        # Separate norm for k_img (simpler than 3-way fuse)
        norm_k_img_tp = TPRMSNormAcrossHeads(
            full_hidden_size=dim,
            local_hidden_size=local_dim,
            tp_group=tp_group,
            eps=eps,
        )
        if not structure_only:
            norm_k_img_tp.weight_loader(norm_k_img_tp.weight, attn.norm_k_img.weight.data)
        attn.norm_k_img = norm_k_img_tp
    else:
        attn._fused_qk_norm = None

    # Output — row-parallel
    _compressor = None
    if compress_config is not None and compress_config.enabled:
        _compressor = TPCompressor(
            config=compress_config,
            hidden_dim=dim,
            layer_id=block_idx * 10 + 2,
            device=attn.o.weight.device,
        )
    attn.o = make_tp_linear(
        attn.o, parallelism="row", tp_group=tp_group,
        input_is_parallel=True, reduce_results=True,
        structure_only=structure_only, compressor=_compressor,
    )

    # TP metadata
    attn._tp_local_heads = local_heads
    attn._tp_rank = tp_rank
    attn._tp_size = tp_size

    # Forward
    @torch.compiler.disable
    def _tp_i2v_cross_attn_forward(self, x, context, context_img_len, transformer_options={}, **kwargs):
        from comfy.ldm.modules.attention import optimized_attention

        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]

        input_dtype = x.dtype
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        k_img = self.k_img(context_img)
        v_img = self.v_img(context_img)

        if self._fused_qk_norm is not None:
            q, k = self._fused_qk_norm(q, k)
            q = q.to(input_dtype)
            k = k.to(input_dtype)
        k_img = self.norm_k_img(k_img)

        img_x = optimized_attention(
            q, k_img, v_img,
            heads=self._tp_local_heads,
            transformer_options=transformer_options,
        )
        x = optimized_attention(
            q, k, v,
            heads=self._tp_local_heads,
            transformer_options=transformer_options,
        )

        x = x + img_x
        x = self.o(x)
        return x

    attn.forward = types.MethodType(_tp_i2v_cross_attn_forward, attn)


# =============================================================================
# FFN
# =============================================================================


def apply_tp_to_wan_ffn(
    ffn: nn.Sequential,
    tp_group: Optional[dist.ProcessGroup] = None,
    compress_config: Optional[TPCompressConfig] = None,
    block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """In-place TP patching of a WAN FFN ``nn.Sequential(Linear, GELU, Linear)``.

    - ``ffn[0]`` (in-proj) → column-parallel, gather_output=False
    - ``ffn[1]`` (GELU) → untouched
    - ``ffn[2]`` (out-proj) → row-parallel, all-reduce
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    # In-projection: column-parallel
    ffn[0] = make_tp_linear(
        ffn[0], parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )

    # Out-projection: row-parallel with all-reduce
    _compressor = None
    if compress_config is not None and compress_config.enabled:
        dim = ffn[2].out_features
        _compressor = TPCompressor(
            config=compress_config,
            hidden_dim=dim,
            layer_id=block_idx * 10 + 5,
            device=ffn[2].weight.device,
        )
    ffn[2] = make_tp_linear(
        ffn[2], parallelism="row", tp_group=tp_group,
        input_is_parallel=True, reduce_results=True,
        structure_only=structure_only, compressor=_compressor,
    )


# =============================================================================
# Block-level patching
# =============================================================================


def apply_tp_to_wan_block(
    block: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    compress_config: Optional[TPCompressConfig] = None,
    block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """Patch all attention and FFN submodules of one ``WanAttentionBlock``.

    Dispatches to the correct attention patcher based on the cross-attention
    type (T2V vs I2V).  Also handles ``WanAttentionBlockAudio`` (HuMo) which
    adds an ``audio_cross_attn_wrapper`` with an inner ``audio_cross_attn``.
    """
    from comfy.ldm.wan.model import (
        WanSelfAttention,
        WanT2VCrossAttention,
        WanI2VCrossAttention,
    )

    # Self-attention
    if hasattr(block, "self_attn") and isinstance(block.self_attn, WanSelfAttention):
        apply_tp_to_wan_self_attention(
            block.self_attn, tp_group=tp_group,
            compress_config=compress_config,
            block_idx=block_idx * 100,
            structure_only=structure_only,
        )

    # Cross-attention — dispatch by type
    if hasattr(block, "cross_attn"):
        if isinstance(block.cross_attn, WanI2VCrossAttention):
            apply_tp_to_wan_i2v_cross_attention(
                block.cross_attn, tp_group=tp_group,
                compress_config=compress_config,
                block_idx=block_idx * 100 + 1,
                structure_only=structure_only,
            )
        elif isinstance(block.cross_attn, WanT2VCrossAttention):
            apply_tp_to_wan_t2v_cross_attention(
                block.cross_attn, tp_group=tp_group,
                compress_config=compress_config,
                block_idx=block_idx * 100 + 1,
                structure_only=structure_only,
            )

    # HuMo: audio cross-attention (via AudioCrossAttentionWrapper)
    if hasattr(block, "audio_cross_attn_wrapper"):
        wrapper = block.audio_cross_attn_wrapper
        inner_attn = getattr(wrapper, "audio_cross_attn", None)
        if inner_attn is not None and isinstance(inner_attn, WanSelfAttention):
            # WanT2VCrossAttentionGather inherits from WanSelfAttention
            # and has the same q/k/v/o/norm structure
            apply_tp_to_wan_t2v_cross_attention(
                inner_attn, tp_group=tp_group,
                compress_config=compress_config,
                block_idx=block_idx * 100 + 2,
                structure_only=structure_only,
            )

    # FFN
    if hasattr(block, "ffn") and isinstance(block.ffn, nn.Sequential):
        apply_tp_to_wan_ffn(
            block.ffn, tp_group=tp_group,
            compress_config=compress_config,
            block_idx=block_idx * 100 + 5,
            structure_only=structure_only,
        )

    # Mark as TP-patched so _discover_compile_keys skips this block
    block._tp_patched = True


# =============================================================================
# Full model patching
# =============================================================================


def apply_tp_to_wan_model(
    model: nn.Module,
    tp_group: Optional[dist.ProcessGroup] = None,
    compress_config: Optional[TPCompressConfig] = None,
    structure_only: bool = False,
) -> None:
    """Full in-place TP patching of a WAN diffusion model.

    Patches all ``WanAttentionBlock`` instances in ``model.blocks``.
    Boundary projections (text_embedding, time_embedding, head) are left
    replicated — they are small relative to the transformer blocks.

    Args:
        model: The inner diffusion model (``comfy_model.diffusion_model``).
        tp_group: TP process group (defaults to global TP group).
        compress_config: Optional compression config for TP all-reduce.
        structure_only: Create TPLinear shells without loading weights
            (used by TPContext streaming path).
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_group = tp_group or get_tp_group()

    # Patch all transformer blocks
    blocks = getattr(model, "blocks", [])
    for idx, block in enumerate(blocks):
        apply_tp_to_wan_block(
            block, tp_group=tp_group,
            compress_config=compress_config,
            block_idx=idx,
            structure_only=structure_only,
        )

    # VACE blocks (VaceWanModel has vace_blocks that mirror the main blocks)
    vace_blocks = getattr(model, "vace_blocks", None)
    if vace_blocks is not None:
        for idx, block in enumerate(vace_blocks):
            apply_tp_to_wan_block(
                block, tp_group=tp_group,
                compress_config=compress_config,
                block_idx=1000 + idx,  # offset to avoid layer_id collision
                structure_only=structure_only,
            )

    print(f"[WAN-TP] Patched {len(blocks)} blocks"
          f"{f' + {len(vace_blocks)} vace_blocks' if vace_blocks else ''}"
          f" (tp_size={tp_size})")
