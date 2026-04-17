from __future__ import annotations

import torch
import torch.nn as nn
import types
from typing import Any, List, Tuple, Union

from comfy.ldm.lightricks.model import CrossAttention, LTXVModel
try:
    from comfy.ldm.lightricks.av_model import LTXAVModel, BasicAVTransformerBlock
except ImportError:
    LTXAVModel = None
    BasicAVTransformerBlock = None

import comfy.ldm.modules.attention as comfy_attn
from raylight.distributed_modules.attention.dispatcher import get_local_attn_fn
import raylight.raylight_types as raylight_types

# Check for apply_rotary_emb
try:
    from comfy.ldm.lightricks.model import apply_rotary_emb
except ImportError:
    try:
        from comfy.ldm.lightricks.symmetric_patchifier import apply_rotary_emb
    except ImportError:
        apply_rotary_emb = None

from raylight.distributed_modules.tensor_parallel import (
    TPLinear,
    TPFusedQKNorm,
    TPRMSNormAcrossHeads,
    TensorParallelState,
    get_tp_group,
)
from raylight.distributed_modules.tp_compress import TPCompressor, TPCompressConfig
from raylight.distributed_modules.tp_linear_factory import make_tp_linear


def _slice_rope_for_tp(
    pe: Any,
    tp_rank: int,
    tp_size: int,
    tp_local_heads: int,
) -> Any:
    """
    Slice RoPE positional embeddings for current rank.
    Handles both LTXAV [B, H, T, D] and Flux [T, 1, H, D] formats.
    """
    if pe is None:
        return None

    # Detect if we have a single RoPE descriptor or a list
    is_list = isinstance(pe, list)
    pe_list = pe if is_list else [pe]

    new_pe = []
    for item in pe_list:
        if len(item) == 3:
            cos, sin, split_mode = item
        else:
            cos, sin = item
            split_mode = False

        # Identify head dimension (H)
        # LTXAV shape: [B, H, T, D]  -> H is dim 1
        # Flux shape:  [T, 1, H, D] -> H is dim 2
        total_heads = tp_local_heads * tp_size
        h_dim = -1
        if cos.ndim == 4:
            if cos.shape[1] == total_heads:
                h_dim = 1
            elif cos.shape[2] == total_heads:
                h_dim = 2
            else:
                # Fallback to standard last-but-one head dimension
                h_dim = 1 if cos.shape[1] > 1 else 2
        
        # RoPE heads match model heads or are 1 (broadcast)
        if h_dim != -1 and cos.shape[h_dim] > 1:
            start = tp_rank * tp_local_heads
            cos_local = cos.narrow(h_dim, start, tp_local_heads)
            sin_local = sin.narrow(h_dim, start, tp_local_heads)
        else:
            # Broadcasted RoPE (1 head), just keep it
            cos_local = cos
            sin_local = sin
            
        if len(item) == 3:
            new_pe.append((cos_local, sin_local, split_mode))
        else:
            new_pe.append((cos_local, sin_local))

    return new_pe if is_list else new_pe[0]


def apply_tp_to_ltxav_cross_attention(
    attn: CrossAttention,
    tp_group=None,
    compress_config: TPCompressConfig | None = None,
    block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """
    In-place TP patching of a CrossAttention module for LTXAV.
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_rank = TensorParallelState.get_rank()

    # Infer dimensions from existing linears (use module attrs, not
    # weight.shape, so this works for both nn.Linear and GGMLOps.Linear).
    query_dim  = attn.to_q.in_features
    inner_dim  = attn.to_q.out_features      # heads * dim_head
    context_dim = attn.to_k.in_features

    local_inner_dim = inner_dim // tp_size
    local_heads = attn.heads // tp_size

    # 1. Replace QKV projections
    new_to_q = make_tp_linear(
        attn.to_q, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    new_to_k = make_tp_linear(
        attn.to_k, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    new_to_v = make_tp_linear(
        attn.to_v, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )

    attn.to_q = new_to_q
    attn.to_k = new_to_k
    attn.to_v = new_to_v

    # 2. Replace QK norms with a fused distributed variant (single all-reduce
    #    for both Q and K sum-of-squares instead of two separate calls).
    fused_qk_norm = TPFusedQKNorm(
        full_hidden_size=inner_dim,
        local_hidden_size=local_inner_dim,
        tp_group=tp_group,
        eps=attn.q_norm.eps,
    )

    if not structure_only:
        fused_qk_norm.weight_loader(fused_qk_norm.q_weight, attn.q_norm.weight.data)
        fused_qk_norm.weight_loader(fused_qk_norm.k_weight, attn.k_norm.weight.data)

    attn._fused_qk_norm = fused_qk_norm
    # Remove the old separate norms — forward no longer calls them.
    attn.q_norm = None
    attn.k_norm = None

    # 3. Replace output projection
    out_dim    = attn.to_out[0].out_features     # query_dim (output of to_out)
    _compressor = None
    if compress_config is not None and compress_config.enabled:
        _compressor = TPCompressor(
            config=compress_config,
            hidden_dim=out_dim,
            layer_id=block_idx * 10,  # unique per block, sub_idx=0 for attn out
            device=attn.to_out[0].weight.device,
        )
    new_to_out = make_tp_linear(
        attn.to_out[0], parallelism="row", tp_group=tp_group,
        input_is_parallel=True, reduce_results=True,
        structure_only=structure_only, compressor=_compressor,
    )

    attn.to_out[0] = new_to_out

    # 4. Store local_heads for use in the new forward method
    attn._tp_local_heads = local_heads
    attn._tp_rank = tp_rank
    attn._tp_size = tp_size

    # Gate logic handling
    if attn.to_gate_logits is not None:
        new_to_gate = make_tp_linear(
            attn.to_gate_logits, parallelism="column", gather_output=False,
            tp_group=tp_group, structure_only=structure_only,
        )
        attn.to_gate_logits = new_to_gate

    # 5. Replace forward with a TP-aware version
    @torch.compiler.disable
    def _tp_forward(self, x, context=None, mask=None, pe=None, k_pe=None,
                    transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
            
        q = self.to_q(x)            # [B, T, local_inner_dim]
        ctx = x if context is None else context

        # K/V cache: only for text cross-attention (attn2/audio_attn2, flagged
        # with _cache_kv=True).  k is returned already normed on cache hit.
        if getattr(self, '_cache_kv', False):
            from raylight.distributed_modules.utils import get_denoising_step
            step = get_denoising_step()
            cache = getattr(self, '_cross_kv_cache', None)
            if cache is not None and step is not None and step != 0:
                # Cache hit: k is pre-normed, only normalise q.
                k, v = cache
                (q,) = self._fused_qk_norm(q)
            else:
                # Cache miss (step 0): project, fused norm both, cache result.
                k = self.to_k(ctx)
                v = self.to_v(ctx)
                q, k = self._fused_qk_norm(q, k)
                self._cross_kv_cache = (k, v)
        else:
            k = self.to_k(ctx)
            v = self.to_v(ctx)
            # Fused path: single all-reduce for both Q and K sumsq.
            q, k = self._fused_qk_norm(q, k)

        if pe is not None and apply_rotary_emb is not None:
            pe_local = _slice_rope_for_tp(pe, self._tp_rank, self._tp_size, self._tp_local_heads)
            q = apply_rotary_emb(q, pe_local)
            kpe_src = k_pe if k_pe is not None else pe
            kpe_local = _slice_rope_for_tp(kpe_src, self._tp_rank, self._tp_size, self._tp_local_heads)
            k = apply_rotary_emb(k, kpe_local)

        # Ensure dtypes match for xformers/attention backends
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # Use Raylight's local attention dispatcher to respect global backend (e.g. SageAttention)
        if not hasattr(self, "_raylight_attn_fn"):
            self._raylight_attn_fn = get_local_attn_fn()
            
        out = self._raylight_attn_fn(
            q, k, v, self._tp_local_heads, mask=mask,
            attn_precision=self.attn_precision,
            transformer_options=transformer_options,
        )

        if self.to_gate_logits is not None:
            gate_logits_local = self.to_gate_logits(x)  # [B, T, local_heads]
            b, t, _ = out.shape
            out = out.view(b, t, self._tp_local_heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits_local)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self._tp_local_heads * self.dim_head)

        # to_out[0] is RowParallelLinear; it all-reduces internally
        out = self.to_out[0](out)   # → [B, T, out_dim], all-reduce done
        return self.to_out[1](out)  # Dropout

    attn.forward = types.MethodType(_tp_forward, attn)


def apply_tp_to_ltxav_feedforward(
    ff, tp_group=None, compress_config: TPCompressConfig | None = None, block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """
    In-place TP patching of a FeedForward module (ComfyUI LTXAV variant).
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    proj_in  = ff.net[0].proj   # GELU_approx.proj: Linear(dim → inner_dim)
    proj_out = ff.net[2]        # Linear(inner_dim → dim_out)

    dim      = proj_in.in_features
    inner_dim = proj_in.out_features
    dim_out  = proj_out.out_features

    # Replace IN projection
    new_in = make_tp_linear(
        proj_in, parallelism="column", gather_output=False,
        tp_group=tp_group, structure_only=structure_only,
    )
    ff.net[0].proj = new_in

    # Replace OUT projection
    _compressor = None
    if compress_config is not None and compress_config.enabled:
        _compressor = TPCompressor(
            config=compress_config,
            hidden_dim=dim_out,
            layer_id=block_idx * 10 + 1,  # sub_idx=1 for ffn out
            device=proj_out.weight.device,
        )
    new_out = make_tp_linear(
        proj_out, parallelism="row", tp_group=tp_group,
        input_is_parallel=True, reduce_results=True,
        structure_only=structure_only, compressor=_compressor,
    )
    ff.net[2] = new_out


def apply_tp_to_ltxav_block(
    block, tp_group=None, compress_config: TPCompressConfig | None = None, block_idx: int = 0,
    structure_only: bool = False,
) -> None:
    """
    Patch all attention and FFN submodules of one BasicAVTransformerBlock.
    """
    # Each attention sub-module within a block gets a unique sub_idx offset
    # via the attn_sub_offset so layer_ids don't collide.
    attn_sub_offset = 0
    for attn_attr in (
        "attn1",              # video self-attn
        "audio_attn1",        # audio self-attn
        "attn2",              # video→prompt cross-attn
        "audio_attn2",        # audio→prompt cross-attn
        "audio_to_video_attn",  # A2V (Q=video, K/V=audio)
        "video_to_audio_attn",  # V2A (Q=audio, K/V=video)
    ):
        if hasattr(block, attn_attr):
            apply_tp_to_ltxav_cross_attention(
                getattr(block, attn_attr), tp_group=tp_group,
                compress_config=compress_config,
                block_idx=block_idx * 100 + attn_sub_offset,
                structure_only=structure_only,
            )
            attn_sub_offset += 1

    if hasattr(block, "ff"):
        apply_tp_to_ltxav_feedforward(
            block.ff, tp_group=tp_group,
            compress_config=compress_config, block_idx=block_idx * 100 + 50,
            structure_only=structure_only,
        )
    if hasattr(block, "audio_ff"):
        apply_tp_to_ltxav_feedforward(
            block.audio_ff, tp_group=tp_group,
            compress_config=compress_config, block_idx=block_idx * 100 + 51,
            structure_only=structure_only,
        )

    # Enable K/V projection cache for text cross-attention modules.
    # Context is constant across denoising steps so to_k + k_norm + to_v
    # results can be reused after the first step (see get_cross_kv_cached).
    for _ca_attr in ("attn2", "audio_attn2"):
        if hasattr(block, _ca_attr):
            getattr(block, _ca_attr)._cache_kv = True

    # Note: We keep AdaLN tables and block-level norms replicated because 
    # activations are gathered between blocks in this TP implementation.
    # This ensures stability and compatibility with torch.compile.

    # Mark TP-patched blocks so _discover_compile_keys skips them.
    # LTXAV block forwards use `del` between graph breaks, causing Dynamo
    # resume-function guard invalidation (___stack0 deallocation) on every
    # step — an upstream PyTorch limitation that prevents stable compilation.
    # The funcol infrastructure in tensor_parallel.py is ready for when this
    # is resolved upstream.
    block._tp_patched = True


def apply_tp_to_ltxav_model(model: nn.Module, tp_group=None, compress_config: TPCompressConfig | None = None,
                            structure_only: bool = False) -> None:
    """
    Full in-place TP patching of the LTXAV or LTXV model.
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    def _replace_boundary_linear(parent, attr, parallelism="column"):
        if not hasattr(parent, attr) or getattr(parent, attr) is None:
            return
        lin = getattr(parent, attr)

        # We use gather_output=True for boundary layers to keep activations 
        # replicated between the blocks and the model boundaries.
        new_lin = make_tp_linear(
            lin, parallelism="column", gather_output=True,
            tp_group=tp_group, structure_only=structure_only,
        )
        setattr(parent, attr, new_lin)

    def _patch_adaln(parent, attr):
        if not hasattr(parent, attr) or getattr(parent, attr) is None:
            return
        adaln = getattr(parent, attr)
        # AdaLayerNormSingle.linear produces modulation params. 
        # We shard the output weights but gather the results so blocks see full modulations.
        _replace_boundary_linear(adaln, "linear", parallelism="column")

    # Boundary projections — Weights are sharded, but activations are gathered/replicated
    _replace_boundary_linear(model, "patchify_proj")
    _replace_boundary_linear(model, "audio_patchify_proj")
    _replace_boundary_linear(model, "proj_out")
    _replace_boundary_linear(model, "audio_proj_out")

    # Global AdaLN modules
    _patch_adaln(model, "adaln_single")
    _patch_adaln(model, "audio_adaln_single")
    _patch_adaln(model, "av_ca_video_scale_shift_adaln_single")
    _patch_adaln(model, "av_ca_audio_scale_shift_adaln_single")
    _patch_adaln(model, "av_ca_a2v_gate_adaln_single")
    _patch_adaln(model, "av_ca_v2a_gate_adaln_single")
    _patch_adaln(model, "prompt_adaln_single")
    _patch_adaln(model, "audio_prompt_adaln_single")

    # Note: Output norms are also kept as standard LayerNorms because 
    # the last block produces gathered/replicated activations.

    # Patch _prepare_context for LTXAV (to handle sharded vx/ax shapes)
    if hasattr(model, "transformer_blocks"):
        blocks = model.transformer_blocks
        if isinstance(blocks, nn.ModuleList):
            for idx, block in enumerate(blocks):
                apply_tp_to_ltxav_block(
                    block, tp_group=tp_group,
                    compress_config=compress_config, block_idx=idx,
                    structure_only=structure_only,
                )

    if LTXAVModel is not None and isinstance(model, LTXAVModel):
        def _tp_prepare_context(self, context, batch_size, x, attention_mask=None):
            # v_context_dim and a_context_dim are the split sizes.
            # We must use logical full dimensions here.
            v_context_dim = self.caption_channels if self.caption_proj_before_connector is False else self.inner_dim
            a_context_dim = self.caption_channels if self.caption_proj_before_connector is False else self.audio_inner_dim

            # Split the full context tensor correctly
            v_context, a_context = torch.split(
                context, [v_context_dim, a_context_dim], len(context.shape) - 1
            )
            
            # Prepare Video Context (replication of LTXVModel._prepare_context but with logical dim)
            if self.caption_proj_before_connector is False:
                v_context = self.caption_projection(v_context)
            v_context = v_context.view(batch_size, -1, self.inner_dim)
            
            # Prepare Audio Context
            if self.caption_proj_before_connector is False:
                a_context = self.audio_caption_projection(a_context)
            a_context = a_context.view(batch_size, -1, self.audio_inner_dim)
            
            # LTXAVModel expects [v_context, a_context], attention_mask
            return [v_context, a_context], attention_mask

        model._prepare_context = types.MethodType(_tp_prepare_context, model)


def load_tp_ltxav_weights(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    tp_group=None,
) -> None:
    """
    Load a full (un-sharded) checkpoint into a TP-patched LTXAV model.
    """
    tp_rank = TensorParallelState.get_rank()

    for name, param in model.named_parameters():
        if name in state_dict:
            loaded_weight = state_dict[name]
            
            # Find the module containing this parameter
            module_path = name.split(".")
            param_attr = module_path[-1]
            parent = model
            for p in module_path[:-1]:
                parent = getattr(parent, p)
                
            # If the parent is a TP layer, use its weight_loader
            if isinstance(parent, TPLinear) or (hasattr(parent, "weight_loader") and callable(parent.weight_loader)):
                 parent.weight_loader(param, loaded_weight)
            elif isinstance(parent, (TPRMSNormAcrossHeads, TPFusedQKNorm)):
                 parent.weight_loader(param, loaded_weight)
            else:
                 # Standard non-TP parameter
                 param.data.copy_(loaded_weight)

