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
    TPRMSNormAcrossHeads,
    TensorParallelState,
    get_tp_group,
)


def tp_rms_norm(x, full_dim, tp_group, eps=1e-6):
    """Distributed-aware RMSNorm."""
    # x: [..., local_D]
    # Sum squares locally
    local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
    # All-reduce to get global sum squares
    import torch.distributed as dist
    dist.all_reduce(local_sumsq, op=dist.ReduceOp.SUM, group=tp_group)
    # Variance is global sum squares / full_dim
    var = local_sumsq / float(full_dim)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype)


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
) -> None:
    """
    In-place TP patching of a CrossAttention module for LTXAV.
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    tp_rank = TensorParallelState.get_rank()

    # Infer dimensions from existing linears
    query_dim  = attn.to_q.weight.shape[1]   # in_features
    inner_dim  = attn.to_q.weight.shape[0]   # out_features = heads * dim_head
    context_dim = attn.to_k.weight.shape[1]
    out_dim    = attn.to_out[0].weight.shape[0]  # query_dim (output of to_out)

    local_inner_dim = inner_dim // tp_size
    local_heads = attn.heads // tp_size

    # 1. Replace QKV projections
    new_to_q = TPLinear(
        query_dim, inner_dim, bias=attn.to_q.bias is not None, parallelism="column",
        gather_output=False, tp_group=tp_group, dtype=attn.to_q.weight.dtype, device=attn.to_q.weight.device
    )
    new_to_k = TPLinear(
        context_dim, inner_dim, bias=attn.to_k.bias is not None, parallelism="column",
        gather_output=False, tp_group=tp_group, dtype=attn.to_k.weight.dtype, device=attn.to_k.weight.device
    )
    new_to_v = TPLinear(
        context_dim, inner_dim, bias=attn.to_v.bias is not None, parallelism="column",
        gather_output=False, tp_group=tp_group, dtype=attn.to_v.weight.dtype, device=attn.to_v.weight.device
    )

    new_to_q.weight_loader(new_to_q.weight, attn.to_q.weight.data)
    if attn.to_q.bias is not None:
        new_to_q.weight_loader(new_to_q.bias, attn.to_q.bias.data)
        
    new_to_k.weight_loader(new_to_k.weight, attn.to_k.weight.data)
    if attn.to_k.bias is not None:
        new_to_k.weight_loader(new_to_k.bias, attn.to_k.bias.data)
        
    new_to_v.weight_loader(new_to_v.weight, attn.to_v.weight.data)
    if attn.to_v.bias is not None:
        new_to_v.weight_loader(new_to_v.bias, attn.to_v.bias.data)

    # Lora copying for QKV
    if hasattr(attn.to_q, "lora_A"): new_to_q.lora_A = attn.to_q.lora_A
    if hasattr(attn.to_q, "lora_B"): new_to_q.lora_B = attn.to_q.lora_B
    if hasattr(attn.to_q, "lora_alpha"): new_to_q.lora_alpha = attn.to_q.lora_alpha
    if hasattr(attn.to_k, "lora_A"): new_to_k.lora_A = attn.to_k.lora_A
    if hasattr(attn.to_k, "lora_B"): new_to_k.lora_B = attn.to_k.lora_B
    if hasattr(attn.to_k, "lora_alpha"): new_to_k.lora_alpha = attn.to_k.lora_alpha
    if hasattr(attn.to_v, "lora_A"): new_to_v.lora_A = attn.to_v.lora_A
    if hasattr(attn.to_v, "lora_B"): new_to_v.lora_B = attn.to_v.lora_B
    if hasattr(attn.to_v, "lora_alpha"): new_to_v.lora_alpha = attn.to_v.lora_alpha

    # INT8 Support: preserve and shard weight scales
    for old_m, new_m in [(attn.to_q, new_to_q), (attn.to_k, new_to_k), (attn.to_v, new_to_v)]:
        if hasattr(old_m, "weight_scale"):
            new_m.weight_scale = old_m.weight_scale
        if hasattr(old_m, "scale"):
            new_m.scale = old_m.scale

    attn.to_q = new_to_q
    attn.to_k = new_to_k
    attn.to_v = new_to_v

    # 2. Replace QK norms with distributed variant
    new_q_norm = TPRMSNormAcrossHeads(
        full_hidden_size=inner_dim,
        local_hidden_size=local_inner_dim,
        tp_group=tp_group,
        eps=attn.q_norm.eps
    )
    new_k_norm = TPRMSNormAcrossHeads(
        full_hidden_size=inner_dim,
        local_hidden_size=local_inner_dim,
        tp_group=tp_group,
        eps=attn.k_norm.eps
    )
    
    # QK Norm parameter loading (not sharded inherently in weight loader but internally handles local vs full)
    new_q_norm.weight_loader(new_q_norm.weight, attn.q_norm.weight.data)
    new_k_norm.weight_loader(new_k_norm.weight, attn.k_norm.weight.data)

    attn.q_norm = new_q_norm
    attn.k_norm = new_k_norm

    # 3. Replace output projection
    new_to_out = TPLinear(
        inner_dim, out_dim, bias=attn.to_out[0].bias is not None, parallelism="row",
        input_is_parallel=True, reduce_results=True, tp_group=tp_group,
        dtype=attn.to_out[0].weight.dtype, device=attn.to_out[0].weight.device
    )
    new_to_out.weight_loader(new_to_out.weight, attn.to_out[0].weight.data)
    if attn.to_out[0].bias is not None and new_to_out.bias is not None:
        new_to_out.bias.data.copy_(attn.to_out[0].bias.data)
        
    if hasattr(attn.to_out[0], "lora_A"): new_to_out.lora_A = attn.to_out[0].lora_A
    if hasattr(attn.to_out[0], "lora_B"): new_to_out.lora_B = attn.to_out[0].lora_B
    if hasattr(attn.to_out[0], "lora_alpha"): new_to_out.lora_alpha = attn.to_out[0].lora_alpha

    if hasattr(attn.to_out[0], "weight_scale"):
        new_to_out.weight_scale = attn.to_out[0].weight_scale
    if hasattr(attn.to_out[0], "scale"):
        new_to_out.scale = attn.to_out[0].scale

    attn.to_out[0] = new_to_out

    # 4. Store local_heads for use in the new forward method
    attn._tp_local_heads = local_heads
    attn._tp_rank = tp_rank
    attn._tp_size = tp_size

    # Gate logic handling
    if attn.to_gate_logits is not None:
        gate_bias = attn.to_gate_logits.bias is not None
        # Using column parallel without output gather to correctly shard gates
        new_to_gate = TPLinear(
            query_dim, attn.heads, bias=gate_bias, parallelism="column",
            gather_output=False, tp_group=tp_group,
            dtype=attn.to_gate_logits.weight.dtype, device=attn.to_gate_logits.weight.device
        )
        new_to_gate.weight_loader(new_to_gate.weight, attn.to_gate_logits.weight.data)
        if gate_bias:
            new_to_gate.weight_loader(new_to_gate.bias, attn.to_gate_logits.bias.data)
        
        if hasattr(attn.to_gate_logits, "weight_scale"):
            new_to_gate.weight_scale = attn.to_gate_logits.weight_scale
        if hasattr(attn.to_gate_logits, "scale"):
            new_to_gate.scale = attn.to_gate_logits.scale
            
        attn.to_gate_logits = new_to_gate

    # 5. Replace forward with a TP-aware version
    @torch.compiler.disable
    def _tp_forward(self, x, context=None, mask=None, pe=None, k_pe=None,
                    transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
            
        q = self.to_q(x)            # [B, T, local_inner_dim]
        ctx = x if context is None else context
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        q = self.q_norm(q)
        k = self.k_norm(k)

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


def _patch_block_forward_for_tp(block, tp_group):
    """
    Monkey-patch the block's forward to use distributed-aware rms_norm.
    """
    import comfy.ldm.common_dit
    
    # We create a local scope version of rms_norm that uses the tp_group
    # Since we can't easily find every local reference, we replace the block's 
    # reference to the module's function if possible, or patch the forward logic.
    
    # Context discovery to handle both LTXV and LTXAV blocks
    is_av = hasattr(block, "audio_attn1")
    
    def _tp_rms_norm_wrapper(x, full_dim_override=None):
        # Infer full dimension from x and tp_size
        tp_size = TensorParallelState.get_size()
        f_dim = full_dim_override or (x.shape[-1] * tp_size)
        return tp_rms_norm(x, f_dim, tp_group)

    if is_av:
        # For BasicAVTransformerBlock
        original_forward = block.forward
        
        @torch.compiler.disable
        def _tp_av_forward(self, *args, **kwargs):
            # Temporarily shadow rms_norm in the global scope of the function? 
            # Riskier than just rewriting the calls.
            # But the AV block uses it many times. Let's redirect it.
            import comfy.ldm.common_dit
            old_rms = comfy.ldm.common_dit.rms_norm
            comfy.ldm.common_dit.rms_norm = _tp_rms_norm_wrapper
            try:
                return original_forward(*args, **kwargs)
            finally:
                comfy.ldm.common_dit.rms_norm = old_rms
                
        block.forward = types.MethodType(_tp_av_forward, block)
    else:
        # For standard BasicTransformerBlock
        original_forward = block.forward
        @torch.compiler.disable
        def _tp_v_forward(self, *args, **kwargs):
            import comfy.ldm.common_dit
            old_rms = comfy.ldm.common_dit.rms_norm
            comfy.ldm.common_dit.rms_norm = _tp_rms_norm_wrapper
            try:
                return original_forward(*args, **kwargs)
            finally:
                comfy.ldm.common_dit.rms_norm = old_rms
        block.forward = types.MethodType(_tp_v_forward, block)


def apply_tp_to_ltxav_feedforward(ff, tp_group=None) -> None:
    """
    In-place TP patching of a FeedForward module (ComfyUI LTXAV variant).
    """
    tp_size = TensorParallelState.get_size()
    if tp_size <= 1:
        return

    proj_in  = ff.net[0].proj   # GELU_approx.proj: Linear(dim → inner_dim)
    proj_out = ff.net[2]        # Linear(inner_dim → dim_out)

    dim      = proj_in.weight.shape[1]
    inner_dim = proj_in.weight.shape[0]
    dim_out  = proj_out.weight.shape[0]

    # Replace IN projection
    new_in = TPLinear(
        dim, inner_dim, bias=proj_in.bias is not None, parallelism="column",
        gather_output=False, tp_group=tp_group,
        dtype=proj_in.weight.dtype, device=proj_in.weight.device
    )
    new_in.weight_loader(new_in.weight, proj_in.weight.data)
    if proj_in.bias is not None:
        new_in.weight_loader(new_in.bias, proj_in.bias.data)
        
    if hasattr(proj_in, "lora_A"): new_in.lora_A = proj_in.lora_A
    if hasattr(proj_in, "lora_B"): new_in.lora_B = proj_in.lora_B
    if hasattr(proj_in, "lora_alpha"): new_in.lora_alpha = proj_in.lora_alpha
    
    if hasattr(proj_in, "weight_scale"):
        new_in.weight_scale = proj_in.weight_scale
    if hasattr(proj_in, "scale"):
        new_in.scale = proj_in.scale
        
    ff.net[0].proj = new_in

    # Replace OUT projection
    new_out = TPLinear(
        inner_dim, dim_out, bias=proj_out.bias is not None, parallelism="row",
        input_is_parallel=True, reduce_results=True, tp_group=tp_group,
        dtype=proj_out.weight.dtype, device=proj_out.weight.device
    )
    new_out.weight_loader(new_out.weight, proj_out.weight.data)
    if proj_out.bias is not None and new_out.bias is not None:
        new_out.bias.data.copy_(proj_out.bias.data)
        
    if hasattr(proj_out, "lora_A"): new_out.lora_A = proj_out.lora_A
    if hasattr(proj_out, "lora_B"): new_out.lora_B = proj_out.lora_B
    if hasattr(proj_out, "lora_alpha"): new_out.lora_alpha = proj_out.lora_alpha
    
    if hasattr(proj_out, "weight_scale"):
        new_out.weight_scale = proj_out.weight_scale
    if hasattr(proj_out, "scale"):
        new_out.scale = proj_out.scale
        
    ff.net[2] = new_out


def apply_tp_to_ltxav_block(block, tp_group=None) -> None:
    """
    Patch all attention and FFN submodules of one BasicAVTransformerBlock.
    """
    for attn_attr in (
        "attn1",              # video self-attn
        "audio_attn1",        # audio self-attn
        "attn2",              # video→prompt cross-attn
        "audio_attn2",        # audio→prompt cross-attn
        "audio_to_video_attn",  # A2V (Q=video, K/V=audio)
        "video_to_audio_attn",  # V2A (Q=audio, K/V=video)
    ):
        if hasattr(block, attn_attr):
            apply_tp_to_ltxav_cross_attention(getattr(block, attn_attr), tp_group=tp_group)

    if hasattr(block, "ff"):
        apply_tp_to_ltxav_feedforward(block.ff, tp_group=tp_group)
    if hasattr(block, "audio_ff"):
        apply_tp_to_ltxav_feedforward(block.audio_ff, tp_group=tp_group)

    # Note: We keep AdaLN tables and block-level norms replicated because 
    # activations are gathered between blocks in this TP implementation.
    # This ensures stability and compatibility with torch.compile.


def apply_tp_to_ltxav_model(model: nn.Module, tp_group=None) -> None:
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
        in_f, out_f = lin.weight.shape[1], lin.weight.shape[0]
        has_bias = lin.bias is not None
        
        # We use gather_output=True for boundary layers to keep activations 
        # replicated between the blocks and the model boundaries.
        new_lin = TPLinear(
            in_f, out_f, bias=has_bias, parallelism="column",
            gather_output=True, tp_group=tp_group,
            dtype=lin.weight.dtype, device=lin.weight.device
        )
        new_lin.weight_loader(new_lin.weight, lin.weight.data)
        if has_bias:
            new_lin.weight_loader(new_lin.bias, lin.bias.data)
        
        if hasattr(lin, "lora_A"): new_lin.lora_A = lin.lora_A
        if hasattr(lin, "lora_B"): new_lin.lora_B = lin.lora_B
        if hasattr(lin, "lora_alpha"): new_lin.lora_alpha = lin.lora_alpha
        
        if hasattr(lin, "weight_scale"):
            new_lin.weight_scale = lin.weight_scale
        if hasattr(lin, "scale"):
            new_lin.scale = lin.scale
            
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
            for block in blocks:
                apply_tp_to_ltxav_block(block, tp_group=tp_group)

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
            elif isinstance(parent, TPRMSNormAcrossHeads):
                 parent.weight_loader(param, loaded_weight)
            else:
                 # Standard non-TP parameter
                 param.data.copy_(loaded_weight)

