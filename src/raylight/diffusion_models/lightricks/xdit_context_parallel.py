import torch
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import raylight.distributed_modules.attention as xfuser_attn
from comfy.ldm.lightricks.model import apply_rotary_emb
from ..utils import pad_to_world_size
from .optimizations import prepare_pe_cached, RAYLIGHT_DEBUG, logger

attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


# A better solution is just to type.MethodType the x, context, and pe construction,
def pad_group_to_world_size(group, dim):
    """
    group: Tensor | list[Tensor] | tuple[Tensor]
    Returns: padded_group, orig_sizes
    """

    if torch.is_tensor(group):
        orig = group.size(dim)
        group, _ = pad_to_world_size(group, dim=dim)
        return group, orig

    elif isinstance(group, (list, tuple)):
        padded = []
        origs = []
        for g in group:
            o = g.size(dim)
            g, _ = pad_to_world_size(g, dim=dim)
            padded.append(g)
            origs.append(o)
        return type(group)(padded), origs

    else:
        return group, None


def pad_and_split_pe(pe, dim, sp_world_size, sp_rank):
    """
    LTX-2 PE is a list of tuples: [(cos, sin, flag), ...]
    cos/sin have shape (B, H, T, D)
    """
    out = []
    for cos, sin, flag in pe:
        # Pad
        cos_padded, _ = pad_to_world_size(cos, dim=dim)
        sin_padded, _ = pad_to_world_size(sin, dim=dim)

        # Split (sequence parallel)
        cos_chunk = torch.chunk(cos_padded, sp_world_size, dim=dim)[sp_rank]
        sin_chunk = torch.chunk(sin_padded, sp_world_size, dim=dim)[sp_rank]

        out.append((cos_chunk, sin_chunk, flag))

    return out


def sp_chunk_group(group, sp_world_size, sp_rank, dim):
    if torch.is_tensor(group):
        return torch.chunk(group, sp_world_size, dim=dim)[sp_rank]

    elif isinstance(group, (list, tuple)):
        return type(group)(
            torch.chunk(g, sp_world_size, dim=dim)[sp_rank]
            for g in group
        )
    else:
        return group


def sp_gather_group(group, orig_sizes, dim):
    if torch.is_tensor(group):
        g = torch.as_tensor(get_sp_group().all_gather(group.contiguous(), dim=dim))
        return g.narrow(dim, 0, orig_sizes)

    elif isinstance(group, (list, tuple)):
        out = []
        for g, o in zip(group, orig_sizes):
            g_out = torch.as_tensor(get_sp_group().all_gather(g.contiguous(), dim=dim))
            g_out = g_out.narrow(dim, 0, o)
            out.append(g_out)
        return type(group)(out)

    else:
        return group


def process_usp_timestep(timestep, sp_rank, sp_world_size):
    """
    Optimized timestep processing for Sequence Parallel.
    Avoids materializing the full expanded timestep tensor on every worker.
    """
    if isinstance(timestep, (list, tuple)):
        return type(timestep)(
            process_usp_timestep(t, sp_rank, sp_world_size) for t in timestep
        )
        
    elif torch.is_tensor(timestep):
        # Optimized Slicing for standard Tensors
        orig_len = timestep.size(1)
        padded_len = ((orig_len + sp_world_size - 1) // sp_world_size) * sp_world_size
        chunk_size = padded_len // sp_world_size
        
        start = sp_rank * chunk_size
        end = (sp_rank + 1) * chunk_size
        
        # All beyond original -> all padding
        if start >= orig_len:
            pad_len = end - start
            return timestep[:, -1:, :].expand(-1, pad_len, -1).clone()
            
        # Entirely within original -> just slice
        if end <= orig_len:
            return timestep[:, start:end, :].clone()
        
        # Spans across boundary -> slice + pad
        real_part = timestep[:, start:, :]
        pad_len = end - orig_len
        padding = timestep[:, -1:, :].expand(-1, pad_len, -1)
        return torch.cat([real_part, padding], dim=1)

    elif hasattr(timestep, "expand"):  # CompressedTimestep (LTXAV-specific)
        ts = timestep
        # Expanded length is frames * patches_per_frame
        total_len = ts.num_frames * ts.patches_per_frame
        padded_len = ((total_len + sp_world_size - 1) // sp_world_size) * sp_world_size
        chunk_size = padded_len // sp_world_size
        
        start_idx = sp_rank * chunk_size
        end_idx = (sp_rank + 1) * chunk_size
        
        # Optimized Expansion: create local indices and gather frame data
        # This replaces the OOM-prone ts.expand() call
        indices = torch.arange(start_idx, end_idx, device=ts.data.device)
        # Handle padding by repeating last frame's data
        valid_indices = torch.clamp(indices, max=total_len - 1)
        frame_indices = valid_indices // ts.patches_per_frame
        
        # Gather relevant frame data: [batch, chunk_size, feature_dim]
        # We use clone() to ensure it doesn't hold a reference to the full data if possible,
        # though slicing here is already a gather which creates a new tensor.
        return ts.data[:, frame_indices, :]
        
    else:
        return timestep


def usp_dit_forward(
    self,
    x,
    timestep,
    context,
    attention_mask,
    frame_rate=25,
    transformer_options={},
    keyframe_idxs=None,
    denoise_mask=None,
    **kwargs
):
    """
    Internal forward pass for LTX models.

    Args:
        x: Input tensor
        timestep: Timestep tensor
        context: Context tensor (e.g., text embeddings)
        attention_mask: Attention mask tensor
        frame_rate: Frame rate for temporal processing
        transformer_options: Additional options for transformer blocks
        keyframe_idxs: Keyframe indices for temporal processing
        **kwargs: Additional keyword arguments

    Returns:
        Processed output tensor
    """
    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if isinstance(x, list):
        input_dtype = x[0].dtype
        batch_size = x[0].shape[0]
    else:
        input_dtype = x.dtype
        batch_size = x.shape[0]
    # Process input
    merged_args = {**transformer_options, **kwargs}
    x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs, denoise_mask, **merged_args)
    merged_args.update(additional_args)

    # Prepare timestep and context
    res_timestep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
    timestep, embedded_timestep = res_timestep[:2]
    if len(res_timestep) > 2:
        merged_args["prompt_timestep"] = res_timestep[2]
    context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)

    # Prepare attention mask
    attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
    
    # Optimization: if attention_mask is trivial (all 0s), set to None to enable FlashAttention path.
    # Cache the result keyed by shape+device — the mask content is constant across denoising steps
    # for a given generation (same sequence lengths), so we only pay the GPU-CPU sync once.
    if attention_mask is not None:
        mask_cache_key = (attention_mask.shape, attention_mask.device)
        cached_key = getattr(self, "_trivial_mask_cache_key", None)
        if cached_key == mask_cache_key:
            # Same shape/device as last check — reuse cached result
            if self._trivial_mask_is_trivial:
                attention_mask = None
        else:
            # New shape/device (new generation or first call) — recompute
            is_trivial = not torch.any(attention_mask != 0).item()
            self._trivial_mask_cache_key = mask_cache_key
            self._trivial_mask_is_trivial = is_trivial
            if is_trivial:
                attention_mask = None

    # Build self-attention mask BEFORE padding and chunking x
    if hasattr(self, "_build_guide_self_attention_mask"):
        self_attention_mask = self._build_guide_self_attention_mask(
            x, transformer_options, merged_args
        )
    else:
        self_attention_mask = None

    # ======================== ADD SEQUENCE PARALLEL ========================= #
    # 1. Pad inputs to world size
    x, x_orig = pad_group_to_world_size(x, dim=1)
    context, _ = pad_group_to_world_size(context, dim=1)
    # Important: pad pixel_coords before slicing to ensure consistent chunking
    pixel_coords_padded, _ = pad_group_to_world_size(pixel_coords, dim=2) # (B, 1, T, 3) -> dim 2 is seq

    # 2. Slice inputs
    x = sp_chunk_group(x, sp_world_size, sp_rank, dim=1)
    context = sp_chunk_group(context, sp_world_size, sp_rank, dim=1)
    pixel_coords_chunk = sp_chunk_group(pixel_coords_padded, sp_world_size, sp_rank, dim=2)
    timestep = process_usp_timestep(timestep, sp_rank, sp_world_size)

    # 3. Generate Positional Embeddings only for this worker's chunk
    # This avoids the massive peak memory of generating full PE (e.g. 50GB for 900 frames)
    # PEs are cached across timesteps since coords and frame_rate don't change during generation
    pe = prepare_pe_cached(self, pixel_coords_chunk, frame_rate, input_dtype)
    if RAYLIGHT_DEBUG:
        if isinstance(pixel_coords_chunk, (list, tuple)):
            pc_shape = [tuple(p.shape) for p in pixel_coords_chunk]
        else:
            pc_shape = tuple(pixel_coords_chunk.shape)
        logger.info("[RayWorker %d] PE Generated for chunk: %s. VRAM: %.1fMB", sp_rank, pc_shape, torch.cuda.memory_allocated()/1024**2)
    # ======================== ADD SEQUENCE PARALLEL ========================= #

    # Process transformer blocks
    x = self._process_transformer_blocks(
        x, context, attention_mask, timestep, pe, transformer_options=transformer_options, self_attention_mask=self_attention_mask, **merged_args
    )
    if RAYLIGHT_DEBUG:
        logger.info("[RayWorker %d] Blocks complete. Peak VRAM: %.1fMB", sp_rank, torch.cuda.max_memory_allocated()/1024**2)
    
    x = sp_gather_group(x, x_orig, dim=1)

    # Process output
    x = self._process_output(x, embedded_timestep, keyframe_idxs, **merged_args)
    return x


def usp_cross_attn_forward(
        self,
        x,
        context=None,
        mask=None,
        pe=None,
        k_pe=None,
        transformer_options={}
):
    q = self.to_q(x)
    context = x if context is None else context
    k = self.to_k(context)
    v = self.to_v(context)

    q = self.q_norm(q)
    k = self.k_norm(k)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    # Optimization: Only pass mask if it's not None and not trivial
    # We check if it's None first as it's the fastest
    if mask is not None:
        # For LTX, masks are usually additive (0 for keep, -inf for mask)
        # We can do a very quick check: if it's all zeros, it's an identity mask.
        # However, to avoid frequent GPU-CPU syncs, we trust the top-level usp_dit_forward
        # which already cleaned up the global attention_mask.
        # But for self_attention_mask or others, we might want to check.
        pass

    # Pass metadata if available in transformer_options
    mod_idx = transformer_options.get("block_index", None)
    from raylight.distributed_modules.compact.main import compact_get_step
    
    out = xfuser_optimized_attention(
        q, k, v, self.heads, 
        mask=mask, 
        mod_idx=mod_idx,
        current_iter=compact_get_step()
    )

    # Apply per-head gating if enabled (LTX-specific)
    if getattr(self, "to_gate_logits", None) is not None:
        gate_logits = self.to_gate_logits(x)  # (B, T, H)
        b, t, _ = out.shape
        # self.heads and self.dim_head should be available on the patched module
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init -> identity
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)
