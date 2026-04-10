from __future__ import annotations

import collections
import logging
import gc

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.utils import _free_storage
from torch.distributed.tensor import DTensor
from typing import cast
import torch.nn as nn

import comfy
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import (get_key_weight,
                                 string_to_seed,
                                 move_weight_functions)

from raylight import comfy_dist
from .fsdp_registry import patch_fsdp


class LowVramPatch:
    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.convert_func = convert_func  # Keep for API compatibility
        self.set_func = set_func  # Keep for API compatibility

    def __call__(self, weight):
        if self.key not in self.patches:
            return weight
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:   # intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return comfy.float.stochastic_rounding(comfy_dist.lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))

        return comfy_dist.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


def _extract_lora_ab(patches):
    """Extract (down, up, scale) triples from a list of LoRA patches.

    Returns ``(list_of_A, list_of_B, dora_scale_or_None)`` on success where
    scale is baked into A, or ``(None, None, None)`` if any patch is
    unsupported (LoCon, offset, custom function, or non-LoRA type).

    DoRA (dora_scale) is supported: when present the scale is extracted and
    returned separately.  All patches in the list must agree (all have
    dora_scale or none do) — mixed sets bail out.
    """
    all_A = []  # [rank, in_features] each
    all_B = []  # [out_features, rank] each
    dora_scales = []  # one per patch, None when absent

    for p in patches:
        strength = float(p[0])
        adapter = p[1]
        offset = p[3]
        function = p[4]

        if offset is not None or function is not None:
            return None, None, None

        if hasattr(adapter, "weights"):
            v = adapter.weights
        elif isinstance(adapter, (tuple, list)) and len(adapter) >= 2:
            v = adapter
        else:
            return None, None, None

        up = v[0]       # mat1 / lora_up  — [full_out, rank]
        down = v[1]     # mat2 / lora_down — [rank, full_in]
        alpha = v[2] if len(v) > 2 else None
        mid = v[3] if len(v) > 3 else None
        dora_scale = v[4] if len(v) > 4 else None

        if mid is not None:
            return None, None, None

        rank = down.shape[0]
        scale = (float(alpha) / rank if alpha else 1.0) * strength

        all_A.append(down * scale)
        all_B.append(up)
        dora_scales.append(dora_scale)

    if not all_A:
        return None, None, None

    # DoRA consistency check: all or none must have dora_scale.
    has_dora = [ds is not None for ds in dora_scales]
    if any(has_dora) and not all(has_dora):
        return None, None, None  # mixed — bail

    # For multi-LoRA with DoRA, only single-LoRA is supported for now
    # (multi-LoRA DoRA would require sequential norm application).
    if all(has_dora) and len(dora_scales) > 1:
        return None, None, None

    final_dora = dora_scales[0] if all(has_dora) else None
    return all_A, all_B, final_dora


# ---------------------------------------------------------------------------
# Full-delta extraction for LoHa / LoKr adapters
# ---------------------------------------------------------------------------

def _compute_loha_delta(v, strength):
    """Compute LoHa delta: ``strength * alpha * (w1a @ w1b) ⊙ (w2a @ w2b)``.

    Returns ``(delta [out, in], dora_scale)`` or ``(None, None)`` on bail.
    """
    w1a = v[0]
    w1b = v[1]
    alpha = v[2]
    w2a = v[3]
    w2b = v[4]
    t1 = v[5] if len(v) > 5 else None
    t2 = v[6] if len(v) > 6 else None
    dora_scale = v[7] if len(v) > 7 else None

    # CP decomposition → 4D conv output → bail
    if t1 is not None or t2 is not None:
        return None, None

    m1 = torch.mm(w1a.float(), w1b.float())
    m2 = torch.mm(w2a.float(), w2b.float())
    delta = m1 * m2  # Hadamard product → [out, in]

    scale_val = (float(alpha) / w1b.shape[0] if alpha is not None else 1.0) * strength
    return (delta * scale_val), dora_scale


def _compute_lokr_delta(v, strength):
    """Compute LoKr delta: ``strength * alpha * kron(w1, w2)``.

    Returns ``(delta [out, in], dora_scale)`` or ``(None, None)`` on bail.
    """
    w1 = v[0]
    w2 = v[1]
    alpha = v[2]
    w1a = v[3] if len(v) > 3 else None
    w1b = v[4] if len(v) > 4 else None
    w2a = v[5] if len(v) > 5 else None
    w2b = v[6] if len(v) > 6 else None
    t2 = v[7] if len(v) > 7 else None
    dora_scale = v[8] if len(v) > 8 else None

    dim = None

    if w1 is None:
        if w1a is None or w1b is None:
            return None, None
        dim = w1b.shape[0]
        w1 = torch.mm(w1a.float(), w1b.float())
    else:
        w1 = w1.float()

    if w2 is None:
        if w2a is None or w2b is None:
            return None, None
        dim = w2b.shape[0]
        if t2 is not None:
            # einsum with t2 produces 4D → conv → bail
            return None, None
        w2 = torch.mm(w2a.float(), w2b.float())
    else:
        w2 = w2.float()

    # 4D w2 → conv → bail
    if w2.ndim == 4:
        return None, None

    if w2.ndim == 4 or w1.ndim == 4:
        return None, None

    delta = torch.kron(w1, w2)

    if alpha is not None and dim is not None:
        scale_val = (float(alpha) / dim) * strength
    else:
        scale_val = 1.0 * strength

    return (delta * scale_val), dora_scale


def _extract_full_delta(patches):
    """Extract a full-rank weight delta for LoHa / LoKr adapters.

    Unlike ``_extract_lora_ab`` which returns low-rank (A, B) matrices,
    this computes the full ``[out, in]`` delta at decompose time.  The
    delta is applied in activation-space as ``F.linear(x, delta)``.

    Returns ``(delta, dora_scale)`` or ``(None, None)`` on bail.
    Scale (alpha * strength) is baked into delta.
    """
    deltas = []
    dora_scales = []

    for p in patches:
        strength = float(p[0])
        adapter = p[1]
        offset = p[3]
        function = p[4]

        if offset is not None or function is not None:
            return None, None

        name = getattr(adapter, "name", None)
        if not hasattr(adapter, "weights"):
            return None, None
        v = adapter.weights

        if name == "loha":
            delta, ds = _compute_loha_delta(v, strength)
        elif name == "lokr":
            delta, ds = _compute_lokr_delta(v, strength)
        else:
            return None, None

        if delta is None:
            return None, None

        deltas.append(delta)
        dora_scales.append(ds)

    if not deltas:
        return None, None

    # DoRA consistency: all or none
    has_dora = [ds is not None for ds in dora_scales]
    if any(has_dora) and not all(has_dora):
        return None, None
    if all(has_dora) and len(dora_scales) > 1:
        return None, None  # multi-DoRA not supported

    # Sum all deltas (multi-adapter stacking)
    total_delta = deltas[0]
    for d in deltas[1:]:
        total_delta = total_delta + d

    final_dora = dora_scales[0] if all(has_dora) else None
    return total_delta, final_dora


def _stack_ab(all_A, all_B):
    """Stack multi-LoRA A/B lists into single tensors."""
    if len(all_A) == 1:
        return all_A[0], all_B[0]
    return torch.cat(all_A, dim=0), torch.cat(all_B, dim=1)


def _decompose_lora_for_tp(module, patches):
    """Decompose LoRA patches into activation-space on a TPLinear module.

    Tries low-rank (A, B) extraction first (LoRA / DoRA), then falls back
    to full-delta extraction (LoHa / LoKr).

    Returns True on success, False if any patch cannot be decomposed.
    """
    from raylight.distributed_modules.lora_triton import attach_lora, attach_delta

    # --- Try low-rank path (LoRA / DoRA) ---
    all_A, all_B, dora_scale = _extract_lora_ab(patches)
    if all_A is not None:
        tp_size = module._tp_size_runtime
        if tp_size > 1:
            tp_rank = module._tp_rank
            for i in range(len(all_A)):
                if module.parallelism == "column":
                    local_out = module.local_out_features
                    if all_B[i].shape[0] != local_out:
                        all_B[i] = all_B[i].narrow(0, tp_rank * local_out, local_out)
                else:  # row
                    local_in = module.local_in_features
                    if all_A[i].shape[1] != local_in:
                        all_A[i] = all_A[i].narrow(1, tp_rank * local_in, local_in)

        if dora_scale is not None and tp_size > 1 and module.parallelism == "column":
            local_out = module.local_out_features
            if dora_scale.shape[0] != local_out:
                dora_scale = dora_scale.narrow(0, module._tp_rank * local_out, local_out)

        lora_A, lora_B = _stack_ab(all_A, all_B)
        attach_lora(module, lora_A, lora_B, scale=1.0, dora_scale=dora_scale)
        return True

    # --- Try full-delta path (LoHa / LoKr) ---
    delta, dora_scale = _extract_full_delta(patches)
    if delta is not None:
        # TP-shard the delta matrix
        tp_size = module._tp_size_runtime
        if tp_size > 1:
            tp_rank = module._tp_rank
            if module.parallelism == "column":
                local_out = module.local_out_features
                if delta.shape[0] != local_out:
                    delta = delta.narrow(0, tp_rank * local_out, local_out)
            else:  # row
                local_in = module.local_in_features
                if delta.shape[1] != local_in:
                    delta = delta.narrow(1, tp_rank * local_in, local_in)

        if dora_scale is not None and tp_size > 1 and module.parallelism == "column":
            local_out = module.local_out_features
            if dora_scale.shape[0] != local_out:
                dora_scale = dora_scale.narrow(0, module._tp_rank * local_out, local_out)

        attach_delta(module, delta, dora_scale=dora_scale)
        return True

    return False


# ---------------------------------------------------------------------------
# Demand-fault helper for layer-aware eviction (Phase 1)
# ---------------------------------------------------------------------------

def _demand_fault_params(cache, diff_model, names: set, sync: bool = True) -> int:
    """Re-inflate specific evicted params from pinned cache before block forward.

    Equivalent to aimdo's ``vbar_fault()`` — make specific pages resident
    in VRAM.  Only reloads the intersection of *names* and
    ``cache._partial_freed``; everything else is left evicted.

    When *sync* is False the caller is responsible for stream
    synchronisation (used by the prefetch pipeline on a side stream).

    Returns count of params restored.
    """
    import torch

    param_dict = dict(diff_model.named_parameters())
    restored = 0

    for name in names:
        entry = cache._partial_freed.get(name)
        if entry is None:
            continue
        storage, nbytes = entry
        storage.resize_(nbytes)

        param = param_dict.get(name)
        if param is None:
            continue

        # Copy from pinned cache (fastest — DMA-capable)
        cpu_buf = cache._get_cpu_buf(name)
        if cpu_buf is not None:
            param.data.copy_(cpu_buf, non_blocking=True)
            restored += 1
        elif cache._mmap_sd is not None:
            # Fallback: copy from mmap source
            mmap_key = cache._resolve_mmap_key(name)
            if mmap_key is not None:
                param.data.copy_(cache._mmap_sd[mmap_key], non_blocking=True)
                restored += 1

        del cache._partial_freed[name]

    if restored > 0:
        if sync:
            torch.cuda.synchronize()
        cache._watermark = len(cache._partial_freed)

    return restored


# ---------------------------------------------------------------------------
# Activation-space LoRA for nn.Linear (LazyTensor / GGUF SP paths)
# ---------------------------------------------------------------------------

_LORA_HOOK_KEY = "_raylight_lora_hook_handle"


def _lora_post_hook(module, input, output):
    """Forward hook that applies the LoRA delta (with optional DoRA norm).

    Supports both low-rank (lora_A/lora_B) and full-delta (lora_delta) paths.
    """
    # --- Low-rank path (LoRA / DoRA) ---
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)
    if lora_A is not None and lora_B is not None:
        from raylight.distributed_modules.lora_triton import lora_forward, apply_dora
        scale = getattr(module, "lora_scale", 1.0) or 1.0
        dora_scale = getattr(module, "lora_dora_scale", None)
        delta = lora_forward(input[0], lora_A, lora_B, float(scale))
        if dora_scale is not None:
            return apply_dora(
                output, delta, module.weight, dora_scale,
                lora_a=lora_A, lora_b=lora_B,
            )
        return output + delta.to(output.dtype)

    # --- Full-delta path (LoHa / LoKr) ---
    lora_delta = getattr(module, "lora_delta", None)
    if lora_delta is not None:
        import torch.nn.functional as F
        from raylight.distributed_modules.lora_triton import apply_dora_with_delta
        x = input[0]
        d = lora_delta.to(device=x.device, dtype=x.dtype, non_blocking=True)
        delta = F.linear(x, d)
        dora_scale = getattr(module, "lora_dora_scale", None)
        if dora_scale is not None:
            return apply_dora_with_delta(
                output, delta, module.weight, dora_scale, lora_delta=d,
            )
        return output + delta.to(output.dtype)

    return output


def _ensure_lora_hook(module):
    """Register the LoRA forward hook if not already present."""
    if getattr(module, _LORA_HOOK_KEY, None) is None:
        handle = module.register_forward_hook(_lora_post_hook)
        setattr(module, _LORA_HOOK_KEY, handle)


def _remove_lora_hook(module):
    """Remove the LoRA forward hook and clear attributes."""
    handle = getattr(module, _LORA_HOOK_KEY, None)
    if handle is not None:
        handle.remove()
        setattr(module, _LORA_HOOK_KEY, None)


def _decompose_lora_for_linear(module, patches):
    """Decompose LoRA patches into activation-space on an nn.Linear.

    Tries low-rank (A, B) first (LoRA / DoRA), then full-delta (LoHa / LoKr).
    Installs a forward hook so the delta is applied additively during
    forward — the base weight is never backed up or mutated.

    Returns True on success, False if any patch cannot be decomposed.
    """
    from raylight.distributed_modules.lora_triton import attach_lora, attach_delta

    # --- Try low-rank path ---
    all_A, all_B, dora_scale = _extract_lora_ab(patches)
    if all_A is not None:
        lora_A, lora_B = _stack_ab(all_A, all_B)
        attach_lora(module, lora_A, lora_B, scale=1.0, dora_scale=dora_scale)
        _ensure_lora_hook(module)
        return True

    # --- Try full-delta path ---
    delta, dora_scale = _extract_full_delta(patches)
    if delta is not None:
        attach_delta(module, delta, dora_scale=dora_scale)
        _ensure_lora_hook(module)
        return True

    return False


def clear_linear_lora(module):
    """Remove activation-space LoRA from an nn.Linear (attrs + hook)."""
    from raylight.distributed_modules.lora_triton import clear_lora
    clear_lora(module)
    _remove_lora_hook(module)


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


class RaylightModelPatcher(comfy.model_patcher.ModelPatcher):
    """Model patcher for non-FSDP flows with pinned-RAM offload support.

    Overrides ``unpatch_model`` to use the pinned param cache instead of
    ``model.to(cpu)`` when a cache is available.  This avoids crashing on
    models whose CUDA storages have been resized to 0 by the cache, and
    enables fast H2D reload via async pinned-memory copies.

    Overrides ``load`` to auto-restore from pinned cache when storages have
    been freed (safety net if Comfy's model management calls ``load`` without
    going through ``LazyTensorContext.hot_load``).
    """

    # Attributes that must survive clone() so hot-reload, offload, and
    # re-streaming work correctly regardless of how ComfyUI propagates
    # the patcher through the execution graph.
    _LIFECYCLE_ATTRS = (
        "pinned_param_cache",
        "_zero_ram",
        "current_device",
        "unet_path",
        "load_config",
        "vram_limit_bytes",
        "mmap_cache",
        "_memory_policy",
    )

    def clone(self, *args, **kwargs):
        n = super().clone(*args, **kwargs)
        n.__class__ = RaylightModelPatcher
        for attr in self._LIFECYCLE_ATTRS:
            if hasattr(self, attr):
                setattr(n, attr, getattr(self, attr))
        # Mark clones so __del__ doesn't destroy the shared cache.
        n._is_cache_owner = False
        return n

    # ------------------------------------------------------- pin guard

    def pin_weight_to_device(self, key):
        """Skip ComfyUI's per-tensor cudaHostRegister when our cache already
        covers the memory region.

        ``SharedPinnedParamCache`` uses a single ``cudaHostRegister`` over the
        entire ``/dev/shm`` segment, but ``tensor.is_pinned()`` still returns
        ``False`` (only ``cudaHostAlloc`` sets that flag).  ComfyUI's
        ``pin_memory()`` would then re-register the sub-region → CUDA error +
        ``discard_cuda_async_error()`` sync on every offloaded module.

        For ``PinnedParamCache`` the tensors are allocated with
        ``pin_memory()`` so ``is_pinned()`` is True and ComfyUI's guard
        already works — but we skip the call anyway to avoid the overhead.
        """
        pinned_cache = getattr(self, "pinned_param_cache", None)
        if pinned_cache is not None and pinned_cache.built:
            return  # already pinned / registered — nothing to do
        super().pin_weight_to_device(key)

    # --------------------------------------------------------- LoRA patching

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False):
        """Route module patches through activation-space hooks when possible.

        TPGGMLLinear: patches go through ``weight_function`` (LowVramPatch).
        TPLinear: decomposed into ``lora_A``/``lora_B`` attrs (read by forward).
        nn.Linear: decomposed into ``lora_A``/``lora_B`` attrs + forward hook.

        For all three cases the base weight is **never** backed up or mutated,
        eliminating the backup-dict RAM overhead and the restore-on-reset copy.
        Falls through to ComfyUI's standard backup+mutate for unsupported
        patch types (LoCon, DoRA, offset, custom function).
        """
        from raylight.distributed_modules.tp_linear_factory import TPGGMLLinear
        from raylight.distributed_modules.tensor_parallel import TPLinear

        parts = key.rsplit('.', 1)
        if len(parts) == 2 and parts[1] == "weight" and key in self.patches:
            try:
                op = comfy.utils.get_attr(self.model, parts[0])
            except (AttributeError, KeyError):
                op = None

            if isinstance(op, TPGGMLLinear):
                attr = "weight_function"
                if not hasattr(op, attr):
                    setattr(op, attr, [])
                setattr(op, attr, [LowVramPatch(key, self.patches)])
                return None

            if isinstance(op, TPLinear):
                if _decompose_lora_for_tp(op, self.patches[key]):
                    return None

            if isinstance(op, nn.Linear) and not isinstance(op, (TPLinear, TPGGMLLinear)):
                if _decompose_lora_for_linear(op, self.patches[key]):
                    return None

        # Handle TPGGMLLinear bias patches
        if len(parts) == 2 and parts[1] == "bias" and key in self.patches:
            try:
                op = comfy.utils.get_attr(self.model, parts[0])
            except (AttributeError, KeyError):
                op = None
            if isinstance(op, TPGGMLLinear):
                attr = "bias_function"
                if not hasattr(op, attr):
                    setattr(op, attr, [])
                setattr(op, attr, [LowVramPatch(key, self.patches)])
                return None

        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight)

    # ------------------------------------------------------------------ load

    def load(self, device_to=None, **kwargs):
        """Auto-restore from pinned cache if storages were freed, then delegate.

        Budget enforcement
        ~~~~~~~~~~~~~~~~~~
        ComfyUI's ``load_models_gpu`` → ``partially_load`` chain computes
        its own VRAM budget from ``get_free_memory()`` and passes it as
        ``lowvram_model_memory``.  On a GPU with lots of free VRAM this
        value can exceed what the user requested via ``vram_limit_gb``.
        We intercept here and cap the budget so the user's limit is
        always honoured.

        Auto-restore
        ~~~~~~~~~~~~
        When no lowvram budget is active, the pinned cache is used to
        bulk-copy all weights back to CUDA before ``super().load()``
        re-applies LoRA patches / hooks.
        """
        # ── Enforce user VRAM budget ──────────────────────────────────
        vram_limit = getattr(self, "vram_limit_bytes", 0)
        if vram_limit > 0 and device_to is not None:
            try:
                from raylight.distributed_actor.model_context import _compute_vram_budget
                model_bytes = self.model_size()
                budget = _compute_vram_budget(
                    device_to, model_bytes, vram_limit_bytes=vram_limit,
                )
                if budget > 0:
                    # Partial load required — cap whatever ComfyUI passed.
                    incoming = kwargs.get("lowvram_model_memory", 0)
                    if incoming == 0 or incoming > budget:
                        kwargs["lowvram_model_memory"] = budget
                    kwargs["full_load"] = False
                    logging.info(
                        "[RaylightModelPatcher] VRAM limit %.1f GB → "
                        "weight budget capped to %.0f MB (was %.0f MB)",
                        vram_limit / 1e9,
                        budget / (1024 ** 2),
                        incoming / (1024 ** 2),
                    )
            except Exception as e:
                logging.warning(
                    "[RaylightModelPatcher] Budget cap failed: %s", e,
                )

        # ── Auto-restore from pinned cache ────────────────────────────
        # After offload_to_cpu(), CUDA storages are resized to 0 and
        # param.data references the empty storage.  We must restore the
        # data BEFORE super().load() tries to move modules around.
        #
        # Two modes:
        #   • Full load (no budget): bulk-copy pinned → CUDA so
        #     super().load() only re-applies LoRA patches / hooks.
        #   • Partial load (budget > 0): restore to pinned CPU tensors
        #     so super().load() can DMA only the budget-worth of modules
        #     to CUDA and leave the rest in pinned RAM with lowvram hooks.
        pinned_cache = getattr(self, "pinned_param_cache", None)
        skip_restore = getattr(self, "_skip_pinned_auto_restore", False)
        if pinned_cache is not None and pinned_cache.built and not pinned_cache.params_on_cuda and not skip_restore:
            effective_budget = kwargs.get("lowvram_model_memory", 0)
            need_partial = 0 < effective_budget < self.model_size()
            if need_partial:
                # Partial load: restore to pinned CPU — super().load()
                # will DMA the budget-worth to CUDA (fast from pinned RAM)
                # and install lowvram hooks for the overflow modules.
                logging.info("[RaylightModelPatcher] Auto-restoring to pinned CPU for partial load...")
                pinned_cache.reload_to_cpu(self.model)
            else:
                # Full load: bulk-copy to CUDA
                logging.info("[RaylightModelPatcher] Auto-reloading from pinned cache (full → CUDA)...")
                pinned_cache.reload_to_cuda(self.model)

        result = super().load(device_to=device_to, **kwargs)

        # ── Update pinned cache CUDA flag ─────────────────────────────
        # After super().load() finishes, some or all params are on CUDA.
        # Keep the cache state consistent so the next offload works.
        if pinned_cache is not None and pinned_cache.built:
            if device_to is not None and str(device_to) != "cpu":
                pinned_cache._on_cuda = True

        # ── Phase 1: Install per-block eviction hooks ─────────────────
        # These protect the currently-executing block from the CUDA
        # allocator interceptor while allowing all other blocks' weights
        # to be evicted under VRAM pressure.
        self._install_eviction_hooks()

        return result

    # ──────────────────────────────────────────────────────────────────
    # Layer-aware eviction hooks (Phase 1)
    # ──────────────────────────────────────────────────────────────────

    _EVICTION_PRE_KEY = "_raylight_eviction_pre_hook"
    _EVICTION_POST_KEY = "_raylight_eviction_post_hook"

    def _install_eviction_hooks(self) -> None:
        """Register pre/post-forward hooks on transformer blocks.

        Each hook protects the block's parameter names during its forward
        pass and demand-faults any evicted weights before compute.
        Equivalent to aimdo's per-layer ``vbar_fault()`` / ``vbar_unpin()``.

        **Phase 2 — Prefetch pipeline**:  The pre-hook of block *N* also
        kicks off an *async* H2D copy of block *N+1*'s evicted weights on
        a dedicated CUDA stream.  When block *N+1*'s pre-hook fires it
        waits on the prefetch event instead of doing a synchronous
        demand-fault, hiding H2D latency behind compute.
        """
        policy = getattr(self, "_memory_policy", None)
        pinned_cache = getattr(self, "pinned_param_cache", None)
        if policy is None or pinned_cache is None:
            return
        can_evict = pinned_cache.built or getattr(pinned_cache, 'has_mmap_fallback', False)
        if not can_evict:
            return

        diff_model = getattr(self, "model", None)
        if diff_model is None:
            return

        from raylight.nodes import _TRANSFORMER_BLOCK_NAMES

        # Remove stale hooks first (idempotent re-install on re-load).
        self._remove_eviction_hooks()

        # ── Collect all transformer blocks in execution order ─────────
        ordered_blocks = []
        for block_attr in _TRANSFORMER_BLOCK_NAMES:
            blocks = getattr(diff_model, block_attr, None)
            if blocks is None:
                continue
            for i, block in enumerate(blocks):
                prefix = f"diffusion_model.{block_attr}.{i}."
                param_names = frozenset(
                    name for name, _ in diff_model.named_parameters()
                    if name.startswith(prefix)
                )
                if param_names:
                    ordered_blocks.append((block, param_names))

        if not ordered_blocks:
            return

        # ── Prefetch state — shared mutable containers for closures ───
        pf_stream = torch.cuda.Stream()
        pf_event = [None]       # pending prefetch CUDA event
        pf_names = [None]       # frozenset of prefetched block's param names

        installed = 0
        for idx, (block, param_names) in enumerate(ordered_blocks):
            next_names = (
                ordered_blocks[idx + 1][1] if idx + 1 < len(ordered_blocks) else None
            )

            def _make_pre_hook(names, nxt_names, cache, pol, diff_mod, stream, evt, pfn, block_idx):
                def hook(module, args):
                    # 0. Update execution cursor for distance-aware eviction.
                    cache.set_execution_cursor(block_idx)

                    # 1. Clean up stale prefetch protection (mis-predicted order).
                    if pfn[0] is not None and pfn[0] is not names:
                        pol.unprotect_params(pfn[0])
                        evt[0] = None
                        pfn[0] = None

                    # 2. Protect this block from eviction.
                    pol.protect_params(names)

                    # 3. Load this block's weights.
                    if evt[0] is not None and pfn[0] is names:
                        # Was prefetched — wait for async H2D to complete.
                        torch.cuda.current_stream().wait_event(evt[0])
                        evt[0] = None
                        pfn[0] = None
                    elif cache._partial_freed:
                        # Not prefetched — demand-fault synchronously.
                        evicted = names & cache._partial_freed.keys()
                        if evicted:
                            _demand_fault_params(cache, diff_mod, evicted)

                    # 4. Prefetch next block on a side stream.
                    if nxt_names is not None and cache._partial_freed:
                        evicted_next = nxt_names & cache._partial_freed.keys()
                        if evicted_next:
                            pol.protect_params(nxt_names)  # guard during async copy
                            with torch.cuda.stream(stream):
                                _demand_fault_params(
                                    cache, diff_mod, evicted_next, sync=False,
                                )
                            evt[0] = stream.record_event()
                            pfn[0] = nxt_names
                return hook

            def _make_post_hook(names, pol):
                def hook(module, args, output):
                    pol.unprotect_params(names)
                return hook

            pre_h = block.register_forward_pre_hook(
                _make_pre_hook(
                    param_names, next_names, pinned_cache, policy,
                    diff_model, pf_stream, pf_event, pf_names, idx,
                )
            )
            post_h = block.register_forward_hook(
                _make_post_hook(param_names, policy)
            )
            setattr(block, self._EVICTION_PRE_KEY, pre_h)
            setattr(block, self._EVICTION_POST_KEY, post_h)
            installed += 1

        # Store block structure for distance-aware eviction.
        pinned_cache.set_block_structure([names for _, names in ordered_blocks])

        # Store prefetch state for cleanup.
        self._pf_names = pf_names
        self._pf_stream = pf_stream

        if installed > 0:
            logging.info(
                "[RaylightModelPatcher] Installed eviction hooks with "
                "prefetch pipeline on %d blocks.",
                installed,
            )

    def _remove_eviction_hooks(self) -> None:
        """Remove all eviction hooks and clean up prefetch state."""
        # Clean up stale prefetch protection.
        pf_names = getattr(self, "_pf_names", None)
        if pf_names is not None and pf_names[0] is not None:
            policy = getattr(self, "_memory_policy", None)
            if policy is not None:
                policy.unprotect_params(pf_names[0])
            pf_names[0] = None
        self._pf_names = None
        self._pf_stream = None

        # Reset execution cursor and block structure.
        pinned_cache = getattr(self, "pinned_param_cache", None)
        if pinned_cache is not None:
            pinned_cache.set_execution_cursor(-1)
            pinned_cache._block_param_groups = None

        diff_model = getattr(self, "model", None)
        if diff_model is None:
            return

        from raylight.nodes import _TRANSFORMER_BLOCK_NAMES

        for block_attr in _TRANSFORMER_BLOCK_NAMES:
            blocks = getattr(diff_model, block_attr, None)
            if blocks is None:
                continue
            for block in blocks:
                for key in (self._EVICTION_PRE_KEY, self._EVICTION_POST_KEY):
                    handle = getattr(block, key, None)
                    if handle is not None:
                        handle.remove()
                        setattr(block, key, None)

    # ------------------------------------------------------------- unpatch

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if self.model is None:
            return

        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            self.unpin_all_weights()
            self._remove_eviction_hooks()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)
                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0
            else:
                # TPGGMLLinear modules always use weight_function (set by
                # our patch_weight_to_device override) even when the model
                # is NOT in lowvram mode.  Clean them up here so stale
                # LoRA patches don't persist after unpatch.
                from raylight.distributed_modules.tp_linear_factory import TPGGMLLinear
                from raylight.distributed_modules.tensor_parallel import TPLinear
                from raylight.distributed_modules.lora_triton import clear_lora
                for m in self.model.modules():
                    if isinstance(m, TPGGMLLinear):
                        wipe_lowvram_weight(m)
                    elif isinstance(m, TPLinear):
                        clear_lora(m)
                    elif isinstance(m, nn.Linear):
                        clear_linear_lora(m)

            keys = list(self.backup.keys())
            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    comfy.utils.copy_to_param(self.model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                pinned_cache = getattr(self, "pinned_param_cache", None)
                if pinned_cache is not None and str(device_to) == "cpu":
                    if pinned_cache.params_on_cuda:
                        # Use pinned cache to offload — do NOT call model.to(cpu)
                        # which would crash on resized-to-0 storages or cause a
                        # slow full-model CPU copy.
                        # offload_to_cpu handles _built=False by building first.
                        try:
                            pinned_cache.offload_to_cpu(self.model)
                        except Exception as e:
                            logging.warning(f"[RaylightModelPatcher] Pinned offload failed: {e}")
                            # Fallback to normal move
                            self.model.to(device_to)
                    # else: already offloaded — CUDA storages are freed (size 0).
                    # Do NOT call model.to(cpu) which would crash on zero-sized
                    # CUDA storages left behind by offload_cuda_only / offload_to_cpu.
                    self.model.device = device_to
                else:
                    self.model.to(device_to)
                    self.model.device = device_to

            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])
        self.object_patches_backup.clear()

    def __del__(self):
        # Only the original model (cache owner) may clean up the pinned
        # cache.  Clones share the same cache object — if a clone's
        # __del__ ran cleanup(), it would destroy the cache while the
        # original model still relies on it for hot-reload.
        if not getattr(self, "_is_cache_owner", True):
            return
        cache = getattr(self, "pinned_param_cache", None)
        if cache is not None:
            try:
                cache.cleanup()
            except Exception:
                pass


class FSDPModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        rank: int = 0,
        fsdp_state_dict: dict | None = None,
        device_mesh=None,
        is_cpu_offload: bool = False,
    ):
        super().__init__(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update,
        )
        self.rank = rank
        self.fsdp_state_dict = fsdp_state_dict
        self.device_mesh = device_mesh
        self.is_cpu_offload = is_cpu_offload
        self.patch_fsdp = patch_fsdp.__get__(self, FSDPModelPatcher)

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        if self.model is not None:
             self.model.to("meta")

    def set_fsdp_state_dict(self, sd):
        self.fsdp_state_dict = sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, convert_dtensor=False):
        inplace_update = True
        if key not in self.patches:
            return
        weight, set_func, convert_func = get_key_weight(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy_dist.lora.calculate_weight(self.patches[key], temp_weight, key, device_mesh=self.device_mesh)
        if set_func is None:
            out_weight = comfy_dist.float.stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key), device_mesh=self.device_mesh)

            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

        else:
            set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

    def clone(self, *args, **kwargs):
        # Call parent clone normally (keeps init signature correct)
        n = super(FSDPModelPatcher, self).clone(*args, **kwargs)

        n.__class__ = FSDPModelPatcher
        n.rank = self.rank
        n.fsdp_state_dict = self.fsdp_state_dict
        n.device_mesh = self.device_mesh
        n.is_cpu_offload = self.is_cpu_offload
        # Share the shard cache reference — both patcher instances back the same model
        n.fsdp_shard_cache = getattr(self, "fsdp_shard_cache", None)
        n._is_cache_owner = False

        return n

    def _load_list(self):
        loading = []
        if self.model is None:
             return loading
        
        model = cast(nn.Module, self.model)
        for n, m in model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True  # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                mem = comfy.model_management.module_size(m)
                loading.append((mem, mem, n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        if self.model is None:
            return

        with self.use_ejected():
            # If we are baking weights (force_patch_weights=True), we MUST NOT wrap FSDP yet.
            # We want to patch the original model first, THEN wrap it later.
            if not isinstance(self.model.diffusion_model, FSDPModule) and not force_patch_weights:
                self.patch_fsdp()
            else:
                # FSDP already wrapped — model was soft-offloaded to pinned CPU RAM.
                # Restore shards via the cache (fast H2D), or fall back to align_model_to_cuda
                # if no cache is present (e.g. cold start before first offload cycle).
                # Skip entirely when CPUOffload is active — FSDP manages shard placement.
                if not self.is_cpu_offload:
                    shard_cache = getattr(self, "fsdp_shard_cache", None)
                    if shard_cache is not None and shard_cache.built and not shard_cache.shards_on_cuda:
                        try:
                            shard_cache.reload_to_cuda(self.model.diffusion_model)
                        except Exception as e:
                            logging.warning(f"[FSDPModelPatcher] Shard cache reload failed: {e}")
                    elif shard_cache is None or not shard_cache.built:
                        # No cache yet — fall back to the legacy align helper
                        from raylight.distributed_modules.utils import align_model_to_cuda
                        try:
                            align_model_to_cuda(self.model)
                        except Exception as e:
                            logging.warning(f"[FSDPModelPatcher] Could not realign FSDP shards to CUDA: {e}")
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[2]
                m = x[3]
                params = x[4]
                module_mem = x[0]

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                            continue

                # This single line, take my entire week, TOUCH THIS will cause 0 tensor on model output
                cast_weight = self.force_cast_weights
                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []

                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = [LowVramPatch(weight_key, self.patches)]
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = [LowVramPatch(bias_key, self.patches)]
                        patch_counter += 1

                cast_weight = True

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights is True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                if self.model is not None:
                     self.model.model_lowvram = True
            else:
                logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                if self.model is not None:
                    self.model.model_lowvram = False
                    if full_load:
                        self.model.to(device_to)
                        mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def cleanup(self):
        self.clean_hooks()
        if self.model is None:
             return
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None
        # Release FSDP shard cache when the patcher is being cleaned up
        shard_cache = getattr(self, "fsdp_shard_cache", None)
        if shard_cache is not None:
            try:
                shard_cache.cleanup()
            except Exception as e:
                logging.warning(f"[FSDPModelPatcher] Shard cache cleanup error: {e}")
        for callback in self.get_all_callbacks(CallbacksMP.ON_CLEANUP):
            callback(self)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if self.model is None:
             return
        
        model = cast(nn.Module, self.model)
        
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            if model.model_lowvram:
                for m in model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                model.model_lowvram = False
                model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                weight, _, _ = get_key_weight(model, k)
                
                # Handling DTensor (FSDP) unpatching
                if isinstance(weight, DTensor):
                    # We cannot simple copy_() a full tensor into a DTensor.
                    # Ideally we should re-shard the backup weight, but for now we skip 
                    # to prevent crashing. Creating a proper distributed copy is complex here.
                    continue

                if bk.inplace_update:
                    comfy.utils.copy_to_param(model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(model, k, bk.weight)

            model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                if next(model.parameters()).device == torch.device("meta"):
                    pass
                elif isinstance(model.diffusion_model, FSDPModule):
                    # CRITICAL: Do NOT call model.to(cpu) on an FSDP-wrapped model.
                    # That would blindly move all DTensor local shards to CPU, causing
                    # FSDP to all-gather from CPU on the next forward (only ~2.7 GB resident).
                    #
                    # Explicit offload (VAE needs VRAM): use the shard cache to move shards
                    # to pinned CPU RAM efficiently, then free CUDA storage.
                    # Implicit post-inference cleanup: shards stay on CUDA — no action needed.
                    shard_cache = getattr(self, "fsdp_shard_cache", None)
                    if (
                        shard_cache is not None
                        and shard_cache.built
                        and shard_cache.shards_on_cuda
                        and device_to is not None
                        and str(device_to) == "cpu"
                    ):
                        try:
                            shard_cache.offload_to_cpu(model.diffusion_model)
                        except Exception as e:
                            logging.warning(f"[FSDPModelPatcher] Shard cache offload failed: {e}")
                else:
                    model.to(device_to)
                    model.device = device_to
            model.model_loaded_weight_memory = 0

            for m in model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def release_memory(self):
        """Explicitly free distributed tensor storage."""
        if self.model is None:
             return
        
        model = cast(nn.Module, self.model)
             
        self.detach(unpatch_all=False)
        for m in model.modules():
            for p in m.parameters(recurse=False):
                if isinstance(p, DTensor):
                    try:
                        local = p.to_local()
                        _free_storage(local.data)
                    except Exception:
                        pass
                elif isinstance(p, torch.Tensor):
                    try:
                        _free_storage(p.data)
                    except Exception:
                        pass
        
        # Release FSDP shard cache
        shard_cache = getattr(self, "fsdp_shard_cache", None)
        if shard_cache is not None:
            try:
                shard_cache.cleanup()
            except Exception:
                pass

    def __del__(self):
        if not getattr(self, "_is_cache_owner", True):
            return
        try:
            self.release_memory()
        except:
            pass
            
        # Release shard cache if release_memory didn't run
        shard_cache = getattr(self, "fsdp_shard_cache", None)
        if shard_cache is not None:
            try:
                shard_cache.cleanup()
            except Exception:
                pass

        try:
            del self.model
        except Exception:
            pass
        self.model = None
        comfy.model_management.soft_empty_cache()
        gc.collect()
