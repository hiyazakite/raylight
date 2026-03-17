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
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:   # intermediate_dtype has to be one that is supported in math ops
            intermediate_dtype = torch.float32
            return comfy.float.stochastic_rounding(comfy_dist.lora.calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key, intermediate_dtype=intermediate_dtype), weight.dtype, seed=string_to_seed(self.key))

        return comfy_dist.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=intermediate_dtype)


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

    def clone(self, *args, **kwargs):
        n = super().clone(*args, **kwargs)
        n.__class__ = RaylightModelPatcher
        n.pinned_param_cache = getattr(self, "pinned_param_cache", None)
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
                from raylight.distributed_worker.model_context import _compute_vram_budget
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
        # Always bulk-copy to CUDA first.  If a partial-load budget is
        # active super().load() will re-split: budget-worth stays on
        # CUDA, the rest gets lowvram hooks — but starting from live
        # CUDA tensors (fast) instead of zero-sized dead storages (crash).
        pinned_cache = getattr(self, "pinned_param_cache", None)
        skip_restore = getattr(self, "_skip_pinned_auto_restore", False)
        if pinned_cache is not None and pinned_cache.built and not pinned_cache.params_on_cuda and not skip_restore:
            logging.info("[RaylightModelPatcher] Auto-reloading from pinned cache (full → CUDA)...")
            pinned_cache.reload_to_cuda(self.model)

        result = super().load(device_to=device_to, **kwargs)

        # ── Update pinned cache CUDA flag ─────────────────────────────
        # After super().load() finishes, some or all params are on CUDA.
        # Keep the cache state consistent so the next offload works.
        if pinned_cache is not None and pinned_cache.built:
            if device_to is not None and str(device_to) != "cpu":
                pinned_cache._on_cuda = True

        return result

    # ------------------------------------------------------------- unpatch

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if self.model is None:
            return

        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            self.unpin_all_weights()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)
                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

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
                if (
                    pinned_cache is not None
                    and pinned_cache.built
                    and str(device_to) == "cpu"
                ):
                    if pinned_cache.params_on_cuda:
                        # Use pinned cache to offload — do NOT call model.to(cpu)
                        # which would crash on resized-to-0 storages or cause a
                        # slow full-model CPU copy.
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
        
        # Additional cleanup for FSDP
        if isinstance(self.model, FSDPModule):
            # Try to force FSDP internal cleanup if possible
            pass

    def __del__(self):
        try:
            self.release_memory()
        except:
            pass
            
        del self.model
        self.model = None
        comfy.model_management.soft_empty_cache()
        gc.collect()
