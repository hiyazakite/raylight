# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import collections
from typing import Any

import ray

import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import move_patch_to_device
from .dequant import is_quantized

from raylight.distributed_actor.actor_pool import ActorPool
from raylight.comfy_dist.lora import calculate_weight as ray_calculate_weight


def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")


# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])


def _is_ltxav_av_block(block) -> bool:
    """Return True if a transformer block is a dual-stream LTXAV audio-video block.

    LTXAV's BasicAVTransformerBlock is identified by its audio-specific attention
    modules. This check is used to filter blocks that cannot be compiled with GGUF.
    """
    return hasattr(block, "audio_attn1")


def _gguf_discover_compile_keys(diffusion_model) -> list[str]:
    """Build per-block compile keys for GGUF models, excluding blocks that would thrash.

    GGUF applies torch.compiler.disable on every linear layer's dequant path.
    When Dynamo traces a transformer block that contains GGUF graph breaks, it
    creates resume functions that capture the local stack state (___stack0).
    Those captured tensors are freshly allocated on every forward call, so their
    addresses always change, invalidating Dynamo's guard every step. This causes
    the model to recompile on every single denoising step, hammering the
    accumulated_recompile_limit and degrading performance severely.

    LTXAV-style dual-stream blocks (audio_attn1 present) are the specific case
    where this matters: they have many intermediate tensors, del-statements
    between GGUF graph breaks, and a complex dual-stream data flow that produces
    many resume points. Excluding them from compile keys stops the thrashing.

    Other architectures (Flux, SD, SDXL, ...) compile as normal.
    """
    from raylight.nodes import _TRANSFORMER_BLOCK_NAMES, _discover_compile_keys

    # Check whether all discovered transformer blocks are LTXAV-style.
    # If the model has no transformer blocks at all, fall back to the standard path.
    for layer_name in _TRANSFORMER_BLOCK_NAMES:
        blocks = getattr(diffusion_model, layer_name, None)
        if blocks is None:
            continue
        av_blocks = [b for b in blocks if _is_ltxav_av_block(b)]
        if av_blocks:
            non_av = [b for b in blocks if not _is_ltxav_av_block(b)]
            n_skipped = len(av_blocks)
            print(
                f"[Raylight GGUF compile] Skipping {n_skipped} LTXAV audio-video block(s) "
                f"from torch.compile: GGUF dequant graph breaks inside these blocks cause "
                f"Dynamo resume-function cache thrashing (___stack0 invalidation every step). "
                f"The remaining {len(non_av)} block(s) will be compiled normally."
            )
            if non_av:
                # Only compile the non-LTXAV-style blocks (unusual edge case)
                keys = []
                for i, b in enumerate(blocks):
                    if not _is_ltxav_av_block(b):
                        keys.append(f"diffusion_model.{layer_name}.{i}")
                return keys

            # All blocks are LTXAV-style — nothing to compile at block level.
            # Return "diffusion_model" so the outer forward is still wrapped;
            # the actual per-GGUF-layer disable annotations prevent any tracing.
            return ["diffusion_model"]

    # No LTXAV blocks found — standard path.
    return _discover_compile_keys(diffusion_model)


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    # Instance attributes:
    # mmap_cache: Dict[str, GGMLTensor]
    # unet_path: str

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        
        patches = self.patches[key]
        
        # GHOST REFRESH: Pull from mmap_cache if available (handles 'meta' restoration)
        # This allows the model to re-hydrate from a fresh mapping.
        if hasattr(self, "mmap_cache") and key in self.mmap_cache:
            weight = self.mmap_cache[key]
        else:
            weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            out_weight.patches = [(patches, key)]
        else:
            # Activation-space LoRA: for non-quantized .weight keys on
            # nn.Linear modules, decompose into lora_A/lora_B attrs +
            # forward hook — avoids backup dict + weight mutation RAM.
            parts = key.rsplit('.', 1)
            if len(parts) == 2 and parts[1] == "weight":
                try:
                    import torch.nn as _nn
                    op = comfy.utils.get_attr(self.model, parts[0])
                    if isinstance(op, _nn.Linear):
                        from raylight.comfy_dist.model_patcher import _decompose_lora_for_linear
                        if _decompose_lora_for_linear(op, patches):
                            return
                except (AttributeError, KeyError):
                    pass

            # Fallback: standard backup + mutate
            inplace_update = self.weight_inplace_update or inplace_update
            
            # CRITICAL: Backup the original (likely mmap) weight before modifying or moving
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = ray_calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            
            # 0. Clear activation-space LoRA attrs + hooks from nn.Linear
            #    modules before standard unpatch touches weights.
            import torch.nn as _nn
            from raylight.comfy_dist.model_patcher import clear_linear_lora
            for m in self.model.modules():
                if isinstance(m, _nn.Linear):
                    clear_linear_lora(m)

            # 1. Standard unpatch for non-weight state.
            super().unpatch_model(device_to=None, unpatch_weights=unpatch_weights)

            # 2. POINTER SWAP: Restore mmap references to parameters
            # This drops the GPU reference and replaces it with the CPU mmap reference
            # without triggering a copy to RAM.
            from .ops import GGMLTensor
            
            moved_to_mmap = 0
            mmap_cache = getattr(self, "mmap_cache", None)
            
            if mmap_cache:
                # Build a local lookup for parameters
                param_map = {name: param for name, param in self.model.named_parameters()}
                
                for name in mmap_cache:
                    mmap_weight = mmap_cache[name]
                    
                    # Extract raw data from GGMLTensor if needed
                    if isinstance(mmap_weight, GGMLTensor):
                        mmap_data = mmap_weight.data
                    else:
                        mmap_data = mmap_weight

                    # Advanced Fuzzy Matching
                    target_param = None
                    
                    # 1. Exact & Standard Prefix Matches
                    if name in param_map:
                        target_param = param_map[name]
                    elif f"diffusion_model.{name}" in param_map:
                        target_param = param_map[f"diffusion_model.{name}"]
                    elif f"model.diffusion_model.{name}" in param_map:
                        target_param = param_map[f"model.diffusion_model.{name}"]
                    elif name.startswith("model.diffusion_model.") and name[len("model.diffusion_model."):] in param_map:
                        target_param = param_map[name[len("model.diffusion_model."):]]
                    
                    # 2. Fuzzy Suffix Match (if still not found)
                    if target_param is None:
                        # Try matching just the suffix (e.g. '0.attn_q.weight')
                        # normalize name
                        norm_name = name.replace("model.diffusion_model.", "")
                        for p_name, p_val in param_map.items():
                             norm_p_name = p_name.replace("model.diffusion_model.", "")
                             if norm_p_name == norm_name:
                                 target_param = p_val
                                 break

                    if target_param is not None:
                        # CRITICAL: Full replacement, not just pointer swap!
                        # Using .data = ... only changes the data pointer but leaves the GPU tensor
                        # object alive. We need to replace the entire parameter with the mmap tensor.
                        # This allows GC to release the old GPU tensor.
                        
                        # Get the matched full name for this parameter
                        matched_name = None
                        for p_name, p_val in param_map.items():
                            if p_val is target_param:
                                matched_name = p_name
                                break
                        
                        if matched_name is not None:
                            # Replace with mmap tensor wrapped in Parameter
                            comfy.utils.set_attr_param(self.model, matched_name, mmap_weight)
                            moved_to_mmap += 1
                        else:
                            # Fallback to pointer swap if we can't find the name
                            target_param.data = mmap_data.data
                            if hasattr(target_param, "patches"):
                                target_param.patches = []
                            moved_to_mmap += 1
            

            
            # CRITICAL: Clear .patches on ALL parameters, not just those matched during swap.
            # GGMLTensor.to() copies .patches, so there may be GPU tensor refs on params
            # that weren't in mmap_cache or weren't matched.
            cleared_patches = 0
            for name, param in self.model.named_parameters():
                if hasattr(param, "patches") and param.patches:
                    param.patches = []
                    cleared_patches += 1

            
            if moved_to_mmap == 0 and device_to is not None and device_to.type == "cpu":
                 self.model.to(device_to)
            
            if device_to is not None:
                self.current_device = device_to
                
            from raylight.utils.common import cleanup_memory
            cleanup_memory()
            
            # CLEAR GGUF LAYER CACHE
            for layer in self.model.modules():
                if hasattr(layer, "dequant_cache"):
                    layer.dequant_cache = {}

        # Move patches themselves back to offload device if they were on GPU
        for key, patch_list in self.patches.items():
            for i, patch in enumerate(patch_list):
                if len(patch) >= 2:
                    v = patch[1] 
                    if isinstance(v, torch.Tensor) and v.device != self.offload_device:
                        if v.numel() > 0:
                            new_patch = list(patch)
                            new_patch[1] = v.to(self.offload_device)
                            patch_list[i] = tuple(new_patch)
                    elif isinstance(v, (tuple, list)) and len(v) > 0:
                        if isinstance(v[0], str) and len(v) > 1:
                            inner_v = v[1]
                            if isinstance(inner_v, tuple) and len(inner_v) > 0:
                                if isinstance(inner_v[0], torch.Tensor) and inner_v[0].device != self.offload_device:
                                    if inner_v[0].numel() > 0:
                                        new_inner = list(inner_v)
                                        new_inner[0] = inner_v[0].to(self.offload_device)
                                        new_v = (v[0], tuple(new_inner)) + tuple(v[2:]) if len(v) > 2 else (v[0], tuple(new_inner))
                                        new_patch = list(patch)
                                        new_patch[1] = new_v
                                        patch_list[i] = tuple(new_patch)
                        elif isinstance(v[0], torch.Tensor):
                             v0_tensor: Any = v[0]
                             if v0_tensor.device != self.offload_device:
                                 if v0_tensor.numel() > 0:
                                     new_v = tuple(t.to(self.offload_device) if isinstance(t, torch.Tensor) and t.device != self.offload_device else t for t in v)
                                     new_patch = list(patch)
                                     new_patch[1] = new_v
                                     patch_list[i] = tuple(new_patch)

        # Manually move non-GGUF weights (buffers etc) to target device
        if device_to is not None and unpatch_weights:
             for buf in self.model.buffers():
                 if buf.device.type == 'cuda' and device_to.type == 'cpu':
                     buf.data = buf.data.to(device_to)

    def load(self, *args, force_patch_weights=False, **kwargs):
        # GHOST RE-HYDRATION: Re-map the GGUF file ONLY if cache is missing and we have a path.
        m_cache = getattr(self, "mmap_cache", None)
        u_path = getattr(self, "unet_path", None)
        
        # Robust check for empty or missing cache
        if (m_cache is None or (isinstance(m_cache, dict) and len(m_cache) == 0)) and u_path:
            from .loader import gguf_sd_loader
            sd, _ = gguf_sd_loader(u_path)
            self.mmap_cache = sd

        # ── Enforce user VRAM budget (partial VRAM load) ──────────
        # When vram_limit_bytes is set, compute how much VRAM is
        # available and pass lowvram_model_memory so ComfyUI's load()
        # places budget-worth of modules on CUDA (quantised weights
        # stay on-device for faster dequant) and streams the rest
        # per-layer via comfy_cast_weights hooks.
        import logging
        vram_limit = getattr(self, "vram_limit_bytes", 0)
        device_to = args[0] if args else kwargs.get("device_to")
        if vram_limit > 0 and device_to is not None:
            try:
                from raylight.distributed_actor.model_context import _compute_vram_budget
                model_bytes = self.model_size()
                budget = _compute_vram_budget(
                    device_to, model_bytes, vram_limit_bytes=vram_limit,
                )
                if budget > 0:
                    incoming = kwargs.get("lowvram_model_memory", 0)
                    if incoming == 0 or incoming > budget:
                        kwargs["lowvram_model_memory"] = budget
                    kwargs["full_load"] = False
                    logging.info(
                        "[GGUFModelPatcher] VRAM limit %.1f GB → "
                        "weight budget capped to %.0f MB (model %.0f MB)",
                        vram_limit / 1e9,
                        budget / (1024 ** 2),
                        model_bytes / (1024 ** 2),
                    )
            except Exception as e:
                logging.warning(
                    "[GGUFModelPatcher] Budget cap failed: %s", e,
                )

        super().load(*args, force_patch_weights=True, **kwargs)

        # Optimization: Clear backup after load to drop extra references?
        # Standard ComfyUI keeps backup to restore later. We need it for unpatching.
        # But if we accumulate backups across reloads (in clones), it leaks.
        # We'll handle leaks in clone().

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_cache = getattr(self, "mmap_cache", {})
        n.vram_limit_bytes = getattr(self, "vram_limit_bytes", 0)
        
        # CRITICAL FIX: Empty backup in clone to prevent mmap reference leaks
        # Clones will repopulate backup when they act.
        n.backup = {}
        
        # FIX: Create a shallow copy of patches dict to prevent shared mutation.
        # The super().clone() already copies patches, but we ensure isolation here.
        if hasattr(self, "patches") and self.patches:
            n.patches = {k: list(v) for k, v in self.patches.items()}
        
        return n


class RayGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet_gguf"),),
                "dequant_dtype": (
                    ["default", "target", "float32", "float16", "bfloat16"],
                    {"default": "default"},
                ),
                "patch_dtype": (
                    ["default", "target", "float32", "float16", "bfloat16"],
                    {"default": "default"},
                ),
                "actors_init": (
                    "RAY_ACTORS_INIT",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "cache_patched_weights": ("BOOLEAN", {"default": False}),
                "torch_compile": (
                    ["disabled", "inductor", "inductor_reduce_overhead"],
                    {"default": "disabled", "tooltip": "Apply torch.compile to transformer blocks for faster inference. 'inductor' uses Triton codegen; 'inductor_reduce_overhead' adds CUDA-graph wrapping around compiled kernels. First run will be slow due to compilation warmup."},
                ),
                "torch_compile_dynamic": (
                    ["auto", "true", "false"],
                    {"default": "auto", "tooltip": "Dynamic shape tracing. 'auto': starts static, switches to dynamic on shape change. 'true': always symbolic shapes (avoids recompiles but slightly slower kernels). 'false': always static (fastest kernels, recompiles on any shape change)."},
                ),
                "fused_cuda_kernel": ("BOOLEAN", {"default": True, "tooltip": "Use fused CUDA kernels for GGUF dequant+GEMM. Disable to fall back to the PyTorch dequant path for comparison."}),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    def load_ray_unet(
        self,
        actors_init,
        unet_name,
        dequant_dtype,
        patch_dtype,
        cache_patched_weights=False,
        torch_compile="disabled",
        torch_compile_dynamic="auto",
        fused_cuda_kernel=True,
    ):
        actors, actors, config = ActorPool.ensure_fresh(actors_init)
        
        model_options = {
            "dequant_dtype": dequant_dtype,
            "patch_dtype": patch_dtype,
            "cache_patched_weights": cache_patched_weights,
            "fused_cuda_kernel": fused_cuda_kernel,
        }
        
        unet_path = folder_paths.get_full_path_or_raise("unet", unet_name)

        loaded_futures = []
        patched_futures = []

        if config.strategy.fsdp_enabled:
            worker0 = ray.get_actor("RayActor:0")
            ray.get(worker0.load_unet.remote(unet_path, model_options=model_options))
            meta_model = ray.get(worker0.get_meta_model.remote())

            for actor in actors:
                if actor != worker0:
                    loaded_futures.append(actor.set_meta_model.remote(meta_model))

            ray.get(loaded_futures)
            loaded_futures = []

            for actor in actors:
                loaded_futures.append(actor.set_state_dict.remote())

            ray.get(loaded_futures)
            loaded_futures = []
        else:
            # Mmap-aware loading strategy:
            # - use_mmap=True: Pure parallel loading (OS page cache handles sharing)
            # - use_mmap=False: Leader-follower sequential (avoid RAM spikes from concurrent copies)
            use_mmap = config.device.use_mmap
            
            if use_mmap:
                # PARALLEL: All actors load simultaneously via mmap
                load_futures = []
                for actor in actors:
                    load_futures.append(
                        actor.load_unet.remote(
                            unet_path,
                            model_options=model_options,
                        )
                    )
                ray.get(load_futures)
            else:
                # SEQUENTIAL: Leader-follower to avoid RAM spikes
                
                # 1. Leader (Actor 0) Load
                worker0 = ray.get_actor("RayActor:0")
                ray.get(
                    worker0.load_unet.remote(
                        unet_path,
                        model_options=model_options,
                    )
                )

                # 2. Get Ref & Metadata
                base_ref = ray.get(worker0.create_patched_ref.remote())
                gguf_metadata = ray.get(worker0.get_gguf_metadata.remote())

                # Prepare params for fallback
                reload_params = {
                    "unet_path": unet_path,
                    "dequant_dtype": dequant_dtype,
                    "patch_dtype": patch_dtype,
                    "model_options": {}
                }

                # 3. Follower Hydration
                for actor in actors:
                    if actor != worker0:
                        loaded_futures.append(
                            actor.init_gguf_from_ref.remote(
                                base_ref,
                                gguf_metadata,
                                reload_params
                            )
                        )
                ray.get(loaded_futures)
                loaded_futures = []

        for actor in actors:
            if config.meta.total_sp_degree > 1:
                patched_futures.append(actor.patch_usp.remote())

        ray.get(patched_futures)

        # Optional torch.compile
        if torch_compile != "disabled":
            from raylight.nodes import _check_compile_compatible
            compile_ok, compile_reason = _check_compile_compatible(unet_name, config, backend=torch_compile)
            if not compile_ok:
                print(f"[Raylight] WARNING: torch.compile skipped — {compile_reason}")
            else:
                import torch as _torch
                from comfy_api.torch_helpers import set_torch_compile_wrapper
                compile_mode = "reduce-overhead" if torch_compile == "inductor_reduce_overhead" else None
                dynamic_map = {"auto": None, "true": True, "false": False}
                compile_dynamic = dynamic_map[torch_compile_dynamic]
                def _apply_compile(model, mode, dynamic):
                    import torch
                    torch._dynamo.config.cache_size_limit = 64
                    try:
                        torch._dynamo.config.recompile_limit = 256
                    except Exception:
                        pass
                    m = model.clone()
                    diffusion_model = m.get_model_object("diffusion_model")
                    keys = _gguf_discover_compile_keys(diffusion_model)
                    set_torch_compile_wrapper(model=m, backend="inductor", mode=mode, dynamic=dynamic, keys=keys)
                    return m
                compile_futures = [actor.model_function_runner.remote(_apply_compile, compile_mode, compile_dynamic) for actor in actors]
                ray.get(compile_futures)
                parts = ["transformer blocks"]
                if compile_mode: parts.append(f"mode={compile_mode}")
                if torch_compile_dynamic != "auto": parts.append(f"dynamic={torch_compile_dynamic}")
                print(f"[Raylight] torch.compile (inductor, {', '.join(parts)}) applied to all GGUF actors")

        return ({"actors": actors},)


NODE_CLASS_MAPPINGS = {
    "RayGGUFLoader": RayGGUFLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayGGUFLoader": "Load Diffusion GGUF Model (Ray)",
}
