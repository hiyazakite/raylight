import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import ray
import ray.actor
from typing import List, Dict, Any, Optional

import comfy.patcher_extension as pe

import raylight.distributed_modules.attention as xfuser_attn

from raylight.distributed_modules.usp import USPInjectRegistry
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry


from raylight.utils.common import patch_ray_tqdm
from raylight.utils.gguf import check_mmap_leak
from raylight.config import RaylightConfig
from raylight.utils.memory import monitor_memory, MemoryPolicy
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from raylight.comfy_dist.utils import cancellable_get

from raylight.distributed_actor.managers.lora_manager import LoraManager
from raylight.distributed_actor.managers.vae_manager import VaeManager
from raylight.distributed_actor.managers.sampler_manager import SamplerManager
from raylight.distributed_actor.managers.idlora_manager import IDLoraManager
from raylight.distributed_actor.loaded_model import LoadedModel



# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.



# Comfy cli args, does not get pass through into ray actor

from .actor_config import ActorConfig


_PATCHED = False

def _patch_once():
    """Apply global monkey-patches exactly once per process."""
    global _PATCHED
    if _PATCHED:
        return
    import comfy.model_patcher as model_patcher
    from raylight.comfy_dist.model_patcher import LowVramPatch
    import comfy.lora
    from raylight.comfy_dist.lora import calculate_weight

    model_patcher.LowVramPatch = LowVramPatch
    comfy.lora.calculate_weight = calculate_weight

    try:
        from raylight.comfy_extra_dist.int8 import _register_layouts
        _register_layouts()
    except Exception:
        pass  # INT8 support optional

    _PATCHED = True


class RayActor:

    # -- model slot (Phase 1: property facade over LoadedModel) -----------

    @property
    def model(self):
        return self._loaded.patcher if self._loaded else None

    @model.setter
    def model(self, value):
        if value is None:
            self._loaded = None
        elif self._loaded is not None:
            self._loaded.patcher = value
        else:
            self._loaded = LoadedModel(patcher=value)

    @property
    def is_gguf(self):
        return self._loaded.is_gguf if self._loaded else False

    @is_gguf.setter
    def is_gguf(self, value):
        if self._loaded is not None:
            self._loaded.is_gguf = value

    @property
    def base_sd_ref(self):
        return self._loaded.base_sd_ref if self._loaded else None

    @base_sd_ref.setter
    def base_sd_ref(self, value):
        if self._loaded is not None:
            self._loaded.base_sd_ref = value

    @property
    def gguf_metadata(self):
        return self._loaded.gguf_metadata if self._loaded else {}

    @gguf_metadata.setter
    def gguf_metadata(self, value):
        if self._loaded is not None:
            self._loaded.gguf_metadata = value

    def __init__(self, local_rank, device_id, parallel_dict, raylight_config=None):
        self.local_rank = local_rank
        self.device_id = device_id
        self.parallel_dict = parallel_dict
        
        # [SENIOR REFACTOR] Fallback to from_env if not provided (safety)
        from raylight.config import RaylightConfig
        self.raylight_config = raylight_config or RaylightConfig.from_env()

        self.global_world_size = self.raylight_config.device.world_size

        self._loaded: Optional[LoadedModel] = None
        self.state_dict = None
        self.overwrite_cast_dtype = None
        
        # Create Immutable Config for managers
        self.config = ActorConfig(
            local_rank=self.local_rank,
            device_id=self.device_id,
            device=torch.device(f"cuda:{self.device_id}"),
            parallel_dict=self.parallel_dict,
            global_world_size=self.global_world_size,
            raylight_config=self.raylight_config
        )

        # Managers
        self.lora_manager = LoraManager() # Stateless
        self.vae_manager = VaeManager() # Stateless
        self.sampler_manager = SamplerManager() # Stateless initialization
        self.idlora_manager = IDLoraManager() # Stateless initialization
        

        
        # Stub for state_cache parameter (LRU cache removed — OS page cache
        # handles mmap re-reads; pinned-cache / shm handle their own lifecycle).
        self.state_cache = None
        self.worker_mmap_cache = None

        _patch_once()
        print(f"[RayActor {self.local_rank}] Applied global monkey-patches.")

        self._init_device_and_comms(local_rank)

    # ------------------------------------------------------------------ cache

    def _teardown_pinned_cache(self):
        """Explicitly free all pinned RAM held by the current model's cache.

        Called on: model switch, kill, error-triggered destruction.
        Safe to call when no model / no cache is attached.
        Handles both standard (pinned_param_cache) and FSDP (fsdp_shard_cache).
        """
        if self.model is None:
            return

        # Standard pinned param cache (non-FSDP)
        cache = getattr(self.model, "pinned_param_cache", None)
        if cache is not None:
            try:
                cache.cleanup()
            except Exception as e:
                print(f"[RayActor {self.local_rank}] Pinned cache cleanup error: {e}")

        # FSDP shard cache
        shard_cache = getattr(self.model, "fsdp_shard_cache", None)
        if shard_cache is not None:
            try:
                shard_cache.cleanup()
            except Exception as e:
                print(f"[RayActor {self.local_rank}] FSDP shard cache cleanup error: {e}")

    def release_pinned_cache(self):
        """Public remote-callable wrapper: free pinned RAM on demand.

        Called when workers are being torn down or an explicit deep cleanup
        is requested.  After this call the model's parameters are empty
        tensors, so ``is_model_loaded`` is set to False to force a full
        reload from disk on the next invocation.
        """
        # Detach model params from pinned/shm tensors BEFORE cache cleanup.
        # Otherwise the model's nn.Parameters still hold references into
        # the shm buffer and prevent SharedMemory.close().
        if self.model is not None:
            diff_model = getattr(self.model, "model", None)
            if diff_model is not None:
                for p in diff_model.parameters():
                    # Replace with a tiny empty tensor to release the shm view
                    p.data = torch.empty(0, dtype=p.dtype, device="cpu")
                for b in diff_model.buffers():
                    b.data = torch.empty(0, dtype=b.dtype, device="cpu")
        self._teardown_pinned_cache()
        # Drop the model reference so any weakref.finalize handlers (e.g.
        # GGUF shm cleanup) fire promptly instead of waiting for process
        # exit.  This is critical for freeing /dev/shm segments that are
        # managed via weakref finalizers rather than pinned_param_cache.
        self.model = None
        # Mark the model as NOT loaded so the next run does a full disk reload
        # instead of trying to hot-load from the (now-empty) parameters.
        self.is_model_loaded = False
        self.memory.teardown()
        import gc; gc.collect()
        print(f"[RayActor {self.local_rank}] Pinned cache released via remote call.")

    # ------------------------------------------------------------------ init (continued)

    def _init_device_and_comms(self, local_rank):
        """Second phase of __init__: device, process group, device mesh."""
        self.device = torch.device(f"cuda:{self.device_id}")
        self.memory = MemoryPolicy(device=self.device)
        self.device_mesh = None
        self.compute_capability = int("{}{}".format(*torch.cuda.get_device_capability()))

        self.is_model_loaded = False
        self.is_cpu_offload = self.raylight_config.strategy.fsdp_cpu_offload

        self._init_process_group(local_rank)
        self._init_device_mesh()

        # Inject unified attention config (Always sync to the module)
        xfuser_attn.set_config(self.raylight_config)

        self._init_xdit()

    def _init_process_group(self, local_rank):
        """Initialize NCCL (Linux) or gloo (Windows) process group."""
        os.environ["XDIT_LOGGING_LEVEL"] = "WARN"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)

        if sys.platform.startswith("linux"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=self.global_world_size,
                timeout=timedelta(minutes=1),
            )
        elif sys.platform.startswith("win"):
            os.environ["USE_LIBUV"] = "0"
            dist.init_process_group(
                "gloo",
                rank=local_rank,
                world_size=self.global_world_size,
                timeout=timedelta(minutes=1),
            )

    def _init_device_mesh(self):
        """Initialize Device Mesh for FSDP, Sequence Parallelism, and/or TP."""
        if not self.raylight_config.meta.is_distributed:
            return

        strategy = self.raylight_config.strategy
        tp_degree = strategy.tensor_parallel_degree
        needs_fsdp_or_sp = (
            strategy.fsdp_enabled or self.raylight_config.meta.total_sp_degree > 1
        )

        if needs_fsdp_or_sp and tp_degree > 1:
            # Hybrid FSDP/SP + TP: build a 2-D DeviceMesh so each
            # parallelism uses its own mesh dimension.
            from raylight.comfy_dist.fsdp_utils import build_hybrid_mesh

            fsdp_size = self.global_world_size // tp_degree
            self.device_mesh = build_hybrid_mesh(fsdp_size, tp_degree)
            self.config.device_mesh = self.device_mesh
            print(
                f"[RayActor] 2-D DeviceMesh: fsdp={fsdp_size} x tp={tp_degree}"
            )
        elif needs_fsdp_or_sp:
            # FSDP or SP only — flat 1-D mesh (existing behaviour).
            self.device_mesh = dist.device_mesh.init_device_mesh(
                "cuda", mesh_shape=(self.global_world_size,)
            )
            self.config.device_mesh = self.device_mesh
        elif tp_degree > 1:
            # Pure-TP: no FSDP mesh needed.  TP groups are created via
            # dist.new_group in _init_tensor_parallel.
            self.device_mesh = None
        else:
            print(
                f"Running Ray in normal separate sampler with: "
                f"{self.global_world_size} number of workers"
            )

    def _init_xdit(self):
        """Initialize XDiT environment for Sequence Parallelism and/or TP."""
        strategy = self.raylight_config.strategy

        # Initialize TP if enabled (must happen before SP so that the TP
        # process groups are available when SP layers are constructed).
        if strategy.tensor_parallel_degree > 1:
            self._init_tensor_parallel()

        if self.raylight_config.meta.total_sp_degree <= 1:
            return

        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        self.ulysses_degree = strategy.ulysses_degree
        self.ring_degree = strategy.ring_degree
        self.cfg_degree = strategy.cfg_degree
        self.cp_degree = self.ulysses_degree * self.ring_degree

        # Cleanup previous xfuser state if any (Robustness for restarts)
        try:
            from xfuser.core.distributed import parallel_state
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            # DO NOT destroy the main process group here!
            # It was just initialized above and xfuser will reuse it.
        except Exception as e:
            print(f"[Raylight] xfuser cleanup warning: {e}")

        init_distributed_environment(
            rank=self.local_rank,
            world_size=self.global_world_size,
            local_rank=self.local_rank
        )
        print("[RayActor] XDiT (Sequence Parallelism) Enabled.")

        initialize_model_parallel(
            sequence_parallel_degree=self.cp_degree,
            classifier_free_guidance_degree=self.cfg_degree,
            ring_degree=self.ring_degree,
            ulysses_degree=self.ulysses_degree
        )

        print(
            f"  -> Parallel Degrees: Ulysses={self.ulysses_degree}, Ring={self.ring_degree}, CFG={self.cfg_degree}"
        )

    def _init_tensor_parallel(self):
        """Initialize Tensor Parallel process group.

        Two paths:
        - Hybrid (FSDP+TP): extract the TP group from the 2-D DeviceMesh
          built by ``_init_device_mesh``.
        - Pure-TP: create TP groups via ``dist.new_group``.
        """
        from raylight.distributed_modules.tensor_parallel import TensorParallelState

        tp_degree = self.raylight_config.strategy.tensor_parallel_degree
        print(f"[RayActor] Tensor Parallelism Enabled (TP={tp_degree})")

        if self.device_mesh is not None and self.device_mesh.ndim == 2:
            # Hybrid path — TP group comes from the mesh's "tp" dimension.
            from raylight.comfy_dist.fsdp_utils import initialize_tp_from_mesh
            initialize_tp_from_mesh(self.device_mesh)
        else:
            # Pure-TP path — creates contiguous-rank TP subgroups.
            TensorParallelState.initialize(tp_degree)

        # Apply model-specific TP patching if a handler is registered and
        # a model is already loaded.
        if hasattr(self, "model") and self.model is not None:
            from raylight.comfy_dist.tp_registry import TPRegistry
            if TPRegistry.has_handler(self.model):
                from raylight.distributed_modules.tp_compress import TPCompressConfig
                _strategy = self.raylight_config.strategy
                _compress_config = TPCompressConfig(
                    mode=_strategy.tp_allreduce_compress,
                    bits=_strategy.tp_compress_bits,
                    group_size=_strategy.tp_compress_group_size,
                    use_residual=_strategy.tp_compress_residual,
                    rotation=_strategy.tp_compress_rotation,
                )
                TPRegistry.apply(
                    self.model,
                    tp_group=TensorParallelState.get_group(),
                    compress_config=_compress_config,
                )


    def get_meta_model(self):
        if self.model is None or not hasattr(self.model, "model") or self.model.model is None:
            raise ValueError("No model loaded or model attribute missing")
        first_param_device = next(self.model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            return self.model
        else:
            raise ValueError("Model recieved is not meta, can cause OOM in large model")

    def set_meta_model(self, model):
        if model is None or not hasattr(model, "model") or model.model is None:
             raise ValueError("Model passed to set_meta_model is None or missing model attribute")
             
        first_param_device = next(model.model.parameters()).device
        if first_param_device == torch.device("meta"):
            self.model = model
            self.model.config_fsdp(self.local_rank, self.device_mesh)
        else:
            raise ValueError("Model being set is not meta, can cause OOM in large model")

    def set_state_dict(self):
        if self.model is not None and self.state_dict is not None:
            self.model.set_fsdp_state_dict(self.state_dict)

    def get_compute_capability(self):
        return self.compute_capability


    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict

    def set_tp_compress_config(self, mode: str, bits: int, rotation: str, residual: bool):
        """Update TP compression config on all TPLinear layers at runtime.

        When mode="none", removes compressors (disabling compression).
        Otherwise, rebuilds compressors with the new config.
        """
        from raylight.distributed_modules.tp_compress import TPCompressConfig, TPCompressor
        from raylight.distributed_modules.tensor_parallel import TPLinear
        from raylight.distributed_modules.tp_linear_factory import TPGGMLLinear

        if self.model is None:
            return

        diffusion_model = None
        if hasattr(self.model, "get_model_object"):
            try:
                diffusion_model = self.model.get_model_object("diffusion_model")
            except Exception:
                pass
        if diffusion_model is None and hasattr(self.model, "model"):
            diffusion_model = getattr(self.model.model, "diffusion_model", self.model.model)
        if diffusion_model is None:
            return

        new_config = TPCompressConfig(
            mode=mode,
            bits=bits,
            group_size=self.raylight_config.strategy.tp_compress_group_size,
            use_residual=residual,
            rotation=rotation,
        )

        layer_id = 0
        for module in diffusion_model.modules():
            if isinstance(module, (TPLinear, TPGGMLLinear)) and module.parallelism == "row":
                if mode == "none":
                    module.compressor = None
                else:
                    device = next(module.parameters()).device
                    module.compressor = TPCompressor(
                        new_config, module.out_features, layer_id, device,
                    )
                layer_id += 1

        # Also update the strategy so any future TP patching is consistent
        from dataclasses import replace
        new_strategy = replace(
            self.raylight_config.strategy,
            tp_allreduce_compress=mode,
            tp_compress_bits=bits,
            tp_compress_residual=residual,
            tp_compress_rotation=rotation,
        )
        object.__setattr__(self.raylight_config, 'strategy', new_strategy)

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)



    def model_function_runner_get_values(self, fn, *args, **kwargs):
        return fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_device_id(self):
        return self.device_id

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def offload_and_clear(self):
        """
        Offloads the model from VRAM and clears tracking state.
        Format-specific logic is delegated to ModelContext.
        """
        if self.model is None:
            print(f"[RayActor {self.local_rank}] No model to offload.")
            return
        
        from raylight.distributed_actor.model_context import get_context
        unet_path = getattr(self.model, "unet_path", "")
        ctx = get_context(unet_path, self.config)
        is_destroyed = ctx.offload(self.model, self.lora_manager, self.state_cache, self.config, memory=self.memory)
        if is_destroyed:
            print(f"[RayActor {self.local_rank}] Model offload resulted in destruction. Clearing reference.")
            self._teardown_pinned_cache()
            self.model = None
        else:
             # Soft offload, keep reference but mark current device as cpu (done by context)
             pass  # type: ignore[arg-type]
        
        print(f"[RayActor {self.local_rank}] Offload complete (Model object preserved for hot-reload).")

    def patch_cfg(self):
        if self.model is not None:
            self.model.add_wrapper(
                pe.WrappersMP.DIFFUSION_MODEL,
                CFGParallelInjectRegistry.inject(self.model)
            )

    def patch_usp(self):
        if self.model is not None:
            self.model.add_callback(
                pe.CallbacksMP.ON_LOAD,
                USPInjectRegistry.inject,
            )

    def apply_ffn_chunking(self, num_chunks: int = 8, verbose: bool = True):
        """
        Apply FFN chunking to reduce peak VRAM for LTX-2 models.
        
        This wraps FeedForward layers with ChunkedFFN to process sequences
        in smaller batches, reducing peak memory by ~8x.
        """
        if self.model is None:
            print(f"[RayActor {self.local_rank}] No model loaded, skipping FFN chunking.")
            return {"ffn_found": 0, "ffn_wrapped": 0}
        
        from raylight.comfy_extra_dist.nodes_ltx_ffn_chunker import wrap_ffn_layers
        
        if verbose:
            print(f"[RayActor {self.local_rank}] Applying FFN chunking with {num_chunks} chunks...")
        
        info = wrap_ffn_layers(self.model, num_chunks, verbose)
        
        if verbose:
            print(f"[RayActor {self.local_rank}] FFN chunking complete: {info}")
        
        return info


    def apply_cachedit(self, config: dict):
        """Apply CacheDiT inter-step caching to the local transformer.

        Resolves model_type == "Auto" from the loaded transformer class, then
        patches transformer.forward with the lightweight warmup+skip cache.
        Config keys: enabled, model_type, warmup_steps, skip_interval, noise_scale.
        """
        if self.model is None:
            print(f"[RayActor {self.local_rank}] No model loaded, skipping CacheDiT.")
            return False

        from raylight.comfy_extra_dist.nodes_cachedit import (
            MODEL_PRESETS,
            CacheDiTConfig,
            _auto_detect_model_type,
            _cleanup_transformer_cache,
            _enable_lightweight_cache,
        )

        if not config.get("enabled", True):
            transformer = getattr(getattr(self.model, "model", None), "diffusion_model", self.model)
            _cleanup_transformer_cache(transformer)
            print(f"[RayActor {self.local_rank}] CacheDiT disabled — cache cleared.")
            return True

        # Resolve transformer
        transformer = getattr(getattr(self.model, "model", None), "diffusion_model", None)
        if transformer is None:
            transformer = self.model

        # Resolve model type
        model_type = config.get("model_type", "Auto")
        if model_type == "Auto":
            model_type = _auto_detect_model_type(transformer)

        preset = MODEL_PRESETS.get(model_type, MODEL_PRESETS["Custom"])
        warmup = config.get("warmup_steps", 0)
        skip = config.get("skip_interval", 0)
        noise = config.get("noise_scale", -1.0)

        resolved_warmup = warmup if warmup > 0 else preset.warmup_steps
        resolved_skip = skip if skip > 0 else preset.skip_interval
        resolved_noise = noise if noise >= 0.0 else preset.noise_scale

        print(
            f"[RayActor {self.local_rank}] CacheDiT: model_type={model_type}, "
            f"warmup={resolved_warmup}, skip={resolved_skip}, noise={resolved_noise:.4f}"
        )

        _enable_lightweight_cache(
            transformer,
            warmup_steps=resolved_warmup,
            skip_interval=resolved_skip,
            noise_scale=resolved_noise,
        )
        return True




    def _load_via_context(self, unet_path, model_options, dequant_dtype=None, patch_dtype=None, force_reload=False):
        """Unified internal loader that leverages ModelContext abstraction."""
        from raylight.distributed_actor.model_context import get_context, ModelState
        
        state = ModelState(unet_path, model_options, dequant_dtype, patch_dtype)
        
        # Check if we can skip load (Idempotency)
        # We compare the requested 'state' against the current model's 'load_config'
        current_config = getattr(self.model, "load_config", None)
        
        if self.model is not None and current_config == state:
             # Exact match of configuration
             if not force_reload:
                 print(f"[RayActor {self.local_rank}] Idempotent Load: {os.path.basename(unet_path)} already loaded.")
                 return

        ctx = get_context(unet_path, self.config, model_options=model_options)

        # OPTIMIZATION: Clear existing model before loading new one to prevent memory spike (2x weights)
        if self.model is not None:
            print(f"[RayActor {self.local_rank}] Clearing previous model to free memory before reload...")
            self._teardown_pinned_cache()
            self.model = None
            self.memory.after_model_swap()
        
        # FUNCTIONAL LOAD: Update references from return values
        self.model = ctx.load(state, self.config, self.state_cache)

        # Stamp user VRAM limit on model for hot_load budget calculation
        vram_limit = self.config.vram_limit_bytes
        if self.model is not None and vram_limit > 0:
            self.model.vram_limit_bytes = vram_limit

        # Format flags and coordinating refs
        self.is_gguf = unet_path.lower().endswith(".gguf") or dequant_dtype is not None
        if self.is_gguf and self.model is not None and hasattr(self.model, "load"):
            # Budget-aware initial load: if vram_limit_bytes is set the
            # budget enforcement inside GGUFModelPatcher.load() will cap
            # lowvram_model_memory so ComfyUI places only budget-worth of
            # modules on CUDA and streams the rest per-layer.
            vram_limit = self.config.vram_limit_bytes
            if vram_limit > 0:
                print(f"[RayActor {self.local_rank}] GGUF: Partial VRAM load (limit {vram_limit / 1e9:.1f} GB)...")
            else:
                print(f"[RayActor {self.local_rank}] GGUF: Forcing immediate VRAM load...")
            self.model.load(self.device, force_patch_weights=True)
            self.model.current_device = self.device
            self.memory.after_reload()
            check_mmap_leak(unet_path)
        elif self.model is not None:
            # Non-GGUF (TP safetensors, legacy): activate via context so the
            # model is on CUDA *before* any LoRA or sampler reload_if_needed.
            ctx.activate(self.model, self.device, memory=self.memory)

        import ray
        self.base_sd_ref = ray.put({
            "mode": "disk_parallel",
            "path": unet_path,
            "is_gguf": self.is_gguf,
            "model_options": model_options,
            "dequant_dtype": dequant_dtype,
            "patch_dtype": patch_dtype,
        })

        # ── Phase 4: Register pinned cache for pressure-triggered offload ──
        cache = getattr(self.model, 'pinned_param_cache', None)
        diff_model = getattr(self.model, 'model', None)
        if cache is not None and diff_model is not None:
            self.memory.set_offload_target(cache, diff_model)

            # Store policy on the model patcher so per-block eviction hooks
            # (Phase 1) can access it from forward-hook closures.
            self.model._memory_policy = self.memory

            # Register mmap state dict as fallback for pressure eviction.
            # When the pinned cache isn't built yet, the interceptor can
            # still free VRAM by dropping CUDA storages — the data is
            # recoverable from the mmap source without extra RAM allocation.
            mmap_sd = getattr(self.model, 'mmap_cache', None)
            if mmap_sd is not None:
                cache.set_mmap_fallback(mmap_sd)

            # Install CUDA allocator interceptor — on-demand eviction when
            # any cudaMalloc fails (handles unpredictable activation spikes).
            try:
                from raylight.lib.alloc_interceptor import install_interceptor
                install_interceptor(self.memory, log_level=1)
            except Exception as e:
                print(f"[RayActor {self.local_rank}] Allocator interceptor not available: {e}")

            # Register CompactFusion cache evictor — frees comm buffers
            # when VRAM pressure is detected (cheaper than weight eviction).
            try:
                from raylight.distributed_modules.attention.backends.fusion.main import compact_evict_cuda_caches
                self.memory.set_compact_cache_evictor(compact_evict_cuda_caches)
            except Exception:
                pass  # CompactFusion may not be available

    def load_unet(self, unet_path, model_options):
        """Unified entry point for all model types (FSDP, GGUF, Safetensors, BNB)."""
        with monitor_memory(f"RayActor {self.local_rank} - load_unet", device=self.device):
            # Unified Path: FSDP, GGUF, Safetensors & BNB
            self._load_via_context(unet_path, model_options)
            
            # FSDP Specific: Link the lazy state dict to the patcher
            if self.config.is_fsdp:
                    # Note: model_context.load() already attaches fsdp_state_dict to the model returned.
                    # We only need to set it if we have an explicit override in self.state_dict.
                if self.state_dict is not None:
                    self.set_state_dict()
                
            self.is_model_loaded = True

            # Explicitly mark as baked for LoRA reloading logic
            if self.config.is_fsdp and self.model is not None:
                self.model.is_fsdp_baked = True

            return True

    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype):
        with monitor_memory(f"RayActor {self.local_rank} - load_gguf_unet", device=self.device):
            if self.config.is_fsdp is True:
                raise ValueError("FSDP Sharding for GGUF is not supported")
            
            self._load_via_context(unet_path, {}, dequant_dtype, patch_dtype)
            self.is_model_loaded = True
            return True

    def init_gguf_from_ref(self, ref, metadata, reload_params=None):
        print(f"[RayActor {self.local_rank}] Initializing GGUF from Shared Reference (Follower Mode)...")
        # Initialize empty model structure
        self.model = ref 
        
        self.gguf_metadata = metadata
        self.is_gguf = True
        
        if reload_params is not None:
             print(f"[RayActor {self.local_rank}] GGUF Optimization: Using Unified Parallel Disk Loading (Mmap)...")
             if self.config.is_fsdp is True:
                raise ValueError("FSDP Sharding for GGUF is not supported")
            
             self._load_via_context(
                 reload_params["unet_path"],
                 reload_params["model_options"],
                 reload_params.get("dequant_dtype"),
                 reload_params.get("patch_dtype")
             )
             self.is_model_loaded = True
             return
        else:
             raise ValueError("[RayActor] Cannot RELOAD GGUF: Missing reload params for parallel loading.")

    # load_bnb_unet removed - migrated to BNBContext

    def load_lora(self, lora_path, strength_model, lora_config_hash=None, seed=318008):
        self.reload_model_if_needed()
        return self.lora_manager.load_lora(
            self.model,
            self.config,
            lora_path,
            strength_model,
            lora_config_hash,
            seed=seed,
        )

    def _force_reload_from_config(self):
        """Reload the current model from its saved ModelState.

        Used when baked state is dirty (e.g. FSDP weight modification) and we
        need a clean copy before applying a different LoRA configuration.
        """
        state = getattr(self.model, "load_config", None)
        if not state:
            raise RuntimeError("Cannot force reload: model has no load_config.")
        self._load_via_context(
            state.unet_path, state.model_options,
            state.dequant_dtype, state.patch_dtype,
            force_reload=True,
        )
        self.lora_manager.clear_tracking()

    def _try_bake_and_sync(self) -> None:
        """Post-sampling: bake FSDP LoRA hooks into weights + sync to shard cache.

        Conditions to bake:
          1. Model is FSDP
          2. Model has active LoRA patches (hooks on modules)
          3. LoRA config hasn't already been baked + cached

        After baking:
          - LowVramPatch hooks are cleared (zero per-forward overhead)
          - Shard cache is updated with baked values (eviction+reload preserves baked state)
          - lora_manager records this config as baked
        """
        if not self.config.is_fsdp or self.model is None:
            return

        config_hash = self.lora_manager._current_lora_config_hash
        if config_hash is None:
            return  # No LoRA active

        if self.lora_manager.is_baked_for(config_hash):
            return  # Already baked + synced for this config

        patches = getattr(self.model, "patches", None)
        if not patches:
            # No patches = either no LoRA, or already baked.
            # If model is marked baked but cache doesn't know, sync now.
            if getattr(self.model, "is_fsdp_baked", False):
                self._sync_shard_cache()
                self.lora_manager.mark_baked(config_hash)
            return

        # ── Bake hooks ──
        from raylight.comfy_dist.fsdp_utils import bake_lora_hooks
        baked = bake_lora_hooks(self.model, local_rank=self.local_rank)
        if baked == 0:
            return

        # ── Sync baked CUDA → pinned shard cache ──
        self._sync_shard_cache()

        # ── Mark as baked ──
        self.lora_manager.mark_baked(config_hash)
        print(f"[RayActor {self.local_rank}] LoRA config {config_hash} baked + cached in shard cache.")

    def _sync_shard_cache(self) -> None:
        """Sync current CUDA shard data → pinned CPU cache."""
        shard_cache = getattr(self.model, "fsdp_shard_cache", None)
        diff_model = getattr(self.model, "model", None)
        if shard_cache is not None and diff_model is not None and shard_cache.built:
            try:
                dm = getattr(diff_model, "diffusion_model", diff_model)
                shard_cache.sync_from_cuda(dm)
            except Exception as e:
                print(f"[RayActor {self.local_rank}] Shard cache sync failed: {e}")

    def reapply_loras_for_config(self, config_hash):
        # SAFETY: If model is baked (FSDP destructively modified weights), we cannot unbake.
        # We must reload from base reference if we need to change LoRA config.
        current_hash = getattr(self.lora_manager, "_current_lora_config_hash", None)
        
        if (self.model is not None and 
            (getattr(self.model, "is_fsdp_baked", False) or self.config.is_fsdp) and 
            current_hash != config_hash):
            
            print(f"[RayActor {self.local_rank}] Model is FSDP-baked and LoRA config changed ({current_hash} -> {config_hash}). Forcing base reload.")
            self._force_reload_from_config()

        self.reload_model_if_needed()
        return self.lora_manager.reapply_loras_for_config(
            self.model, 
            self.config, 
            config_hash
        )

    def apply_model_sampling(self, model_sampling_patch):
        if self.model is not None:
            self.model.add_object_patch("model_sampling", model_sampling_patch)

    def set_base_ref(self, ref):
        self.base_sd_ref = ref

    def create_patched_ref(self):
        if not hasattr(self, 'base_sd_ref') or self.base_sd_ref is None:
            raise RuntimeError("Base Ref not set!")
        return self.base_sd_ref

    def get_gguf_metadata(self):
        return getattr(self, "gguf_metadata", {})



    def load_unet_from_state_dict(self, state_dict, model_options):
        import ray
        if isinstance(state_dict, ray.ObjectRef):
            if self.model is not None and getattr(self, "current_sd_ref", None) == state_dict:
                 print(f"[RayActor {self.local_rank}] Smart Reload Ref match.")
                 self.is_model_loaded = True
                 return

            self.current_sd_ref = state_dict
            state_dict = ray.get(state_dict)

            if isinstance(state_dict, dict):
                if state_dict.get("mode") == "disk_parallel":
                     print(f"[RayActor {self.local_rank}] Reloading from disk reference (FSDP/Parallel aware)...")
                     self._load_via_context(
                         state_dict["path"],
                         state_dict.get("model_options", {}),
                         state_dict.get("dequant_dtype"),
                         state_dict.get("patch_dtype")
                     )
                     return

        from raylight.distributed_actor.model_context import get_context, ModelState
        is_gguf = getattr(self, "is_gguf", False)
        if isinstance(state_dict, dict) and state_dict.get("is_gguf"):
            is_gguf = True
        
        # Determine path from loaded model config (current_unet_path eliminated)
        unet_path = ""
        if self.model is not None:
            load_config = getattr(self.model, "load_config", None)
            if load_config:
                unet_path = load_config.unet_path
        if not unet_path:
             # Fallback if unet_path isn't set (e.g. strict state dict load), rely on config or default
             unet_path = "model.gguf" if is_gguf else "model.safetensors"

        ctx = get_context(unet_path, self.config)
        state_dict = ctx.prepare_state_dict(state_dict, self.config)
        
        # FIX: Ensure GGUF config is propagated during full reload
        dequant_dtype = None
        patch_dtype = None
        
        # Try to get dtypes from existing model config if available
        if self.model is not None and hasattr(self.model, "load_config"):
             dequant_dtype = self.model.load_config.dequant_dtype
             patch_dtype = self.model.load_config.patch_dtype
             
        # Also check model_options if passed explicitly
        if isinstance(model_options, dict):
             if "dequant_dtype" in model_options: dequant_dtype = model_options["dequant_dtype"]
             if "patch_dtype" in model_options: patch_dtype = model_options["patch_dtype"]

        state = ModelState(
             unet_path=unet_path, 
             model_options=model_options,
             dequant_dtype=dequant_dtype,
             patch_dtype=patch_dtype
        )

        
        self.model = ctx.instantiate_model(state_dict, state, self.config)
        
        self.is_model_loaded = True
        self.memory.after_reload()

    def reload_model_if_needed(self, load_device=None):
        from raylight.distributed_actor.model_context import get_context
        target_device = load_device if load_device is not None else self.device
        
        if self.model is not None:
            # Use attached config to get context
            config_state = getattr(self.model, "load_config", None)
            path = config_state.unet_path if config_state else getattr(self.model, "unet_path", "unknown")
            opts = config_state.model_options if config_state else None

            ctx = get_context(path, self.config, model_options=opts)

            # Context checks if migration is needed
            self.model = ctx.reload_if_needed(
                self.model,
                target_device,
                self.config,
            )

        elif hasattr(self, 'base_sd_ref') and self.base_sd_ref is not None:
            # Fallback for state_dict based loading
            print(f"[RayActor] Reloading from base_sd_ref...")
            self.load_unet_from_state_dict(self.base_sd_ref, model_options={})
        else:
            # No model loaded
            pass


    def kill(self):
        self._teardown_pinned_cache()
        self.model = None
        dist.destroy_process_group()
        ray.actor.exit_actor()

    def ray_vae_loader(self, vae_path):
        # ── Pressure-aware VAE load: partially evict UNET tail instead of full offload ──
        # Estimate VAE weight size from file (avoid loading just to measure).
        import os
        vae_file_bytes = 0
        try:
            vae_file_bytes = os.path.getsize(vae_path)
        except Exception:
            pass
        if vae_file_bytes > 0:
            freed = self.memory.relieve_pressure(needed_bytes=vae_file_bytes)
            if freed > 0:
                print(f"[RayActor {self.local_rank}] Pressure relief: freed {freed/1e9:.2f} GB "
                      f"of UNET for VAE ({vae_file_bytes/1e9:.2f} GB)")

        self.vae_manager.load_vae(vae_path, self.config, self.state_cache, memory=self.memory)
        self.vae_model = self.vae_manager.vae_model
        # Return compression factors to avoid extra round-trips
        return (
            self.vae_manager.get_temporal_compression(),
            self.vae_manager.get_spatial_compression(),
        )

    def ray_vae_offload(self):
        """Offload VAE from VRAM after all work-stealing chunks complete."""
        self.vae_manager.offload_vae_from_device(self.config, self.lora_manager, memory=self.memory)

        # ── Restore any UNET params evicted by pressure relief during VAE load ──
        cache = getattr(self.model, 'pinned_param_cache', None) if self.model else None
        diff_model = getattr(self.model, 'model', None) if self.model else None
        if cache is not None and diff_model is not None and cache._partial_freed:
            cache.reload_evicted(diff_model)
            print(f"[RayActor {self.local_rank}] UNET tail restored after VAE offload.")

    def ray_vae_release(self):
        result = self.vae_manager.release_vae(self.local_rank, memory=self.memory)
        self.vae_model = None
        return result

    def get_vae_temporal_compression(self):
        return self.vae_manager.get_temporal_compression()

    def get_vae_spatial_compression(self):
        return self.vae_manager.get_spatial_compression()

    def _check_vae_health(self, samples, shard_index):
        self.vae_manager.check_health(samples, shard_index, self.config)

    def ray_vae_decode(
        self,
        shard_index,
        samples,
        tile_size,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
        discard_latent_frames=0,
        vae_dtype="auto",
        mmap_path=None,
        mmap_shape=None,
        output_offset=0,
        latent_ref=None,
        latent_slice_start=None,
        latent_slice_end=None,
        skip_offload=False,
    ):
        return self.vae_manager.decode(
            shard_index,
            samples,
            tile_size,
            overlap,
            temporal_size,
            temporal_overlap,
            discard_latent_frames,
            vae_dtype,
            mmap_path,
            mmap_shape,
            output_offset,
            config=self.config,
            lora_manager=self.lora_manager,
            state_cache=self.state_cache,
            latent_ref=latent_ref,
            latent_slice_start=latent_slice_start,
            latent_slice_end=latent_slice_end,
            skip_offload=skip_offload,
            memory=self.memory,
        )

    def get_raylight_config(self) -> RaylightConfig:
        return self.raylight_config

    def get_parallel_dict(self) -> Dict[str, Any]:
        return self.parallel_dict


    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def custom_sampler(
        self,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
    ):
        result, model_modified = self.sampler_manager.custom_sampler(
            self.model,
            self.config,
            add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image,
            state_dict=getattr(self, "state_dict", None),
            memory=self.memory,
        )

        if model_modified:
            print(f"[RayActor {self.local_rank}] Model modified by SamplerManager (e.g. baked weights).")
            # Invalidate pinned cache signature — baked CUDA data diverged from pinned buffers.
            cache = getattr(self.model, 'pinned_param_cache', None)
            if cache is not None:
                cache.invalidate()

        # ── FSDP LoRA bake caching ──────────────────────────────────
        # After the first run with LoRA hooks, bake them into weights and
        # sync to the FSDP shard cache.  Subsequent reruns with the same
        # LoRA config pay zero per-forward overhead.
        self._try_bake_and_sync()
        return result


    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    def common_ksampler(
        self,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        sigmas=None,
    ):
        result, model_modified = self.sampler_manager.common_ksampler(
            self.model,
            self.config,
            seed, steps, cfg, sampler_name, scheduler, positive, negative,
            latent, denoise, disable_noise, start_step, last_step,
            force_full_denoise, sigmas,
             state_dict=getattr(self, "state_dict", None),
             memory=self.memory,
        )
                 
        if model_modified:
             print(f"[RayActor {self.local_rank}] Model modified by SamplerManager (e.g. baked weights).")
             cache = getattr(self.model, 'pinned_param_cache', None)
             if cache is not None:
                 cache.invalidate()

        self._try_bake_and_sync()
        return result

    @patch_temp_fix_ck_ops
    def idlora_sample(
        self,
        video_dict,
        audio_dict,
        positive,
        negative,
        sigmas,
        ref_audio=None,
        denoise_config=None,
        v_clean=None,
        a_clean=None,
    ):
        self.reload_model_if_needed()

        # Context extraction + denoising loop handled entirely in idlora_manager.
        return self.idlora_manager.idlora_denoise(
            model=self.model,
            config=self.config,
            video_dict=video_dict,
            audio_dict=audio_dict,
            positive=positive,
            negative=negative,
            sigmas=sigmas,
            ref_audio=ref_audio,
            denoise_config=denoise_config,
            v_clean=v_clean,
            a_clean=a_clean,
        )




