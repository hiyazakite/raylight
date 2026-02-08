import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import ray

import comfy.patcher_extension as pe

import raylight.distributed_modules.attention as xfuser_attn

from raylight.distributed_modules.usp import USPInjectRegistry
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry


from raylight.utils.common import patch_ray_tqdm, cleanup_memory
from raylight.utils.gguf import check_mmap_leak
from raylight.utils.memory import monitor_memory
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from raylight.comfy_dist.utils import cancellable_get
from ray.exceptions import RayActorError

from raylight.distributed_worker.managers.lora_manager import LoraManager
from raylight.distributed_worker.managers.vae_manager import VaeManager
from raylight.distributed_worker.managers.sampler_manager import SamplerManager



# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.



# Comfy cli args, does not get pass through into ray actor

from .worker_config import WorkerConfig


class RayWorker:



    def __init__(self, local_rank, device_id, parallel_dict):
        self.local_rank = local_rank
        self.device_id = device_id
        self.parallel_dict = parallel_dict
        self.global_world_size = self.parallel_dict["global_world_size"]

        self.model = None
        self.model_type = None
        self.state_dict = None
        self.overwrite_cast_dtype = None
        
        # Create Immutable Config
        self.config = WorkerConfig(
            local_rank=self.local_rank,
            device_id=self.device_id,
            device=torch.device(f"cuda:{self.device_id}"),
            parallel_dict=self.parallel_dict,
            global_world_size=self.global_world_size
        )

        # Managers
        self.lora_manager = LoraManager() # Stateless
        self.vae_manager = VaeManager() # Stateless
        self.sampler_manager = SamplerManager() # Stateless initialization
        

        
        # LRU cache for mmap'd state dicts (replaces worker_mmap_cache)
        from raylight.utils.cache import LRUStateCache
        cache_size = parallel_dict.get("mmap_cache_size", 4) 
        if cache_size < 3:
            print(f"[RayWorker {self.local_rank}] WARNING: mmap_cache_size={cache_size} is too small for UNET+VAE. Forcing to 4.")
            cache_size = 4
        print(f"[RayWorker {self.local_rank}] Initialized LRU State Cache with size={cache_size}")
        self.state_cache = LRUStateCache(max_size=cache_size)
        
        # Legacy alias for backward compatibility
        self.worker_mmap_cache = self.state_cache

        # CRITICAL: Apply LowVramPatch monkey-patch early to ensure ALL model types
        # (GGUF, standard, FSDP) use Raylight's LoRA-compatible patch handler.
        # This must happen before any model loading.
        import comfy.model_patcher as model_patcher
        from raylight.comfy_dist.model_patcher import LowVramPatch
        import comfy.lora
        from raylight.comfy_dist.lora import calculate_weight
        
        # Patch LowVramPatch to use Raylight's LoRA-compatible calculate_weight
        model_patcher.LowVramPatch = LowVramPatch
        # Patch comfy.lora.calculate_weight to support LoRAAdapter objects
        comfy.lora.calculate_weight = calculate_weight
        
        print(f"[RayWorker {self.local_rank}] Applied global monkey-patches.")

        self.device = torch.device(f"cuda:{self.device_id}")
        self.device_mesh = None
        self.compute_capability = int("{}{}".format(*torch.cuda.get_device_capability()))

        self.is_model_loaded = False
        self.is_cpu_offload = self.parallel_dict.get("fsdp_cpu_offload", False)

        os.environ["XDIT_LOGGING_LEVEL"] = "WARN"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)

        if sys.platform.startswith("linux"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=self.global_world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )
        elif sys.platform.startswith("win"):
            os.environ["USE_LIBUV"] = "0"
            dist.init_process_group(
                "gloo",
                rank=local_rank,
                world_size=self.global_world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )

        # (TODO-Komikndr) Should be modified so it can do support DP on top of FSDP
        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(self.global_world_size,))
            self.config.device_mesh = self.device_mesh
        else:
            print(f"Running Ray in normal seperate sampler with: {self.global_world_size} number of workers")

        # From mochi-xdit, xdit, pipelines.py
        if self.parallel_dict["is_xdit"]:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            xfuser_attn.set_attn_type(self.parallel_dict["attention"])
            xfuser_attn.set_sync_ulysses(self.parallel_dict["sync_ulysses"])

            self.cp_degree = self.parallel_dict["ulysses_degree"] * parallel_dict["ring_degree"]
            self.cfg_degree = self.parallel_dict["cfg_degree"]
            self.ulysses_degree = self.parallel_dict["ulysses_degree"]
            self.ring_degree = self.parallel_dict["ring_degree"]
            self.cfg_degree = self.parallel_dict["cfg_degree"]
            
            # Cleanup previous state if any (Robustness for restarts)
            try:
                from xfuser.core.distributed import parallel_state
                if parallel_state.get_world_size() > 1:
                    parallel_state.destroy_model_parallel()
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[Raylight] Cleanup warning: {e}")

            init_distributed_environment(rank=self.local_rank, world_size=self.global_world_size)
            print("XDiT is enable")

            initialize_model_parallel(
                sequence_parallel_degree=self.cp_degree,
                classifier_free_guidance_degree=self.cfg_degree,
                ring_degree=self.ring_degree,
                ulysses_degree=self.ulysses_degree
            )

            print(
                f"Parallel Degree: Ulysses={self.ulysses_degree}, Ring={self.ring_degree}, CFG={self.cfg_degree}"
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

    def get_parallel_dict(self):
        return self.parallel_dict

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)

    # Legacy debug_memory_leaks removed


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
            print(f"[RayWorker {self.local_rank}] No model to offload.")
            return
        
        from raylight.distributed_worker.model_context import get_context
        unet_path = getattr(self.model, "unet_path", "")
        ctx = get_context(unet_path, self.config)
        is_destroyed = ctx.offload(self.model, self.lora_manager, self.state_cache, self.config)
        if is_destroyed:
            print(f"[RayWorker {self.local_rank}] Model offload resulted in destruction. Clearing reference.")
            self.model = None
        else:
             # Soft offload, keep reference but mark current device as cpu (done by context)
             pass  # type: ignore[arg-type]
        
        # Clear references
        # self.model = None  <-- DISABLED to enable Hot-Load (Soft Offload)
        # self.current_unet_path = None
        # self.current_sd_ref = None
        
        print(f"[RayWorker {self.local_rank}] Offload complete (Model object preserved for hot-reload).")

    
    def _common_offload_cleanup(self):
        """Shared post-offload cleanup for all model formats."""
        # Clear LoRA tracking for fresh re-application after reload
        self.lora_manager.clear_tracking()
        print(f"[RayWorker {self.local_rank}] LoRA tracking cleared.")
        
        # Move buffers to CPU if model still exists
        if self.model is not None:
            diffusion_model = getattr(self.model, "model", None)
            if diffusion_model is not None:
                try:
                    for name, buf in diffusion_model.named_buffers():
                        if buf is not None and buf.device.type == 'cuda':
                            buf.data = buf.data.to('cpu')
                except Exception as e:
                    print(f"[RayWorker {self.local_rank}] Buffer cleanup warning: {e}")
                
                # Delete cached PE tensors
                for attr in ['_cached_pe', 'cached_pe', 'pe_cache', '_pe']:
                    if hasattr(diffusion_model, attr):
                        delattr(diffusion_model, attr)
                    if hasattr(diffusion_model, 'diffusion_model'):
                        inner = diffusion_model.diffusion_model
                        if hasattr(inner, attr):
                            delattr(inner, attr)
        
        # Force garbage collection and CUDA cleanup
        cleanup_memory()
        mem_now = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"[RayWorker {self.local_rank}] Post-Offload VRAM: {mem_now:.2f}GB")


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
            print(f"[RayWorker {self.local_rank}] No model loaded, skipping FFN chunking.")
            return {"ffn_found": 0, "ffn_wrapped": 0}
        
        from raylight.comfy_extra_dist.nodes_ltx_ffn_chunker import wrap_ffn_layers
        
        if verbose:
            print(f"[RayWorker {self.local_rank}] Applying FFN chunking with {num_chunks} chunks...")
        
        info = wrap_ffn_layers(self.model, num_chunks, verbose)
        
        if verbose:
            print(f"[RayWorker {self.local_rank}] FFN chunking complete: {info}")
        
        return info




    def _load_via_context(self, unet_path, model_options, dequant_dtype=None, patch_dtype=None, force_reload=False):
        """Unified internal loader that leverages ModelContext abstraction."""
        from raylight.distributed_worker.model_context import get_context, ModelState
        
        state = ModelState(unet_path, model_options, dequant_dtype, patch_dtype)
        
        # Check if we can skip load (Idempotency)
        # We compare the requested 'state' against the current model's 'load_config'
        current_config = getattr(self.model, "load_config", None)
        
        if self.model is not None and current_config == state:
             # Exact match of configuration
             if not force_reload:
                 print(f"[RayWorker {self.local_rank}] Idempotent Load: {os.path.basename(unet_path)} already loaded.")
                 return

        ctx = get_context(unet_path, self.config, model_options=model_options)

        # OPTIMIZATION: Clear existing model before loading new one to prevent memory spike (2x weights)
        if self.model is not None:
            print(f"[RayWorker {self.local_rank}] Clearing previous model to free memory before reload...")
            self.model = None
            cleanup_memory()
        
        # FUNCTIONAL LOAD: Update references from return values
        self.model = ctx.load(state, self.config, self.state_cache)
        
        # Format flags and coordinating refs
        self.is_gguf = unet_path.lower().endswith(".gguf") or dequant_dtype is not None
        if self.is_gguf and self.model is not None and hasattr(self.model, "load"):
            print(f"[RayWorker {self.local_rank}] GGUF: Forcing immediate VRAM load...")
            self.model.load(self.device, force_patch_weights=True)
            self.model.current_device = self.device
            cleanup_memory()
            check_mmap_leak(unet_path)

        import ray
        self.base_sd_ref = ray.put({
            "mode": "disk_parallel",
            "path": unet_path,
            "is_gguf": self.is_gguf,
            "model_options": model_options,
            "dequant_dtype": dequant_dtype,
            "patch_dtype": patch_dtype,
        })

    def load_unet(self, unet_path, model_options):
        """Unified entry point for all model types (FSDP, GGUF, Safetensors, BNB)."""
        with monitor_memory(f"RayWorker {self.local_rank} - load_unet", device=self.device):
            self.current_unet_path = unet_path
            
            # Unified Path: FSDP, GGUF, Safetensors & BNB
            self._load_via_context(unet_path, model_options)
            
            # FSDP Specific: Link the lazy state dict to the patcher
            if self.parallel_dict.get("is_fsdp"):
                    # Note: model_context.load() already attaches fsdp_state_dict to the model returned.
                    # We only need to set it if we have an explicit override in self.state_dict.
                if self.state_dict is not None:
                    self.set_state_dict()
                
            self.is_model_loaded = True
            return True

    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype):
        with monitor_memory(f"RayWorker {self.local_rank} - load_gguf_unet", device=self.device):
            if self.parallel_dict["is_fsdp"] is True:
                raise ValueError("FSDP Sharding for GGUF is not supported")
            
            self._load_via_context(unet_path, {}, dequant_dtype, patch_dtype)
            self.is_model_loaded = True
            return True

    def init_gguf_from_ref(self, ref, metadata, reload_params=None):
        print(f"[RayWorker {self.local_rank}] Initializing GGUF from Shared Reference (Follower Mode)...")
        # Initialize empty model structure
        self.model = ref 
        
        self.gguf_metadata = metadata
        self.is_gguf = True
        
        if reload_params is not None:
             print(f"[RayWorker {self.local_rank}] GGUF Optimization: Using Unified Parallel Disk Loading (Mmap)...")
             if self.parallel_dict["is_fsdp"] is True:
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
             raise ValueError("[RayWorker] Cannot RELOAD GGUF: Missing reload params for parallel loading.")

    # load_bnb_unet removed - migrated to BNBContext

    def load_lora(self, lora_path, strength_model, lora_config_hash=None):
        self.reload_model_if_needed()
        return self.lora_manager.load_lora(
            self.model, 
            self.config, 
            lora_path, 
            strength_model, 
            lora_config_hash
        )

    def reapply_loras_for_config(self, config_hash):
        # SAFETY: If model is baked (FSDP destructively modified weights), we cannot unbake.
        # We must reload from base reference if we need to change LoRA config.
        # FIX: Only force reload if the config hash implies a CHANGE.
        current_hash = getattr(self.lora_manager, "_current_lora_config_hash", None)
        
        if (self.model is not None and 
            getattr(self.model, "is_fsdp_baked", False) and 
            current_hash != config_hash):
            
            print(f"[RayWorker {self.local_rank}] Model is FSDP-baked and LoRA config changed ({current_hash} -> {config_hash}). Forcing base reload.")
            
            # Manual Reload Logic because we can't rely on 'reload_params=None' anymore
            # AND we don't have the explicit args here, we must use the attached config.
            state = getattr(self.model, "load_config", None)
            if state:
                # Clear old
                self.model = None
                cleanup_memory()
                
                # Load new
                from raylight.distributed_worker.model_context import get_context
                ctx = get_context(state.unet_path, self.config, model_options=state.model_options)
                self.model = ctx.load(state, self.config, self.state_cache)
                
                # CRITICAL: Since we reloaded a fresh model, we must reset LoraManager's tracking
                # state so it knows to re-apply patches even if the hash hasn't changed.
                self.lora_manager.clear_tracking()
            else:
                raise RuntimeError("Cannot force reload FSDP model: Missing load_config state.")
            self.lora_manager.clear_tracking() # Inserted line

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
                 print(f"[RayWorker {self.local_rank}] Smart Reload Ref match.")
                 self.is_model_loaded = True
                 return

            self.current_sd_ref = state_dict
            state_dict = ray.get(state_dict)

            if isinstance(state_dict, dict):
                if state_dict.get("mode") == "disk_parallel":
                     print(f"[RayWorker {self.local_rank}] Reloading from disk reference (FSDP/Parallel aware)...")
                     self._load_via_context(
                         state_dict["path"],
                         state_dict.get("model_options", {}),
                         state_dict.get("dequant_dtype"),
                         state_dict.get("patch_dtype")
                     )
                     return

        from raylight.distributed_worker.model_context import get_context, ModelState
        is_gguf = getattr(self, "is_gguf", False)
        if isinstance(state_dict, dict) and state_dict.get("is_gguf"):
            is_gguf = True
        
        # Determine path safely
        unet_path = getattr(self, "current_unet_path", "")
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
        cleanup_memory()

    def reload_model_if_needed(self, load_device=None):
        from raylight.distributed_worker.model_context import get_context
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
                 self.state_cache
             )

        elif hasattr(self, 'base_sd_ref') and self.base_sd_ref is not None:
             # Fallback for state_dict based loading
             print(f"[RayWorker] Reloading from base_sd_ref...")
             self.load_unet_from_state_dict(self.base_sd_ref, model_options={})
        else:
            # No model loaded
            pass


    def kill(self):
        self.model = None
        dist.destroy_process_group()
        ray.actor.exit_actor()

    def ray_vae_loader(self, vae_path):
        self.vae_manager.load_vae(vae_path, self.config, self.state_cache)
        self.vae_model = self.vae_manager.vae_model

    def ray_vae_release(self):
        result = self.vae_manager.release_vae(self.local_rank)
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
        )



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
            state_dict=getattr(self, "state_dict", None)
        )

        if model_modified:
            print(f"[RayWorker {self.local_rank}] Model modified by SamplerManager (e.g. baked weights).")
            # OPTIMIZATION: Do not clear reload_params blindly. 
            # We now handle "Dirty Baked State" safety via is_fsdp_baked flag in reapply_loras_for_config.
            # self.reload_params = None 
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
             state_dict=getattr(self, "state_dict", None)
        )
                 
        if model_modified:
             print(f"[RayWorker {self.local_rank}] Model modified by SamplerManager (e.g. baked weights).")
             # OPTIMIZATION: Do not clear reload_params blindly.
             # self.reload_params = None
        return result



class RayCOMMTester:
    def __init__(self, local_rank, world_size, device_id):
        device = torch.device(f"cuda:{device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        if sys.platform.startswith("linux"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )
        elif sys.platform.startswith("win"):
            os.environ["USE_LIBUV"] = "0"
            if local_rank == 0:
                print("Windows detected, falling back to GLOO backend, consider using WSL, GLOO is slower than NCCL")
            dist.init_process_group(
                "gloo",
                rank=local_rank,
                world_size=world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )
        print("Running COMM pre-run")

        # Each rank contributes rank+1
        x = torch.ones(1, device=device) * (local_rank + 1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        result = x.item()

        # Expected sum = N(N+1)/2
        expected = world_size * (world_size + 1) // 2

        if abs(result - expected) > 1e-3:
            raise RuntimeError(
                f"[Rank {local_rank}] COMM test failed: "
                f"got {result}, expected {expected}. "
                f"world_size may be mismatched!"
            )
        else:
            print(f"[Rank {local_rank}] COMM test passed ✅ (result={result})")

    def kill(self):
        dist.destroy_process_group()
        ray.actor.exit_actor()


def ray_nccl_tester(world_size):
    gpu_actor = ray.remote(RayCOMMTester)
    gpu_actors = []

    for local_rank in range(world_size):
        gpu_actors.append(
            gpu_actor.options(num_gpus=1, name=f"RayTest:{local_rank}").remote(
                local_rank=local_rank,
                world_size=world_size,
                device_id=0,
            )
        )
    for actor in gpu_actors:
        ray.get(actor.__ray_ready__.remote())

    for actor in gpu_actors:
        actor.kill.remote()


def make_ray_actor_fn(
    world_size,
    parallel_dict
):
    def _init_ray_actor(
        world_size=world_size,
        parallel_dict=parallel_dict
    ):
        ray_actors = dict()
        gpu_actor = ray.remote(RayWorker)
        gpu_actors = []

        for local_rank in range(world_size):
            # Prepare runtime_env for backend config
            init_runtime_env = {}
            env_vars = {}
            
            if "attention_backend" in parallel_dict:
                env_vars["RAYLIGHT_ATTN_BACKEND"] = parallel_dict["attention_backend"]
            
            if parallel_dict.get("use_kitchen"):
                env_vars["RAYLIGHT_ENABLE_KITCHEN"] = "1"

            if env_vars:
                init_runtime_env["env_vars"] = env_vars

            gpu_actors.append(
                gpu_actor.options(
                    num_gpus=1, 
                    name=f"RayWorker:{local_rank}",
                    runtime_env=init_runtime_env
                ).remote(
                    local_rank=local_rank,
                    device_id=0,
                    parallel_dict=parallel_dict,
                )
            )
        ray_actors["workers"] = gpu_actors
        ray_actors["parallel_dict"] = parallel_dict
        
        # Parallel initialization check (faster startup)
        # Instead of waiting sequentially, we fire all check tasks and wait for them in parallel
        ready_refs = [actor.__ray_ready__.remote() for actor in ray_actors["workers"]]
        cancellable_get(ready_refs)
        
        return ray_actors

    return _init_ray_actor


def ensure_fresh_actors(ray_actors_init):
    """
    Ensures that the Ray actors are 'fresh' - i.e. if a model is currently 
    loaded, it kills and restarts them to ensure a clean slate.
    """
    ray_actors, ray_actor_fn = ray_actors_init
    gpu_actors = ray_actors["workers"]

    needs_restart = False
    try:
        # Check first actor; if it has a model, we force a restart for 'freshness'
        is_loaded = cancellable_get(gpu_actors[0].get_is_model_loaded.remote())
        if is_loaded:
            print("[Raylight] Workers already have a model loaded. Restarting for fresh slate...")
            needs_restart = True
    except (RayActorError, IndexError, Exception):
        # Actor already dead, crashed, or some other issue
        print("[Raylight] Workers in bad state or crashed. Restarting...")
        needs_restart = True

    if needs_restart:
        for actor in gpu_actors:
            try:
                # Fire and forget kill, then assume they're gone or will be
                actor.kill.remote()
            except Exception:
                pass
        
        # Give Ray a moment to cleanup references before re-spawning
        import time
        time.sleep(1)
        
        # Re-initialize using the factory function (which closure-captures the config)
        ray_actors = ray_actor_fn()
        gpu_actors = ray_actors["workers"]

    # Re-verify parallel_dict from the (potentially new) actors
    parallel_dict = cancellable_get(gpu_actors[0].get_parallel_dict.remote())

    return ray_actors, gpu_actors, parallel_dict
