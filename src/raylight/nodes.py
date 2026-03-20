import raylight
import os
import gc
import math
from typing import Any, cast
from pathlib import Path
from copy import deepcopy

import ray
import torch
import comfy
import folder_paths
import tempfile
import uuid
from yunchang.kernels import AttnType

# Must manually insert comfy package or ray cannot import raylight to cluster
from comfy import sd, sample, utils # type: ignore

from .distributed_worker.ray_worker import (
    make_ray_actor_fn,
    ensure_fresh_actors,
    ray_nccl_tester,
)
from .comfy_dist.utils import cancellable_get, clear_ray_cluster, get_ray_cluster_epoch
from raylight.utils.memory import monitor_memory
from .config import (
    RaylightConfig, 
    ExecutionStrategy, 
    CompactConfig, 
    ClusterConfig, 
    DeviceConfig, 
    DebugConfig, 
    SystemConfig, 
    RaylightAttnType, 
    CompactCompressType
)


try:
    import server
except ImportError:
    server = None

# ---------------------------------------------------------------------------
# Global worker tracking + pinned-cache lifecycle
# ---------------------------------------------------------------------------
# The pinned RAM cache lives in host memory (/dev/shm or private pinned pages)
# and is designed to survive VRAM unloads so that models can be hot-reloaded in
# milliseconds.  Internal Raylight callers (samplers) use the original
# ``_orig_unload_all_models`` to free main-process VRAM without disturbing
# worker caches.
#
# The public ``unload_all_models()`` (monkey-patched below) additionally
# releases worker pinned caches so that ComfyUI's "Unload model" and
# "Free model and node cache" buttons fully reclaim VRAM and pinned RAM.
#
# Pinned caches are also released when:
#   1. Workers are being replaced (model change → ensure_fresh_actors).
#   2. Workers crash (dead-actor detection + /dev/shm cleanup).
# ---------------------------------------------------------------------------
_active_workers: list = []  # Current Ray actor handles (set by loader nodes)


def _register_workers(gpu_actors: list):
    """Store a reference to the current set of Ray worker actors."""
    global _active_workers
    _active_workers = list(gpu_actors)


def release_all_pinned_caches():
    """Tell every live Ray worker to free its pinned RAM cache + VRAM.

    Handles dead/crashed workers gracefully: unreachable actors are pruned
    and orphaned /dev/shm segments are cleaned up.
    """
    global _active_workers
    if not _active_workers:
        # No workers tracked — just clean up any orphaned shm segments.
        _cleanup_stale_shm()
        return

    futures = []
    live_actors = []
    for actor in _active_workers:
        try:
            futures.append(actor.release_pinned_cache.remote())
            live_actors.append(actor)
        except Exception:
            pass  # actor already dead — skip

    if futures:
        try:
            ray.get(futures, timeout=30)
        except Exception as e:
            print(f"[Raylight] Pinned cache release warning: {e}")

    # Prune dead actors from the tracked list
    _active_workers = live_actors

    # Always clean up /dev/shm — dead actors may have left orphaned segments.
    _cleanup_stale_shm()
    print("[Raylight] release_all_pinned_caches: cleanup complete.")


def _cleanup_stale_shm():
    """Remove orphaned /dev/shm segments from previous Raylight sessions.

    Safe to call at startup or after releasing caches — removes ALL
    raylight shm segments that are not actively mmap'd.
    """
    import glob
    for pattern in ("/dev/shm/raylight_pc_*", "/dev/shm/raylight_vae_*"):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass


# Run stale-shm cleanup once at import time (before any workers exist).
_cleanup_stale_shm()

# ---------------------------------------------------------------------------
# Monkey-patch unload_all_models so ComfyUI buttons release worker caches
# ---------------------------------------------------------------------------
# Save the original for internal use (samplers call this to free main-process
# VRAM without touching worker caches — preserving hot-reload).
_orig_unload_all_models = comfy.model_management.unload_all_models


def _patched_unload_all_models():
    """Drop-in replacement that cleans orphaned /dev/shm on full unload.

    ComfyUI calls ``unload_all_models`` from multiple sites:

    * ``main.py`` — "Unload model" / "Free model and node cache" buttons
    * ``execution.py`` — OOM recovery, ``DISABLE_SMART_MEMORY`` end-of-prompt
    * Custom nodes that want a full VRAM flush

    We must NOT call ``release_all_pinned_caches()`` here because the
    pinned RAM cache is designed to survive across prompts — that's the
    whole point of hot-reload.  Pinned caches are released via explicit
    paths instead:

    * ``_teardown_pinned_cache`` — on model switch (``load_unet``)
    * ``release_pinned_cache`` — ``RayWorker.kill()`` / actor shutdown
    * ``release_all_pinned_caches`` — can still be invoked directly by
      custom nodes that need a hard reset.

    Orphaned /dev/shm segments from dead actors are cleaned up below.
    """
    _orig_unload_all_models()
    _cleanup_stale_shm()


comfy.model_management.unload_all_models = _patched_unload_all_models

# Workaround https://github.com/comfyanonymous/ComfyUI/pull/11134
# since in FSDPModelPatcher mode, ray cannot pickle None type cause by getattr
def _monkey():
    from raylight.comfy_dist.supported_models_base import BASE as PatchedBASE
    import comfy.supported_models_base as supported_models_base
    OriginalBASE = supported_models_base.BASE

    if hasattr(PatchedBASE, "__getattr__"):
        setattr(OriginalBASE, "__getattr__", PatchedBASE.__getattr__)


def _resolve_module_dir(module):
    module_file = getattr(module, '__file__', None)
    if module_file:
        path = Path(module_file).resolve()
        if path.is_file():
            return path.parent

    module_paths = getattr(module, '__path__', None)
    if module_paths:
        for path in module_paths:
            if path:
                resolved = Path(path).resolve()
                if resolved.exists():
                    return resolved

    raise RuntimeError(f"Unable to determine module path for {getattr(module, '__name__', module)}")


def _resolve_repo_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'main.py').exists() and (parent / 'execution.py').exists():
            return parent
    raise RuntimeError('Unable to locate ComfyUI repository root')


def _ensure_runtime_workdir(module_dir: Path) -> Path:
    runtime_dir = module_dir.parent / '_ray_runtime_env'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _build_local_runtime_env(module_dir: Path, repo_root: Path, runtime_workdir: Path):
    # For local clusters, avoid py_modules and working_dir entirely.
    # py_modules triggers zipping/uploading the module (~5-10s),
    # working_dir triggers hashing the directory tree (~2-5s).
    # Instead, rely on PYTHONPATH so workers import directly from disk.
    python_path_entries = [str(module_dir.parent), str(repo_root)]
    existing = os.environ.get('PYTHONPATH')
    if existing:
        python_path_entries.extend(part for part in existing.split(os.pathsep) if part)
    python_path = os.pathsep.join(dict.fromkeys(python_path_entries))

    env_vars = {
        'PYTHONPATH': python_path,
        'COMFYUI_BASE_DIRECTORY': str(repo_root),
        'RAY_enable_metrics_collection': '0',
        'RAY_USAGE_STATS_ENABLED': '0',
        'RAY_METRICS_EXPORT_INTERVAL_MS': '0',  # Fully disable metrics export
    }

    return {
        'env_vars': env_vars,
    }


def _build_remote_runtime_env(module_dir: Path, repo_root: Path):
    excludes = [
        '.git',
        '.git/**',
        '__pycache__',
        '**/__pycache__',
        '*.pyc',
    ]

    return {
        'py_modules': [str(module_dir)],
        'working_dir': str(repo_root),
        'env_vars': {
            'COMFYUI_BASE_DIRECTORY': '.',
            'RAY_enable_metrics_collection': '0',
            'RAY_USAGE_STATS_ENABLED': '0',
            'RAY_METRICS_EXPORT_INTERVAL_MS': '0',  # Fully disable metrics export
        },
        'excludes': excludes,
        'config': {'eager_install': False},  # Defer module install until actors spawn
    }


_RAYLIGHT_MODULE_PATH = _resolve_module_dir(raylight)
_COMFY_ROOT_PATH = _resolve_repo_root()
_RAYLIGHT_RUNTIME_WORKDIR = _ensure_runtime_workdir(_RAYLIGHT_MODULE_PATH)

# ---------------------------------------------------------------------------
# CRITICAL: Set PYTHONPATH in os.environ BEFORE ray.init() so that ALL Ray
# worker processes (forked by the Raylet) inherit it at startup and populate
# sys.path correctly.  runtime_env.env_vars only sets os.environ on the
# worker but does NOT update sys.path — Python reads PYTHONPATH once at
# interpreter startup, and Ray reuses long-lived worker processes.
# ---------------------------------------------------------------------------
_python_path_entries = [str(_RAYLIGHT_MODULE_PATH.parent), str(_COMFY_ROOT_PATH)]
_existing_pypath = os.environ.get('PYTHONPATH', '')
if _existing_pypath:
    _python_path_entries.extend(p for p in _existing_pypath.split(os.pathsep) if p)
os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(_python_path_entries))

_RAY_RUNTIME_ENV_LOCAL = _build_local_runtime_env(
    _RAYLIGHT_MODULE_PATH, _COMFY_ROOT_PATH, _RAYLIGHT_RUNTIME_WORKDIR
)
_RAY_RUNTIME_ENV_REMOTE = _build_remote_runtime_env(_RAYLIGHT_MODULE_PATH, _COMFY_ROOT_PATH)
_LOCAL_CLUSTER_ADDRESSES = {None, '', 'local', 'LOCAL'}

# ---------------------------------------------------------------------------
# Pre-set Ray env vars at import time so the head node itself boots faster.
# These must be in the OS environment BEFORE ray.init() is called.
# ---------------------------------------------------------------------------
os.environ.setdefault('RAY_USAGE_STATS_ENABLED', '0')        # skip usage-stats phone-home
os.environ.setdefault('RAY_enable_metrics_collection', '0')   # no Prometheus metrics agent
os.environ.setdefault('RAY_METRICS_EXPORT_INTERVAL_MS', '0')  # fully disable metrics export
os.environ.setdefault('RAY_DEDUP_LOGS', '0')                  # skip log deduplication
os.environ.setdefault('PYTHON_GIL', '1')                      # silence PEP 703 warning


class RayInitializer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {"default": "local"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
                "GPU": ("INT", {"default": 2}),
                "ulysses_degree": ("INT", {"default": 2}),
                "ring_degree": ("INT", {"default": 1}),
                "cfg_degree": ("INT", {"default": 1}),
                "sync_ulysses": ("BOOLEAN", {"default": False}),
                "FSDP": ("BOOLEAN", {"default": False}),
                "FSDP_CPU_OFFLOAD": ("BOOLEAN", {"default": False}),
                "XFuser_attention": (
                    [member.name for member in AttnType],
                    {"default": "TORCH"},
                ),
                "use_kitchen": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable FP8 weight quantization using ComfyUI-Kitchen for reduced VRAM."
                }),
                "attention_backend": (
                    ["STANDARD", "COMPACT"],
                    {"default": "STANDARD"},
                ),
                "ring_impl_type": (
                    ["basic", "zigzag"],
                    {"default": "basic", "tooltip": "Ring attention implementation. 'zigzag' provides load-balancing for causal attention."}
                ),
                "ray_object_store_gb": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Ray shared memory object store size in GB. 0.0 = Auto (Use Ray default ~30% of System RAM). Increase if you see spilling to disk."}),
                "kv_cache_quant_enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable KV cache quantization for CompactFusion."
                }),
                "kv_cache_quant_bits": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 8,
                    "step": 4, # Only allow 4 or 8
                    "tooltip": "Bit-width for KV cache quantization. Supports 4 or 8 bits."
                }),
                "delta_compression": (
                    ["BINARY", "INT2", "INT4", "IDENTITY", "SPARSE"],
                    {"default": "BINARY",
                     "tooltip": "Compression type for delta updates (CompactFusion). BINARY (INT1) is fastest/smallest."}
                ),
                "compact_warmup_steps": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Number of warmup steps before compression starts (CompactFusion). 1 is recommended by paper, but higher values (e.g. 5) are safer."
                }),
            },
            "optional": {
                "gpu_indices": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated list of GPU indices to use (e.g., '0,1'). Overrides automatic selection."
                }),
                "skip_comm_test": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip NCCL communication test at startup. Saves ~10-15s but won't detect comm issues early."
                }),
                "use_mmap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use memory-mapped (zero-copy) model loading. Enabled: parallel loading with OS page cache sharing. Disabled: sequential leader-follower loading (use if mmap causes issues)."
                }),
                "mmap_cache_size": ("INT", {
                    "default": 4, # Increased default for safety
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of models to keep in mmap cache (NOT GB). Zero-copy, so high values are cheap."
                }),
                "vram_limit_gb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.5,
                    "tooltip": "Max VRAM (GB) per GPU for model weights. 0 = auto (use all available). Useful for sharing VRAM with other tasks or forcing CPU offload for large models."
                }),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return get_ray_cluster_epoch()

    def spawn_actor(
        self,
        ray_cluster_address: str,
        ray_cluster_namespace: str,
        GPU: int,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        cfg_degree: int = 1,
        sync_ulysses: bool = False,
        FSDP: bool = False,
        FSDP_CPU_OFFLOAD: bool = False,
        XFuser_attention: str = "TORCH",
        use_kitchen: bool = False,
        attention_backend: str = "STANDARD",
        ray_object_store_gb: float = 0.0,
        ray_dashboard_address: str = "None",
        torch_dist_address: str = "None",
        gpu_indices: str = "",
        skip_comm_test: bool = True,
        use_mmap: bool = True,
        mmap_cache_size: int = 4,
        kv_cache_quant_enable: bool = False,
        kv_cache_quant_bits: int = 8,
        delta_compression: str = "BINARY",
        compact_warmup_steps: int = 1,
        ring_impl_type: str = "basic",
        vram_limit_gb: float = 0.0,
    ):
        with monitor_memory("RayInitializer.spawn_actor"):
            # THIS IS PYTORCH DIST ADDRESS
            # (TODO) Change so it can be use in cluster of nodes. but it is long waaaaay down in the priority list
            # os.environ['TORCH_CUDA_ARCH_LIST'] = ""
            # Map string inputs to Enums
            # [SENIOR REFACTOR] Parse distributed sync addresses
            master_addr, master_port = "127.0.0.1", "29500"
            if torch_dist_address and ":" in torch_dist_address:
                master_addr, master_port = torch_dist_address.split(":", 1)
            
            dashboard_host, dashboard_port = "127.0.0.1", None
            if ray_dashboard_address != "None" and ":" in ray_dashboard_address:
                try:
                    _host, _port = ray_dashboard_address.rsplit(":", 1)
                    dashboard_host = _host
                    dashboard_port = int(_port)
                except ValueError:
                    pass

            # Map string inputs to Enums
            try:
                attn_type_enum = RaylightAttnType[XFuser_attention]
            except KeyError:
                attn_type_enum = RaylightAttnType.TORCH
                
            try:
                comp_type_enum = CompactCompressType(delta_compression)
            except ValueError:
                comp_type_enum = CompactCompressType.BINARY

            # [SENIOR REFACTOR] Initialize the Unified Configuration Engine
            config = RaylightConfig(
                strategy=ExecutionStrategy(
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                    cfg_degree=cfg_degree,
                    sync_ulysses=sync_ulysses,
                    fsdp_enabled=FSDP,
                    fsdp_cpu_offload=FSDP_CPU_OFFLOAD,
                    attention_backend=attention_backend,
                    attention_type=attn_type_enum,
                    ring_impl=ring_impl_type,
                ),
                compact=CompactConfig(
                    enabled=delta_compression != "IDENTITY",
                    warmup_steps=compact_warmup_steps,
                    delta_compression=comp_type_enum,
                    kv_cache_quant_enable=kv_cache_quant_enable,
                    kv_cache_quant_bits=kv_cache_quant_bits,
                ),
                cluster=ClusterConfig(
                    address=ray_cluster_address,
                    namespace=ray_cluster_namespace,
                    object_store_gb=ray_object_store_gb,
                    dashboard_host=dashboard_host,
                    dashboard_port=dashboard_port,
                ),
                device=DeviceConfig(
                    world_size=GPU,
                    gpu_indices=[int(x.strip()) for x in gpu_indices.split(",") if x.strip()],
                    vram_limit_gb=vram_limit_gb,
                    use_mmap=use_mmap,
                    mmap_cache_size=mmap_cache_size,
                ),
                debug=DebugConfig(
                    skip_comm_test=skip_comm_test,
                    enable_kitchen=use_kitchen,
                ),
                system=SystemConfig(
                    master_addr=master_addr,
                    master_port=master_port,
                )
            )

            world_size = config.device.world_size

            # Use legacy dict for backward compatibility with downstream Samplers/Loaders
            self.parallel_dict = config.to_legacy_dict()
            
            # The config.__post_init__ handles validation for ulysses * ring * cfg vs world_size
            if cfg_degree > 2:
                raise ValueError("CFG batch only can be divided into 2 degree of parallelism")

            _monkey()
            
            # [SENIOR REFACTOR] Apply system-level environment settings via Unified Config
            config.apply_to_env()

            if config.cluster.object_store_gb <= 0:
                print("[Raylight] object_store_memory set to Auto (Ray default).")
            else:
                print(f"[Raylight] object_store_memory set to {config.cluster.object_store_gb} GB.")

            runtime_env_base = _RAY_RUNTIME_ENV_LOCAL
            if ray_cluster_address not in _LOCAL_CLUSTER_ADDRESSES:
                runtime_env_base = _RAY_RUNTIME_ENV_REMOTE

            # GPU Pinning Logic
            original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                if gpu_indices.strip():
                    # Validate and set
                    indices = [x.strip() for x in gpu_indices.split(",") if x.strip()]
                    if len(indices) < world_size:
                         raise ValueError(f"gpu_indices contains {len(indices)} GPUs, but {world_size} (GPU input) were requested.")
                    
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
                    print(f"[Raylight] Pinning Ray Cluster to GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

                # ===== OPTIMIZATION: Cluster Reuse =====
                # Check if Ray is already initialized with matching config to skip expensive re-init
                is_local = ray_cluster_address in _LOCAL_CLUSTER_ADDRESSES
                should_reuse = False
                
                if ray.is_initialized():
                    try:
                        # Check if the cluster has matching GPU count
                        existing_resources = ray.cluster_resources()
                        existing_gpus = int(existing_resources.get('GPU', 0))
                        if existing_gpus >= world_size:
                            print(f"[Raylight] Reusing existing Ray cluster (GPUs available: {existing_gpus})")
                            should_reuse = True
                    except Exception:
                        pass
                
                if not should_reuse:
                    # Shut down so if comfy user try another workflow it will not cause error
                    ray.shutdown()
                    
                    # Build init kwargs - disable metrics agent for faster startup
                    init_kwargs = {
                        'namespace': config.cluster.namespace,
                        'runtime_env': deepcopy(runtime_env_base),
                        'include_dashboard': config.cluster.dashboard_port is not None,
                        'dashboard_host': config.cluster.dashboard_host,
                        'dashboard_port': config.cluster.dashboard_port,
                        '_metrics_export_port': None,  # Disable metrics agent to avoid connection retries
                        'configure_logging': False,     # Skip Ray's logging setup
                        'logging_level': 'warning',     # Reduce log processing overhead
                    }
                    
                    # Only set object_store_memory if explicitly configured (not for reused clusters)
                    if config.cluster.object_store_gb > 0:
                        init_kwargs['object_store_memory'] = int(config.cluster.object_store_gb * 1024**3)
                    
                    ray.init(config.cluster.address, **init_kwargs)
                    print(f"[Raylight] Ray cluster initialized (new instance)")
                
            except Exception as e:
                ray.shutdown()
                ray.init(
                    runtime_env=deepcopy(runtime_env_base),
                    _metrics_export_port=None,
                )
                raise RuntimeError(f"Ray connection failed: {e}")
            # Restore original environment to avoid affecting other nodes
            if original_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
            elif "CUDA_VISIBLE_DEVICES" in os.environ and gpu_indices.strip():
                 del os.environ["CUDA_VISIBLE_DEVICES"]

            # ===== OPTIMIZATION: Skip NCCL Test =====
            # NCCL test spawns/kills separate actors before real workers - saves ~10-15s
            if not skip_comm_test:
                print("[Raylight] Running NCCL communication test...")
                ray_nccl_tester(world_size)
            else:
                print("[Raylight] Skipping NCCL test (skip_comm_test=True)")
            
            ray_actor_fn = make_ray_actor_fn(world_size, self.parallel_dict, raylight_config=config)
            ray_actors = ray_actor_fn()
            
            # Store GPU indices for later use in samplers (for partial offload matching)
            if gpu_indices.strip():
                ray_actors["gpu_indices"] = [int(x.strip()) for x in gpu_indices.split(",") if x.strip()]
            else:
                # Default: 0, 1, 2, ...
                ray_actors["gpu_indices"] = list(range(world_size))

            # Track workers for the unload hook (will be updated by loader nodes)
            _register_workers(ray_actors["workers"])

            return ([ray_actors, ray_actor_fn],)


class RayInitializerAdvanced(RayInitializer):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {
                    "default": "local",
                    "tooltip": "Address of Ray cluster different than torch distributed address"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
                "ray_object_store_gb": ("FLOAT", {
                    "default": 0.0,
                    "tooltip": "Ray shared memory object store size in GB. 0.0 = Auto (Use Ray default ~30% of System RAM). Increase if you see spilling to disk."}),
                "ray_dashboard_address": ("STRING", {
                    "default": "None",
                    "tooltip": "Same format as torch_dist_address, you need to install ray dashboard to monitor"}),
                "torch_dist_address": ("STRING", {
                    "default": "127.0.0.1:29500",
                    "tooltip": "Might need to restart ComfyUI to apply"}),
                "GPU": ("INT", {"default": 2}),
                "ulysses_degree": ("INT", {"default": 2}),
                "ring_degree": ("INT", {"default": 1}),
                "cfg_degree": ("INT", {"default": 1}),
                "sync_ulysses": ("BOOLEAN", {"default": False}),
                "FSDP": ("BOOLEAN", {"default": False}),
                "FSDP_CPU_OFFLOAD": ("BOOLEAN", {"default": False}),
                "XFuser_attention": (
                    [
                        "TORCH",
                        "FLASH_ATTN",
                        "FLASH_ATTN_3",
                        "SAGE_AUTO_DETECT",
                        "SAGE_FP16_TRITON",
                        "SAGE_FP16_CUDA",
                        "SAGE_FP8_CUDA",
                        "SAGE_FP8_SM90",
                        "AITER_ROCM",
                    ],
                    {"default": "TORCH"},
                ),
                "use_kitchen": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable FP8 weight quantization using ComfyUI-Kitchen for reduced VRAM."
                }),
                "kv_cache_quant_enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable KV cache quantization for CompactFusion."
                }),
                "kv_cache_quant_bits": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 8,
                    "step": 4,
                    "tooltip": "Bit-width for KV cache quantization. Supports 4 or 8 bits."
                }),
                "delta_compression": (
                    ["BINARY", "INT2", "INT4", "IDENTITY", "SPARSE"],
                    {"default": "BINARY",
                     "tooltip": "Compression type for delta updates (CompactFusion). BINARY (INT1) is fastest/smallest."}
                ),
                "ring_impl_type": (
                    ["basic", "zigzag"],
                    {"default": "basic", "tooltip": "Ring attention implementation. 'zigzag' provides load-balancing for causal attention."}
                ),
            },
            "optional": {
                "gpu_indices": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated list of GPU indices to use (e.g., '0,1'). Overrides automatic selection."
                }),
                "skip_comm_test": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip NCCL communication test at startup. Saves ~10-15s but won't detect comm issues early."
                }),
                "use_mmap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use memory-mapped (zero-copy) model loading. Enabled: parallel loading with OS page cache sharing. Disabled: sequential leader-follower loading."
                }),
                "mmap_cache_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "LRU cache size for mmap'd model state dicts."
                }),
                "vram_limit_gb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.5,
                    "tooltip": "Max VRAM (GB) per GPU for model weights. 0 = auto (use all available). Useful for sharing VRAM with other tasks or forcing CPU offload for large models."
                }),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS_INIT",)
    RETURN_NAMES = ("ray_actors_init",)

    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"


# ---------------------------------------------------------------------------
# INT8 architecture exclusion lists (shared with standalone INT8 loader)
# ---------------------------------------------------------------------------
# Known transformer block attribute names across architectures.
_TRANSFORMER_BLOCK_NAMES = [
    "double_blocks", "single_blocks",   # Flux / HunyuanDiT
    "layers",                            # Lumina / Z-Image
    "transformer_blocks",                # SD / SDXL
    "blocks",                            # generic
    "visual_transformer_blocks",         # Qwen
    "text_transformer_blocks",           # Qwen
]


def _discover_compile_keys(diffusion_model) -> list[str]:
    """Find per-block compile keys for a diffusion model.

    Compiling individual transformer blocks avoids tracing the outer forward
    (which may contain concrete-int ops that crash Inductor, e.g. Lumina
    pos_ids_x) while still getting Triton speedups on the heavy matmul path.
    Falls back to ["diffusion_model"] if no known block attributes are found.
    """
    keys: list[str] = []
    for layer_name in _TRANSFORMER_BLOCK_NAMES:
        if hasattr(diffusion_model, layer_name):
            blocks = getattr(diffusion_model, layer_name)
            for i in range(len(blocks)):
                keys.append(f"diffusion_model.{layer_name}.{i}")
    return keys if keys else ["diffusion_model"]


def _check_compile_compatible(unet_name: str, config: RaylightConfig, backend: str = "inductor") -> tuple[bool, str]:
    if backend == "disabled":
        return True, ""
        
    if config.device.vram_limit_gb > 0:
        return False, "torch.compile is incompatible with VRAM limits (LowVram mode)"
    
    return True, ""


def _guess_int8_exclusions(unet_name: str) -> list:
    """Best-effort architecture detection from filename for INT8 layer exclusions."""
    try:
        from raylight.comfy_extra_dist.int8.nodes_int8_loader import (
            _MODEL_EXCLUSIONS, _guess_exclusions,
        )
        return _guess_exclusions(unet_name)
    except ImportError:
        return []


class RayUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models")
                              + folder_paths.get_filename_list("checkpoints"),),
                "weight_dtype": (
                    [
                        "default",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                        "bf16",
                        "fp16",
                        "int8",
                        "int8_dynamic",
                    ],
                ),
                "ray_actors_init": (
                    "RAY_ACTORS_INIT",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "torch_compile": (
                    ["disabled", "inductor", "inductor_reduce_overhead"],
                    {"default": "disabled", "tooltip": "Apply torch.compile to transformer blocks for faster inference. 'inductor' uses Triton codegen; 'inductor_reduce_overhead' adds CUDA-graph wrapping around compiled kernels. First run will be slow due to compilation warmup."},
                ),
                "torch_compile_dynamic": (
                    ["auto", "true", "false"],
                    {"default": "auto", "tooltip": "Dynamic shape tracing. 'auto': starts static, switches to dynamic on shape change. 'true': always symbolic shapes (avoids recompiles but slightly slower kernels). 'false': always static (fastest kernels, recompiles on any shape change)."},
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_ray_unet"

    CATEGORY = "Raylight"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return get_ray_cluster_epoch()

    def load_ray_unet(self, ray_actors_init, unet_name, weight_dtype, torch_compile="disabled", torch_compile_dynamic="auto"):
        with monitor_memory("RayUNETLoader.load_ray_unet"):
            ray_actors, gpu_actors, config = ensure_fresh_actors(ray_actors_init)

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif weight_dtype in ("int8", "int8_dynamic"):
            model_options["int8_quantize"] = True
            model_options["int8_dynamic"] = (weight_dtype == "int8_dynamic")
            # Auto-detect architecture exclusions from filename
            model_options["int8_excluded_names"] = _guess_int8_exclusions(unet_name)

        try:
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        except:
            unet_path = folder_paths.get_full_path_or_raise("checkpoints", unet_name)


        loaded_futures = []
        patched_futures = []

        # Check for Kitchen FP8 Quantization which is memory intensive (RAM spike during conversion)
        use_kitchen = config.debug.enable_kitchen
        
        if use_kitchen:
            # Reverting sequential loading.
            # With streaming quantization in fsdp.py, the peak RAM usage is per-block
            # (e.g. 500MB) instead of full-model (24GB).
            # Parallel loading (4x 500MB) is perfectly safe and much faster.
            pass

        # PARALLEL LOADING (Fast)
        # Uses ModelContext with zero-copy mmap for optimal speed
        print(f"[Raylight] Parallel Load ({'FSDP' if config.strategy.fsdp_enabled else 'Standard'}): Creating lazy wrappers...")
        for actor in gpu_actors:
            loaded_futures.append(
                actor.load_unet.remote(unet_path, model_options=model_options)
            )
        cancellable_get(loaded_futures)
        loaded_futures = []

        # Patching: Done in small batches to limit peak RAM while still
        # overlapping work across independent GPU workers.  Each actor runs
        # in its own process so memory pressure is per-worker, but issuing
        # all at once can spike the Ray object store.  Batches of 2 keep
        # wall-clock time ~halved vs fully sequential for 4+ GPUs.
        _PATCH_BATCH = 2
        
        # Access USP/CFG degrees safely from Strategy
        strategy = config.strategy
        if config.meta.total_sp_degree > 1:
            if strategy.ulysses_degree > 1 or strategy.ring_degree > 1:
                print(f"[Raylight] Batched USP Patching ({_PATCH_BATCH} workers at a time)...")
                for i in range(0, len(gpu_actors), _PATCH_BATCH):
                    batch = gpu_actors[i:i + _PATCH_BATCH]
                    cancellable_get([a.patch_usp.remote() for a in batch])
            if strategy.cfg_degree > 1:
                print(f"[Raylight] Batched CFG Patching ({_PATCH_BATCH} workers at a time)...")
                for i in range(0, len(gpu_actors), _PATCH_BATCH):
                    batch = gpu_actors[i:i + _PATCH_BATCH]
                    cancellable_get([a.patch_cfg.remote() for a in batch])

        # Optional torch.compile
        if torch_compile != "disabled":
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
                    keys = _discover_compile_keys(diffusion_model)
                    set_torch_compile_wrapper(model=m, backend="inductor", mode=mode, dynamic=dynamic, keys=keys)
                    return m
                compile_futures = [actor.model_function_runner.remote(_apply_compile, compile_mode, compile_dynamic) for actor in gpu_actors]
                cancellable_get(compile_futures)
                parts = ["transformer blocks"]
                if compile_mode: parts.append(f"mode={compile_mode}")
                if torch_compile_dynamic != "auto": parts.append(f"dynamic={torch_compile_dynamic}")
                print(f"[Raylight] torch.compile (inductor, {', '.join(parts)}) applied to all workers")

        # Track workers globally so the unload hook can reach them
        _register_workers(gpu_actors)

        return (ray_actors,)





class XFuserKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
        sigmas=None,
    ):
        try:
            with monitor_memory("XFuserKSampler.ray_sample"):
                # Clean VRAM for preparation to load model
                pass
            force_full_denoise = True
            if return_with_leftover_noise == "enable":
                force_full_denoise = False
            disable_noise = False
            if add_noise == "disable":
                disable_noise = True

            gpu_actors = ray_actors["workers"]

            # Re-apply LoRAs for this branch's config_hash if needed (Always call to handle None/reset)
            lora_config_hash = ray_actors.get("lora_config_hash")
            print(f"[XFuserKSamplerAdvanced] Syncing LoRA state (config_hash={lora_config_hash})...")
            lora_futures = [actor.reapply_loras_for_config.remote(lora_config_hash) for actor in gpu_actors]
            cancellable_get(lora_futures)

            futures = [
                actor.common_ksampler.remote(
                    noise_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_image,
                    denoise=denoise,
                    disable_noise=disable_noise,
                    start_step=start_at_step,
                    last_step=end_at_step,
                    force_full_denoise=force_full_denoise,
                    sigmas=sigmas,
                )
                for actor in gpu_actors
            ]

            results: Any = cancellable_get(futures)
            if results is None:
                 raise RuntimeError("Raylight: Sampling failed or was cancelled.")
            return (results[0][0],)
        except Exception as e:
            clear_ray_cluster(ray_actors, reason=f"sampling error in XFuserKSamplerAdvanced: {type(e).__name__}")
            raise


class DPKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "ray_actors": (
                    "RAY_ACTORS",
                    {"tooltip": "Ray Actor to submit the model into"},
                ),
                "noise_list": ("NOISE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "ray_sample"

    CATEGORY = "Raylight"

    def ray_sample(
        self,
        ray_actors,
        add_noise,
        noise_list,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        denoise=1.0,
    ):
        ray_actors = ray_actors[0]
        try:
            add_noise = add_noise[0]
            steps = steps[0]
            cfg = cfg[0]
            sampler_name = sampler_name[0]
            scheduler = scheduler[0]
            positive = positive[0]
            negative = negative[0]
            start_at_step = start_at_step[0]
            end_at_step = end_at_step[0]
            return_with_leftover_noise = return_with_leftover_noise[0]

            gpu_actors = ray_actors["workers"]

            config = cast(RaylightConfig, cancellable_get(gpu_actors[0].get_raylight_config.remote()))
            if config is not None and config.meta.total_sp_degree > 1:
                raise ValueError(
                    """
                Data Parallel KSampler only supports FSDP or standard Data Parallel (DP).
                Please set both 'ulysses_degree' and 'ring_degree' to 0,
                or use the XFuser KSampler instead. More info on Raylight mode https://github.com/komikndr/raylight
                """
                )

            if len(latent_image) != len(gpu_actors):
                latent_image = [latent_image[0]] * len(gpu_actors)

            # Clean VRAM for preparation to load model
            force_full_denoise = True
            if return_with_leftover_noise == "enable":
                force_full_denoise = False
            disable_noise = False
            if add_noise == "disable":
                disable_noise = True

            # Re-apply LoRAs for this branch's config_hash if needed (Always call to handle None/reset)
            lora_config_hash = ray_actors.get("lora_config_hash")
            print(f"[DPKSamplerAdvanced] Syncing LoRA state (config_hash={lora_config_hash})...")
            lora_futures = [actor.reapply_loras_for_config.remote(lora_config_hash) for actor in gpu_actors]
            cancellable_get(lora_futures)

            futures = [
                actor.common_ksampler.remote(
                    noise_list[i],
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_image[i],
                    denoise=denoise,
                    disable_noise=disable_noise,
                    start_step=start_at_step,
                    last_step=end_at_step,
                    force_full_denoise=force_full_denoise,
                )
                for i, actor in enumerate(gpu_actors)
            ]

            results: Any = cancellable_get(futures)
            if results is None:
                 raise RuntimeError("Raylight: Sampling failed or was cancelled.")
            
            if config is not None and config.meta.total_sp_degree > 1:
                 return (results[0][0],)

            # Standard mode: concat shards
            out: Any = latent_image[0]
            out = out.copy()
            samples = []
            for res in results:
                 if res is not None:
                      samples.append(res[0]["samples"])
        
            out["samples"] = torch.cat(samples, dim=0)
            return (out,)
        except Exception as e:
            clear_ray_cluster(ray_actors, reason=f"sampling error in DPKSamplerAdvanced: {type(e).__name__}")
            raise


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class DPNoiseList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **{
                    f"noise_seed_{i}": (
                        "INT",
                        {
                            "default": 0,
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                            "control_after_generate": True,
                        },
                    )
                    for i in range(8)
                }
            }
        }

    RETURN_TYPES = ("NOISE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "get_noise"
    CATEGORY = "Raylight"

    def get_noise(self, **kwargs):
        noise_list = []
        for key, seed in kwargs.items():
            if key.startswith("noise_seed_"):
                noise_list.append(seed)
        return (noise_list,)


class RayVAEDecodeDistributed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray Actor to submit the model into"}),
                "samples": ("LATENT",),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "vae_dtype": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "VAE precision: auto=bf16 on RTX3000+/fp32 fallback, bf16=bfloat16 (recommended), fp16=half (may cause NaN), fp32=full (stable but 2x memory)"
                }),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32},),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only used for video VAEs: Amount of frames to decode at a time.",
                    },
                ),
                "temporal_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 4,
                        "max": 4096,
                        "step": 4,
                        "tooltip": "Only used for video VAEs: Amount of frames to overlap.",
                    },
                ),
                "release_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Release VAE from worker RAM after decode. Recommended True to free ~2GB per worker. Set False if you need the VAE for multiple decodes."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ray_decode"

    CATEGORY = "Raylight"

    def ray_decode(self, ray_actors, vae_name, samples, tile_size, vae_dtype="auto", overlap=64, temporal_size=64, temporal_overlap=8, release_vae=True):
        gpu_actors = ray_actors["workers"]
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

        # === PIPELINED VAE DECODE ===
        # Fire VAE load on all workers (returns compression factors)
        load_futures = [actor.ray_vae_loader.remote(vae_path) for actor in gpu_actors]

        # While VAE loads on GPU workers, prepare sharding info on CPU (no blocking wait)
        latents = samples["samples"]
        num_workers = len(gpu_actors)
        total_frames = latents.shape[2]

        # Core ComfyUI VAEDecodeTiled safety checks
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2

        # Put entire latent tensor into Ray object store ONCE (zero-copy shared memory)
        # Workers will slice their own portion — avoids N CPU clones on master
        latent_ref = ray.put(latents)

        # NOW wait for VAE load + compression factors (load was running in parallel)
        load_results = cancellable_get(load_futures)
        temporal_compression, spatial_compression = load_results[0]
        temporal_compression = temporal_compression or 1
        spatial_compression = spatial_compression or 1

        # Causal VAE output formula: (Latent_T - 1) * compression + 1
        if isinstance(temporal_compression, (int, float)) and temporal_compression > 1:
            total_output_frames = (total_frames - 1) * temporal_compression + 1
        else:
            total_output_frames = total_frames
            
        # Calculate Master Output Shape (B, T, H, W, C)
        # We assume Batch=1 for video usually, but we handle standard (B, T, H, W, C) structure 
        # or (T, H, W, C) if squeezed. ComfyUI usually expects (TotalFrames, H, W, 3) for video batch.
        H_out = latents.shape[3] * spatial_compression
        W_out = latents.shape[4] * spatial_compression
        # Final shape: (Batch, TotalFrames, Height, Width, 3)
        master_shape = (latents.shape[0], total_output_frames, H_out, W_out, 3)
        
        # 3. Create Shared Memory File (Pre-allocation)
        mmap_path = f"/dev/shm/raylight_vae_out_{uuid.uuid4().hex}.bin"
        num_elements = 1
        for dim in master_shape: num_elements *= dim
        file_size_bytes = num_elements * 4 # float32 = 4 bytes
        
        print(f"[RayVAEDecode] Pre-allocating shared output buffer: {mmap_path} ({file_size_bytes/1024**3:.2f} GB)")
        print(f"[RayVAEDecode] Output Shape: {master_shape}")
        
        full_image = None # Initialize outside try block for finally access
        try:
            with open(mmap_path, "wb") as f:
                f.seek(file_size_bytes - 1)
                f.write(b"\0")
                
            # Create the tensor wrapper immediately
            full_image = torch.from_file(mmap_path, shared=True, size=num_elements, dtype=torch.float32).reshape(master_shape)
            
            # === DYNAMIC WORK-STEALING VAE DISPATCH ===
            # Instead of 1 big static shard per worker, split into many small
            # micro-batches. Workers pull the next chunk when they finish,
            # so faster GPUs naturally process more chunks and no one idles.
            
            # Chunk size heuristic: aim for ~2-3x more chunks than workers
            # to allow load balancing, but not so small that overhead dominates.
            # Minimum 2 latent frames per chunk to keep context overhead low.
            target_chunks = num_workers * 3
            frames_per_chunk = max(2, total_frames // target_chunks)
            # For very short videos, fall back to 1 chunk per worker
            if total_frames <= num_workers:
                frames_per_chunk = 1
            
            # Build ordered chunk descriptors
            chunks = []  # list of (chunk_id, context_start, actual_end, discard_latent_frames, start_video_frame)
            chunk_id = 0
            pos = 0
            while pos < total_frames:
                end = min(pos + frames_per_chunk, total_frames)
                context_start = max(0, pos - 1)
                # Extend by 1 for continuity into next chunk (unless last chunk)
                actual_end = min(end + (1 if end < total_frames else 0), total_frames)
                discard_latent_frames = pos - context_start
                
                if isinstance(temporal_compression, (int, float)) and temporal_compression > 1:
                    start_video_frame = pos * temporal_compression
                else:
                    start_video_frame = pos
                
                chunks.append((chunk_id, context_start, actual_end, discard_latent_frames, start_video_frame))
                chunk_id += 1
                pos = end
            
            print(f"[RayVAEDecode] Dynamic dispatch: {len(chunks)} chunks of ~{frames_per_chunk} latent frames across {num_workers} workers")
            
            # Helper to dispatch a chunk to an actor
            def _dispatch_chunk(actor, chunk):
                cid, ctx_start, ctx_end, discard, vid_offset = chunk
                print(f"[RayVAEDecode] Chunk {cid}: Latents {ctx_start}:{ctx_end} (discard {discard}) -> Video Frame {vid_offset}")
                return actor.ray_vae_decode.remote(
                    cid,
                    {},
                    tile_size,
                    overlap=overlap,
                    temporal_size=temporal_size,
                    temporal_overlap=temporal_overlap,
                    discard_latent_frames=discard,
                    vae_dtype=vae_dtype,
                    mmap_path=mmap_path,
                    mmap_shape=master_shape,
                    output_offset=vid_offset,
                    latent_ref=latent_ref,
                    latent_slice_start=ctx_start,
                    latent_slice_end=ctx_end,
                    skip_offload=True,  # Keep VAE on GPU for work-stealing
                )
            
            # Seed the queue: dispatch first batch (1 chunk per worker)
            chunk_queue = list(chunks)  # remaining chunks to dispatch
            inflight = {}  # ray_ref -> (actor, chunk_descriptor)
            
            initial_batch = min(num_workers, len(chunk_queue))
            for i in range(initial_batch):
                chunk = chunk_queue.pop(0)
                ref = _dispatch_chunk(gpu_actors[i], chunk)
                inflight[ref] = (gpu_actors[i], chunk)
            
            print(f"[RayVAEDecode] Seeded {initial_batch} workers, {len(chunk_queue)} chunks queued. Work-stealing enabled.")
            
            # Dynamic gather + re-dispatch loop
            received_count = 0
            
            while inflight:
                # Check ComfyUI cancel status
                if comfy.model_management.processing_interrupted():
                    print("[Raylight] Cancellation detected during VAE decoding! Force-canceling Ray tasks...")
                    for ref in inflight:
                        try:
                            ray.cancel(ref, force=True, recursive=True)
                        except:
                            pass
                    raise Exception("Raylight: VAE Decode canceled by user.")

                ready, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=1.0)
                if not ready:
                    continue
                    
                for ray_ref in ready:
                    actor, completed_chunk = inflight.pop(ray_ref)
                    res_data: Any = cancellable_get(ray_ref)
                    
                    if res_data is not None:
                        shard_index, result = res_data
                        received_count += 1
                        
                        if isinstance(result, dict) and result.get("mmap", False):
                            shape = result["shape"]
                            print(f"[RayVAEDecode] Chunk {shard_index} wrote {shape} to mmap.")
                        else:
                            # Fallback: tensor returned directly
                            shard_data: Any = result
                            print(f"[RayVAEDecode] Chunk {shard_index} returned tensor {shard_data.shape} (Fallback path)")
                            _, _, _, _, vid_offset = completed_chunk
                            s_len = shard_data.shape[1]
                            if vid_offset + s_len > total_output_frames:
                                s_len = max(0, total_output_frames - vid_offset)
                                shard_data = shard_data[:, :s_len]
                            if s_len > 0:
                                shard_data_any: Any = shard_data
                                full_image_any: Any = full_image
                                full_image_any[:, vid_offset : vid_offset + s_len] = shard_data_any.to(torch.float32)
                            del shard_data
                    
                    # WORK STEALING: if chunks remain, immediately send next to this now-idle worker
                    if chunk_queue:
                        next_chunk = chunk_queue.pop(0)
                        ref = _dispatch_chunk(actor, next_chunk)
                        inflight[ref] = (actor, next_chunk)
            
            print(f"[RayVAEDecode] All {received_count} chunks complete.")
            
            # Offload VAE from VRAM on all workers (skip_offload=True kept it on GPU)
            print("[RayVAEDecode] Offloading VAE from VRAM on all workers...")
            offload_futures = [actor.ray_vae_offload.remote() for actor in gpu_actors]
            cancellable_get(offload_futures)
            
            # Release the shared latent tensor from Ray object store
            del latent_ref

            try:
                os.unlink(mmap_path)
                print(f"[RayVAEDecode] Unlinked temp mmap file: {mmap_path}")
            except Exception as e:
                print(f"[RayVAEDecode] Warning: Could not unlink mmap file: {e}")

            # 7. Release VAE from workers to free RAM for downstream operations (e.g., AudioVAE)
            if release_vae:
                print("[RayVAEDecode] Releasing VAE from workers to free RAM...")
                release_futures = [actor.ray_vae_release.remote() for actor in gpu_actors]
                cancellable_get(release_futures)
                print("[RayVAEDecode] VAE released from all workers.")

            # 8. Return result
            # ComfyUI's downstream video nodes expect (Time, H, W, 3) 
            # while our internal master_shape is (Batch, Time, H, W, 3).
            # Squeeze batch dimension if Batch=1 for compatibility.
            if full_image.shape[0] == 1:
                return (full_image.squeeze(0),)
            else:
                # If Batch > 1, we return 5D. 
                # Note: Most ComfyUI image nodes expect (Batch, H, W, 3).
                # For video Batch > 1, standard nodes might fail anyway without flattening.
                return (full_image,)
            
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(mmap_path):
                try: 
                    os.unlink(mmap_path) 
                except: pass
            raise e


class RayOffloadModel:
    """
    Offloads the diffusion model from all Ray workers' VRAM.
    Place this node after the sampler to free GPU memory.
    
    This is an OUTPUT node - it will always execute if connected.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Passthrough for workflow chaining"}),
            }
        }

    # Return the latent passthrough for chaining
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "offload"
    CATEGORY = "Raylight"
    OUTPUT_NODE = True  # Marks this as an output node - always executes

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Return NaN to force re-execution on every run.
        # NaN != NaN, so this node is always considered "changed"
        return float("nan")

    def offload(self, ray_actors, latent=None):
        import sys
        print("[RayOffloadModel] ========== OFFLOAD NODE EXECUTING ==========", flush=True)
        sys.stdout.flush()
        
        gpu_actors = ray_actors["workers"]
        
        # Offload from all workers IN PARALLEL
        print(f"[RayOffloadModel] Starting PARALLEL offload for {len(gpu_actors)} workers...", flush=True)
        
        futures = []
        for i, actor in enumerate(gpu_actors):
            print(f"[RayOffloadModel] Triggering offload for worker {i}...", flush=True)
            futures.append(actor.offload_and_clear.remote())
            
        try:
            # Wait for all to complete
            cancellable_get(futures)
            print("[RayOffloadModel] All workers offloaded.", flush=True)
        except Exception as e:
            print(f"[RayOffloadModel] Error during parallel offload: {e}", flush=True)
            
        # Free local memory handles immediately
        gc.collect()

        print("[RayOffloadModel] ========== ALL WORKERS OFFLOADED ==========", flush=True)
        return (latent,)



class RayLoraLoaderModelOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "load_lora_model_only"

    CATEGORY = "Raylight"

    def load_lora_model_only(self, ray_actors, lora_name, strength_model):
        
        # 1. Extract existing chain from ray_actors
        # Default to empty string if this is the first lora in the chain
        current_chain = ray_actors.get("lora_chain", "")
        
        # 2. Append new LoRA to the chain
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        this_lora_sig = f"{lora_name}:{strength_model}"
        
        if current_chain:
            new_chain = current_chain + "|" + this_lora_sig
        else:
            new_chain = this_lora_sig
            
        # 3. Calculate new hash
        lora_config_hash = hash(new_chain)
        
        if strength_model == 0:
            return (ray_actors,)

        gpu_actors = ray_actors["workers"]

        # 4. Dispatch UNET patching to Ray Workers with config hash for branch isolation
        print(f"[RayLoraLoaderModelOnly] Dispatching LoRA {lora_name} to {len(gpu_actors)} workers (config_hash={lora_config_hash})...")
        futures = [actor.load_lora.remote(lora_path, strength_model, lora_config_hash) for actor in gpu_actors]
        cancellable_get(futures)

        # 5. Update ray_actors with new chain and hash
        updated_ray_actors = {
            **ray_actors,
            "lora_config_hash": lora_config_hash,
            "lora_chain": new_chain,
        }
        return (updated_ray_actors,)


NODE_CLASS_MAPPINGS = {
    "XFuserKSamplerAdvanced": XFuserKSamplerAdvanced,
    "DPKSamplerAdvanced": DPKSamplerAdvanced,
    "RayUNETLoader": RayUNETLoader,
    "RayLoraLoaderModelOnly": RayLoraLoaderModelOnly,

    "RayInitializer": RayInitializer,
    "RayInitializerAdvanced": RayInitializerAdvanced,
    "DPNoiseList": DPNoiseList,
    "RayVAEDecodeDistributed": RayVAEDecodeDistributed,
    "RayOffloadModel": RayOffloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserKSamplerAdvanced": "XFuser KSampler (Advanced)",
    "DPKSamplerAdvanced": "Data Parallel KSampler (Advanced)",
    "RayUNETLoader": "Load Diffusion Model (Ray)",
    "RayLoraLoaderModelOnly": "Ray LoRA Loader",

    "RayInitializer": "Ray Init Actor",
    "RayInitializerAdvanced": "Ray Init Actor (Advanced)",
    "DPNoiseList": "Data Parallel Noise List",
    "RayVAEDecodeDistributed": "Distributed VAE (Ray)",
    "RayOffloadModel": "Offload Model (Ray)",
}




