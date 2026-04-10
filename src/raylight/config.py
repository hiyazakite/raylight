import os
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Callable, Dict, Any, Tuple
import torch

class RaylightAttnType(Enum):
    TORCH = "TORCH"
    FLASH_ATTN = "FLASH_ATTN"
    FLASH_ATTN_3 = "FLASH_ATTN_3"
    SAGE_AUTO_DETECT = "SAGE_AUTO_DETECT"
    SAGE_FP16_TRITON = "SAGE_FP16_TRITON"
    SAGE_FP16_CUDA = "SAGE_FP16_CUDA"
    SAGE_FP8_CUDA = "SAGE_FP8_CUDA"
    SAGE_FP8_SM90 = "SAGE_FP8_SM90"
    AITER_ROCM = "AITER_ROCM"

class CompactCompressType(Enum):
    WARMUP = "warmup"
    SPARSE = "sparse"
    BINARY = "binary"
    INT2 = "int2"
    INT2_MINMAX = "int2-minmax"
    INT4 = "int4"
    IDENTITY = "identity"
    LOW_RANK = "low-rank"
    LOW_RANK_Q = "low-rank-int4"
    LOW_RANK_AWL = "low-rank-awl"

def _get_default_world_size() -> int:
    """Smart default: query available GPUs if in a CUDA environment."""
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    except:
        return 1

@dataclass(frozen=True)
class ExecutionStrategy:
    """
    Unified strategy for parallelism and attention kernels.
    
    Attributes:
        ulysses_degree: Degree of Ulysses (sequence) parallelism.
        ring_degree: Degree of Ring Attention parallelism.
        cfg_degree: Degree of Classifier-Free Guidance parallelism.
        sync_ulysses: Enable synchronous all-to-all for Ulysses.
        fsdp_enabled: Enable Fully Sharded Data Parallelism.
        fsdp_cpu_offload: Offload FSDP parameters to CPU.
        fsdp_parallel_load: Load FSDP shards in parallel from disk.
        attention_backend: The high-level backend (e.g., "STANDARD", "COMPACT").
        attention_type: The specific attention kernel implementation.
        ring_impl: The ring attention implementation strategy (e.g., "basic", "zigzag").
    """
    # Parallelism degrees
    ulysses_degree: int = 1
    ring_degree: int = 1
    cfg_degree: int = 1
    tensor_parallel_degree: int = 1
    
    # Flags
    sync_ulysses: bool = False
    fsdp_enabled: bool = False
    fsdp_cpu_offload: bool = False
    fsdp_parallel_load: bool = True
    
    # Kernel selection
    attention_backend: str = "STANDARD"  # "STANDARD" or "COMPACT"
    attention_type: RaylightAttnType = RaylightAttnType.TORCH
    ring_impl: str = "basic"              # "basic" or "zigzag"

    # TP Communication Compression
    tp_allreduce_compress: str = "none"    # "none" | "fp8" | "turboquant"
    tp_compress_bits: int = 4              # 2 | 3 | 4  (bit-width for turboquant)
    tp_compress_group_size: int = 64       # must divide hidden_dim
    tp_compress_residual: bool = False     # step-to-step residual caching (Phase 2)
    tp_compress_rotation: str = "signperm" # "signperm" | "wht" (Phase 3)

    @property
    def is_fsdp(self) -> bool:
        """True when Fully Sharded Data Parallelism is active."""
        return self.fsdp_enabled

    @property
    def is_tp(self) -> bool:
        """True when Tensor Parallelism is active."""
        return self.tensor_parallel_degree > 1

    @property
    def is_xdit(self) -> bool:
        """True when any sequence/CFG parallelism degree exceeds 1."""
        return self.ulysses_degree * self.ring_degree * self.cfg_degree > 1

    @property
    def total_parallel_degree(self) -> int:
        """Total parallelism degree (product of all degrees)."""
        return (
            self.ulysses_degree
            * self.ring_degree
            * self.cfg_degree
            * self.tensor_parallel_degree
        )

    def __post_init__(self):
        """Cross-field validation for strategy consistency."""
        total_sp = self.ulysses_degree * self.ring_degree * self.cfg_degree
        if self.fsdp_enabled and total_sp > 1:
            # FSDP can technically work with SP, but often requires specific group orchestration
            pass 

@dataclass(frozen=True)
class CompactConfig:
    """
    Configuration for the COMPACT (Activation Compression) fusion system.
    
    Attributes:
        enabled: Whether to enable Delta-based activation compression.
        warmup_steps: Number of steps to run in 'warmup' mode before activating compression.
        delta_compression: The type of compression to use for activation deltas.
        kv_cache_quant_enable: Enable quantization for the KV cache.
        kv_cache_quant_bits: Bit-depth for KV cache quantization (default 8).
    """
    enabled: bool = False
    warmup_steps: int = 1
    delta_compression: CompactCompressType = CompactCompressType.BINARY
    kv_cache_quant_enable: bool = False
    kv_cache_quant_bits: int = 8
    # Advanced / Internal
    residual: int = 1
    error_feedback: bool = True
    simulate: bool = False
    fastpath: bool = False
    quantized_cache: bool = True
    cache_quant_bits: int = 8
    delta_decay_factor: Optional[float] = None

@dataclass(frozen=True)
class ClusterConfig:
    """
    Ray cluster and namespace coordination.
    
    Attributes:
        address: The Ray cluster address (e.g., 'auto', 'local', or a node IP).
        namespace: The Ray namespace for actor discovery and isolation.
        object_store_gb: Maximum size of the Ray object store in Gigabytes.
    """
    address: str = "local"
    namespace: str = "default"
    object_store_gb: float = 0.0
    dashboard_host: str = "127.0.0.1"
    dashboard_port: Optional[int] = None

@dataclass(frozen=True)
class DeviceConfig:
    """
    Hardware and memory allocation settings.
    
    Attributes:
        world_size: Total number of GPUs/actors in the cluster.
        gpu_indices: Specific CUDA device indices to use.
        vram_limit_gb: Soft memory limit to prevent OOM.
        use_mmap: Use memory-mapped files for model weights.
        zero_ram: Stream weights directly from NVMe to VRAM with no RAM build-up.
    """
    world_size: int = field(default_factory=_get_default_world_size)
    gpu_indices: List[int] = field(default_factory=list)
    vram_limit_gb: float = 0.0
    use_mmap: bool = True
    zero_ram: bool = False

@dataclass(frozen=True)
class DebugConfig:
    """
    Observability and developer-only flags.
    
    Attributes:
        verbose_attn: Print detailed timing and shape info for every attention call.
        overlap_decomp: Overlap GPU decompression with compute.
        allow_deprecated: Enable legacy compression modes for testing.
        dedup_logs: Prevent log spam from multiple Ray actors.
    """
    verbose_attn: bool = False
    overlap_decomp: bool = True
    allow_deprecated: bool = False
    use_awl: bool = False
    skip_comm_test: bool = True
    enable_kitchen: bool = False
    dedup_logs: bool = False
    trace_fsdp: bool = False
    profile_sampler: bool = False
    debug_vae: bool = False
    # Stats / Research
    calc_similarity: bool = False
    calc_more_similarity: bool = False
    print_all_error: bool = False
    ref_activation_path: str = "ref_activations"
    dump_activations: bool = False
    calc_total_error: bool = False
    calc_error: bool = False

@dataclass(frozen=True)
class SystemConfig:
    """
    Environment and system-level tuning (Tokenizers, GIL, Networking).
    
    Attributes:
        tokenizers_parallelism: Force enable/disable parallelism in HuggingFace Tokenizers.
        python_gil: Set the Python Global Interpreter Lock behavior.
        master_addr: IP address of the primary node for distributed sync.
        master_port: Port number for the primary node listener.
    """
    ray_usage_stats: bool = False
    ray_metrics_collection: bool = False
    ray_metrics_export_interval: int = 0
    ray_dedup_logs: bool = False
    tokenizers_parallelism: bool = False
    python_gil: bool = True
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    host_ipc_mode: str = "auto"
    host_ipc_path: Optional[str] = None

@dataclass
class RuntimeMetadata:
    """
    Calculated properties that are derived from the primary config at runtime.
    Note: These are not serializable and are recomputed on every config init.
    
    Attributes:
        total_sp_degree: product of all parallelism degrees (ulysses * ring * cfg).
        is_distributed: True if running on more than one GPU/process.
        active_device: The primary device string ('cuda', 'xpu', 'cpu').
    """
    total_sp_degree: int = 0
    is_distributed: bool = False
    active_device: str = "cpu"

@dataclass(frozen=True)
class RaylightConfig:
    """
    The definitive "Single Source of Truth" configuration for Raylight.
    
    This object encapsulates all hardware, execution, and research parameters.
    It provides factory methods for environment variable loading and
    built-in validation to prevent invalid distributed states.
    """
    strategy: ExecutionStrategy = field(default_factory=ExecutionStrategy)
    compact: CompactConfig = field(default_factory=CompactConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Derived metadata (Calculated in __post_init__)
    meta: RuntimeMetadata = field(default_factory=RuntimeMetadata, init=False, repr=False)

    def __post_init__(self):
        """Perform system-wide validation and calculate metadata."""
        # 1. Calculate Metadata
        object.__setattr__(self, 'meta', RuntimeMetadata(
            total_sp_degree = self.strategy.ulysses_degree * self.strategy.ring_degree * self.strategy.cfg_degree,
            is_distributed = self.device.world_size > 1,
            active_device = "cuda" if torch.cuda.is_available() else "cpu"
        ))

        # 2. Functional Validations
        if self.meta.total_sp_degree > self.device.world_size:
            raise ValueError(
                f"Total parallelism degree ({self.meta.total_sp_degree}) "
                f"cannot exceed available world size ({self.device.world_size})"
            )
        
        if self.compact.enabled and self.strategy.ring_degree == 1:
            # We don't raise error here, but it's a "silent" concern in research code
            pass

    @classmethod
    def from_env(cls) -> "RaylightConfig":
        """Centralized factory for environment-based configuration."""
        # For Enums, we must handle strings safely
        def _get_enum(env_name, enum_cls, default):
            val = os.getenv(env_name)
            if val:
                try: return enum_cls[val]
                except KeyError: pass
            return default

        return cls(
            strategy=ExecutionStrategy(
                attention_backend=os.getenv("RAYLIGHT_ATTN_BACKEND", "STANDARD"),
                attention_type=_get_enum("RAYLIGHT_ATTN_TYPE", RaylightAttnType, RaylightAttnType.TORCH),
                ring_impl=os.getenv("RAYLIGHT_RING_IMPL", "basic"),
                fsdp_parallel_load=os.getenv("RAYLIGHT_FSDP_PARALLEL_LOAD", "1") == "1",
            ),
            compact=CompactConfig(
                enabled=os.getenv("RAYLIGHT_COMPACT", "0") == "1",
                warmup_steps=int(os.getenv("RAYLIGHT_COMPACT_WARMUP_STEPS", "1")),
                delta_compression=_get_enum("RAYLIGHT_DELTA_COMPRESSION", CompactCompressType, CompactCompressType.IDENTITY),
                kv_cache_quant_enable=os.getenv("RAYLIGHT_COMPACT_QUANTIZED_CACHE", "0") == "1",
                kv_cache_quant_bits=int(os.getenv("RAYLIGHT_COMPACT_CACHE_QUANT_BITS", "8")),
            ),
            debug=DebugConfig(
                verbose_attn=os.getenv("RAYLIGHT_VERBOSE_ATTN", "0") == "1",
                overlap_decomp=os.getenv("RAYLIGHT_OVERLAP_DECOMP", "1") == "1",
                allow_deprecated=os.getenv("COMPACT_ALLOW_DEPRECATED", "0") == "1",
                use_awl=os.getenv("USE_AWL", "0") == "1",
                enable_kitchen=os.getenv("RAYLIGHT_ENABLE_KITCHEN", "0") == "1",
                dedup_logs=os.getenv("RAY_DEDUP_LOGS", "0") == "1",
                profile_sampler=os.getenv("RAYLIGHT_PROFILE_SAMPLER", "0") == "1",
                debug_vae=os.getenv("RAYLIGHT_DEBUG_VAE", "0") == "1",
                calc_similarity=os.getenv("CALC_SIMILARITY", "0") == "1",
                calc_more_similarity=os.getenv("CALC_MORE_SIMILARITY", "0") == "1",
                print_all_error=os.getenv("PRINT_ALL_ERROR", "0") == "1",
                ref_activation_path=os.getenv("REF_ACTIVATION_PATH", "ref_activations"),
                dump_activations=os.getenv("DUMP_ACTIVATIONS", "0") == "1",
                calc_total_error=os.getenv("CALC_TOTAL_ERROR", "0") == "1",
                calc_error=os.getenv("CALC_ERROR", "0") == "1",
            ),
            system=SystemConfig(
                ray_usage_stats=os.getenv("RAY_USAGE_STATS_ENABLED", "0") == "1",
                ray_metrics_collection=os.getenv("RAY_enable_metrics_collection", "0") == "1",
                ray_metrics_export_interval=int(os.getenv("RAY_METRICS_EXPORT_INTERVAL_MS", "0")),
                ray_dedup_logs=os.getenv("RAY_DEDUP_LOGS", "0") == "1",
                tokenizers_parallelism=os.getenv("TOKENIZERS_PARALLELISM", "false").lower() == "true",
                python_gil=os.getenv("PYTHON_GIL", "1") == "1",
                master_addr=os.getenv("MASTER_ADDR", "127.0.0.1"),
                master_port=os.getenv("MASTER_PORT", "29500"),
            )
        )

    def apply_to_env(self):
        """Apply system-level configuration to the current OS environment."""
        os.environ.update({
            "MASTER_ADDR": self.system.master_addr,
            "MASTER_PORT": self.system.master_port,
            "TOKENIZERS_PARALLELISM": "true" if self.system.tokenizers_parallelism else "false",
            "RAY_USAGE_STATS_ENABLED": "1" if self.system.ray_usage_stats else "0",
            "RAY_enable_metrics_collection": "1" if self.system.ray_metrics_collection else "0",
            "RAY_METRICS_EXPORT_INTERVAL_MS": str(self.system.ray_metrics_export_interval),
            "RAY_DEDUP_LOGS": "1" if (self.system.ray_dedup_logs or self.debug.dedup_logs) else "0",
            "PYTHON_GIL": "1" if self.system.python_gil else "0",
        })

    def to_json(self) -> str:
        """Export config for experiment tracking or workflow metadata."""
        def _enum_encoder(obj):
            if isinstance(obj, Enum):
                return obj.name
            return obj
        
        # Clean dict for JSON serialization (convert enums to strings)
        clean_dict = json.loads(json.dumps(asdict(self), default=_enum_encoder))
        return json.dumps(clean_dict, indent=2)

    def to_yaml(self) -> str:
        """Simple YAML-like export without external dependencies."""
        def _format(d, indent=0):
            lines = []
            for k, v in d.items():
                if isinstance(v, dict):
                    lines.append(f"{'  ' * indent}{k}:")
                    lines.extend(_format(v, indent + 1))
                else:
                    val = v.name if isinstance(v, Enum) else v
                    lines.append(f"{'  ' * indent}{k}: {val}")
            return lines
        return "\n".join(_format(asdict(self)))

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Bridge for existing dictionary-based internal APIs."""
        return {
            "ulysses_degree": self.strategy.ulysses_degree,
            "ring_degree": self.strategy.ring_degree,
            "cfg_degree": self.strategy.cfg_degree,
            "sync_ulysses": self.strategy.sync_ulysses,
            "attention": self.strategy.attention_type.name,
            "attention_backend": self.strategy.attention_backend,
            "ring_impl_type": self.strategy.ring_impl,
            "is_xdit": self.meta.total_sp_degree > 1,
            "is_fsdp": self.strategy.fsdp_enabled,
            "fsdp_cpu_offload": self.strategy.fsdp_cpu_offload,
            "use_kitchen": self.debug.enable_kitchen,
            "kv_cache_quant_enable": self.compact.kv_cache_quant_enable,
            "kv_cache_quant_bits": self.compact.kv_cache_quant_bits,
            "delta_compression": self.compact.delta_compression.name,
            "compact_warmup_steps": self.compact.warmup_steps,
            "use_mmap": self.device.use_mmap,
            "fsdp_parallel_load": self.strategy.fsdp_parallel_load,
            "enable_kitchen": self.debug.enable_kitchen,
            "global_world_size": self.device.world_size,
            "vram_limit_bytes": int(self.device.vram_limit_gb * (1024 ** 3)) if self.device.vram_limit_gb > 0 else 0,
        }
