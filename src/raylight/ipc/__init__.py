"""Host-managed IPC contracts for Raylight.

Phase 1 scaffolded the interface and lifecycle contracts.
Phase 1 concrete: FileMmapHostIpcBackend, resolver, DefaultHostIpcService,
  config plumbing, and VAE decode artifact lifecycle wired in nodes.py.
"""

from .backend import HostIpcBackend
from .errors import (
    HostIpcArtifactAttachTimeoutError,
    HostIpcArtifactLifecycleError,
    HostIpcBackendUnavailableError,
    HostIpcBackendValidationError,
    HostIpcError,
    HostIpcStaleArtifactError,
)
from .lifecycle import assert_valid_transition, can_transition
from .cleanup import cleanup_stale, cleanup_all_stale, SHM_CLEANUP_PREFIXES
from .memory_stats import (
    collect_all_stats,
    collect_backend_stats,
    MemorySourceStats,
    collect_ipc_artifact_stats,
    collect_legacy_pt_stats,
)
from .posix_shm import PosixShmBackend
from .resolver import (
    build_default_host_ipc_service,
    build_file_mmap_backend,
    build_posix_shm_backend,
    is_ram_backed_path,
    resolve_file_mmap_root,
)
from .service import HostIpcService
from .service_impl import DefaultHostIpcService
from .types import (
    HostIpcAccessMode,
    HostIpcArtifactMetadata,
    HostIpcBackendInfo,
    HostIpcBackendKind,
    HostIpcBufferSpec,
    HostIpcCleanupPattern,
    HostIpcLifecycleState,
)
from .vae_ipc import (
    VAE_CLEANUP_PREFIXES,
    VAE_OUT_PREFIX,
    begin_vae_decode_job,
    release_vae_decode_job,
)

__all__ = [
    # Core IPC types
    "HostIpcAccessMode",
    "HostIpcArtifactAttachTimeoutError",
    "HostIpcArtifactLifecycleError",
    "HostIpcArtifactMetadata",
    "HostIpcBackend",
    "HostIpcBackendInfo",
    "HostIpcBackendKind",
    "HostIpcBackendUnavailableError",
    "HostIpcBackendValidationError",
    "HostIpcBufferSpec",
    "HostIpcCleanupPattern",
    "HostIpcError",
    "HostIpcLifecycleState",
    "HostIpcStaleArtifactError",
    "HostIpcService",
    "DefaultHostIpcService",
    "build_default_host_ipc_service",
    "assert_valid_transition",
    "build_file_mmap_backend",
    "build_posix_shm_backend",
    "PosixShmBackend",
    "can_transition",
    "is_ram_backed_path",
    "resolve_file_mmap_root",
    # Memory helpers
    "cleanup_stale",
    "cleanup_all_stale",
    "SHM_CLEANUP_PREFIXES",
    "collect_all_stats",
    "collect_backend_stats",
    "MemorySourceStats",
    "collect_ipc_artifact_stats",
    "collect_legacy_pt_stats",
    # VAE decode adapter
    "VAE_CLEANUP_PREFIXES",
    "VAE_OUT_PREFIX",
    "begin_vae_decode_job",
    "release_vae_decode_job",
]