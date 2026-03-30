"""Thin VAE-decode adapter over HostIpcService.

Maps VAE-decode semantics (master_shape, float32 output buffer) onto the
generic HostIpcService contract so that nodes.py has no knowledge of
HostIpcBufferSpec details.
"""

from __future__ import annotations

from .services import HostIpcService
from .types import HostIpcAccessMode, HostIpcArtifactMetadata, HostIpcBufferSpec

VAE_OUT_PREFIX = "raylight_vae_out"

# Prefixes that stale-cleanup should scan via HostIpcService.
# Note: raylight_pc segments are NOT file-mmap artifacts — they are POSIX shm
# segments created by SharedPinnedParamCache. They must be cleaned up via a
# separate posix-shm cleanup path, not through the file-mmap service.
VAE_CLEANUP_PREFIXES: list[str] = [VAE_OUT_PREFIX]


def begin_vae_decode_job(
    service: HostIpcService,
    master_shape: tuple[int, ...],
    *,
    scope: str = "ray_decode",
) -> HostIpcArtifactMetadata:
    """Allocate a shared RAM-backed float32 output buffer for one VAE decode job.

    The returned metadata carries ``metadata.path`` which callers pass as
    ``mmap_path`` to worker actors and to ``torch.from_file``.
    """
    num_elements = 1
    for d in master_shape:
        num_elements *= d
    size_bytes = num_elements * 4  # float32 = 4 bytes per element

    spec = HostIpcBufferSpec(
        prefix=VAE_OUT_PREFIX,
        logical_name="vae_output",
        owner_scope=scope,
        size_bytes=size_bytes,
        shape=master_shape,
        dtype_name="float32",
        access_mode=HostIpcAccessMode.READ_WRITE,
    )
    return service.allocate_writable_artifact(spec)


def release_vae_decode_job(
    service: HostIpcService,
    metadata: HostIpcArtifactMetadata,
) -> None:
    """Release and unlink the shared output buffer for a VAE decode job.

    Safe to call on both the success path and the exception path.
    """
    service.release_artifact(metadata, unlink=True)
