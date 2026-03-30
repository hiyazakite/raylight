"""Manager-facing Host IPC service contract.

Managers should talk to this service boundary rather than directly owning
backend-specific allocation details. Concrete implementations can wrap a
single backend or choose among multiple validated backends.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ..types import HostIpcArtifactMetadata, HostIpcBackendInfo, HostIpcBufferSpec, HostIpcCleanupPattern


@runtime_checkable
class HostIpcService(Protocol):
    """High-level orchestration contract used by managers and teardown flows."""

    def get_backend_info(self) -> HostIpcBackendInfo:
        """Return the active backend description after validation/selection."""
        ...

    def allocate_writable_artifact(self, spec: HostIpcBufferSpec) -> HostIpcArtifactMetadata:
        """Create a writable artifact for a producing workflow stage."""
        ...

    def attach_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        """Attach a consumer or secondary writer to an existing artifact."""
        ...

    def seal_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        """Seal a writable artifact once producers have finished writing."""
        ...

    def release_artifact(self, metadata: HostIpcArtifactMetadata, *, unlink: bool = False) -> None:
        """Release a local reference and optionally unlink the shared artifact."""
        ...

    def iter_cleanup_patterns(self, prefixes: Iterable[str]) -> Iterable[HostIpcCleanupPattern]:
        """Return cleanup patterns for stale artifact scanning."""
        ...
