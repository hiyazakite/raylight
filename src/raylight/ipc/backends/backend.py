"""Host-managed IPC backend protocol (moved to backends package).

This module was moved from the top-level `ipc` package into
`ipc.backends` to group concrete implementations together.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ..types import HostIpcArtifactMetadata, HostIpcBackendInfo, HostIpcBufferSpec, HostIpcCleanupPattern


@runtime_checkable
class HostIpcBackend(Protocol):
    """Structural contract for a RAM-backed Host IPC backend."""

    @property
    def info(self) -> HostIpcBackendInfo: ...

    def validate(self) -> None:
        """Validate backend capability and raise on failure."""
        ...

    def create_artifact(self, spec: HostIpcBufferSpec) -> HostIpcArtifactMetadata:
        """Create a new artifact and return serializable metadata."""
        ...

    def attach_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        """Attach to an existing artifact and return updated metadata."""
        ...

    def seal_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        """Transition an artifact from writable to sealed."""
        ...

    def release_artifact(self, metadata: HostIpcArtifactMetadata, *, unlink: bool = False) -> None:
        """Release a local attachment and optionally unlink the artifact."""
        ...

    def iter_cleanup_patterns(self, prefixes: Iterable[str]) -> Iterable[HostIpcCleanupPattern]:
        """Return backend-specific patterns for stale artifact cleanup."""
        ...
