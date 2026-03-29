"""Default Host IPC service implementation."""

from __future__ import annotations

from collections.abc import Iterable

from .backend import HostIpcBackend
from .types import HostIpcArtifactMetadata, HostIpcBackendInfo, HostIpcBufferSpec, HostIpcCleanupPattern


class DefaultHostIpcService:
    """Thin service wrapper used by managers and teardown flows."""

    def __init__(self, backend: HostIpcBackend) -> None:
        self._backend = backend

    def get_backend_info(self) -> HostIpcBackendInfo:
        self._backend.validate()
        return self._backend.info

    def allocate_writable_artifact(self, spec: HostIpcBufferSpec) -> HostIpcArtifactMetadata:
        return self._backend.create_artifact(spec)

    def attach_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        return self._backend.attach_artifact(metadata)

    def seal_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        return self._backend.seal_artifact(metadata)

    def release_artifact(self, metadata: HostIpcArtifactMetadata, *, unlink: bool = False) -> None:
        self._backend.release_artifact(metadata, unlink=unlink)

    def iter_cleanup_patterns(self, prefixes: Iterable[str]) -> Iterable[HostIpcCleanupPattern]:
        return self._backend.iter_cleanup_patterns(prefixes)