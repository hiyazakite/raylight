"""RAM-backed file-mmap Host IPC backend (moved into backends package).

Original module moved from top-level to keep concrete backends together.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import uuid

from collections.abc import Iterable

from ..errors import HostIpcBackendValidationError, HostIpcStaleArtifactError
from ..lifecycle import assert_valid_transition
from ..types import (
    HostIpcArtifactMetadata,
    HostIpcBackendInfo,
    HostIpcBackendKind,
    HostIpcBufferSpec,
    HostIpcCleanupPattern,
    HostIpcLifecycleState,
)


class FileMmapHostIpcBackend:
    """Concrete backend for file-backed RAM mmap artifacts."""

    def __init__(self, root_dir: str | Path, *, backend_name: str = "linux_shm_file_mmap") -> None:
        self._root_dir = Path(root_dir).resolve()
        self._backend_name = backend_name
        self._info = HostIpcBackendInfo(
            name=backend_name,
            kind=HostIpcBackendKind.FILE_MMAP,
            ram_backed=True,
            supports_file_paths=True,
            supports_pinned_registration=True,
            description=f"RAM-backed file mmap backend rooted at {self._root_dir}",
        )

    @property
    def info(self) -> HostIpcBackendInfo:
        return self._info

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def validate(self) -> None:
        if not self._root_dir.exists():
            raise HostIpcBackendValidationError(
                f"Host IPC root does not exist: {self._root_dir}"
            )
        if not self._root_dir.is_dir():
            raise HostIpcBackendValidationError(
                f"Host IPC root is not a directory: {self._root_dir}"
            )
        if not os.access(self._root_dir, os.R_OK | os.W_OK | os.X_OK):
            raise HostIpcBackendValidationError(
                f"Host IPC root is not accessible for read/write/execute: {self._root_dir}"
            )

    def create_artifact(self, spec: HostIpcBufferSpec) -> HostIpcArtifactMetadata:
        self.validate()
        artifact_id = f"{spec.prefix}_{uuid.uuid4().hex}"
        artifact_path = self._root_dir / f"{artifact_id}.bin"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "wb") as handle:
            if spec.size_bytes > 0:
                handle.truncate(spec.size_bytes)

        return HostIpcArtifactMetadata(
            artifact_id=artifact_id,
            backend_name=self.info.name,
            backend_kind=self.info.kind,
            owner_scope=spec.owner_scope,
            state=HostIpcLifecycleState.WRITABLE,
            size_bytes=spec.size_bytes,
            path=str(artifact_path),
            shape=spec.shape,
            dtype_name=spec.dtype_name,
            access_mode=spec.access_mode,
            metadata=spec.metadata,
        )

    def attach_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        if metadata.path is None:
            raise HostIpcStaleArtifactError(
                f"Artifact '{metadata.artifact_id}' is missing file path metadata"
            )
        artifact_path = Path(metadata.path)
        if not artifact_path.exists():
            raise HostIpcStaleArtifactError(
                f"Artifact '{metadata.artifact_id}' no longer exists at {artifact_path}"
            )
        if artifact_path.stat().st_size < metadata.size_bytes:
            raise HostIpcStaleArtifactError(
                f"Artifact '{metadata.artifact_id}' is truncated: expected {metadata.size_bytes} bytes"
            )
        assert_valid_transition(metadata.state, HostIpcLifecycleState.ATTACHED)
        return replace(metadata, state=HostIpcLifecycleState.ATTACHED)

    def seal_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        assert_valid_transition(metadata.state, HostIpcLifecycleState.SEALED)
        return replace(metadata, state=HostIpcLifecycleState.SEALED)

    def release_artifact(self, metadata: HostIpcArtifactMetadata, *, unlink: bool = False) -> None:
        if metadata.path is None:
            return
        if not unlink:
            return
        try:
            Path(metadata.path).unlink(missing_ok=True)
        except OSError as exc:
            raise HostIpcStaleArtifactError(
                f"Failed to unlink Host IPC artifact '{metadata.artifact_id}': {exc}"
            ) from exc

    def iter_cleanup_patterns(self, prefixes: Iterable[str]):
        for prefix in prefixes:
            yield HostIpcCleanupPattern(
                pattern=str(self._root_dir / f"{prefix}_*.bin"),
                backend_name=self.info.name,
                recursive=False,
            )
