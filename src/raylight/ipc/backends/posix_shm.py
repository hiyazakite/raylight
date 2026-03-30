"""POSIX named shared-memory Host IPC backend (moved into backends package).

Moved from top-level ipc package; adjusted relative imports.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch as _mock_patch

from ..errors import HostIpcBackendValidationError, HostIpcStaleArtifactError
from ..lifecycle import assert_valid_transition
from ..types import (
    HostIpcAccessMode,
    HostIpcArtifactMetadata,
    HostIpcBackendInfo,
    HostIpcBackendKind,
    HostIpcBufferSpec,
    HostIpcCleanupPattern,
    HostIpcLifecycleState,
)

# Linux POSIX shm root.
_SHM_ROOT = "/dev/shm"


def _weak_unlink_shm(name: str) -> None:
    """weakref.finalize callback — best-effort unlink of a POSIX shm segment."""
    try:
        shm = SharedMemory(name=name, create=False)
        shm.unlink()
        try:
            shm.close()
        except BufferError:
            pass
    except Exception:
        pass


def _noop(*_args, **_kwargs) -> None:  # noqa: ANN002
    """No-op used to suppress the multiprocessing resource tracker."""


class PosixShmBackend:
    """Concrete backend for POSIX named shared-memory segments.

    Each artifact maps to a single ``SharedMemory`` object.  The creator
    calls :meth:`create_artifact`, non-creators call :meth:`attach_artifact`
    with resource-tracker suppression to avoid the Python bug where
    non-owner processes incorrectly unlink on exit.
    """

    def __init__(self, *, backend_name: str = "linux_posix_shm") -> None:
        self._backend_name = backend_name
        self._info = HostIpcBackendInfo(
            name=backend_name,
            kind=HostIpcBackendKind.NAMED_SHARED_MEMORY,
            ram_backed=True,
            supports_file_paths=False,
            supports_pinned_registration=True,
            description=f"POSIX named shared-memory backend (root: {_SHM_ROOT})",
        )
        # Track live SharedMemory handles so consumers can access .buf
        self._handles: dict[str, SharedMemory] = {}

    @property
    def info(self) -> HostIpcBackendInfo:
        return self._info

    def validate(self) -> None:
        import os

        if not os.path.isdir(_SHM_ROOT):
            raise HostIpcBackendValidationError(
                f"POSIX shm root does not exist: {_SHM_ROOT}"
            )
        if not os.access(_SHM_ROOT, os.R_OK | os.W_OK):
            raise HostIpcBackendValidationError(
                f"POSIX shm root is not accessible: {_SHM_ROOT}"
            )

    def create_artifact(self, spec: HostIpcBufferSpec) -> HostIpcArtifactMetadata:
        """Create a new POSIX shm segment.

        If a segment with the same name already exists it is unlinked first
        (handles stale leftovers from crashed processes).

        Registers a ``weakref.finalize`` on the returned ``SharedMemory``
        handle as a crash-safety net (mirrors the vLLM pattern).
        """
        self.validate()
        shm_name = f"{spec.prefix}_{spec.logical_name}"

        # Clean up pre-existing segment (stale leftover from crash).
        try:
            old = SharedMemory(name=shm_name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        shm = SharedMemory(name=shm_name, create=True, size=spec.size_bytes)
        weakref.finalize(shm, _weak_unlink_shm, shm_name)
        self._handles[shm_name] = shm

        return HostIpcArtifactMetadata(
            artifact_id=shm_name,
            backend_name=self.info.name,
            backend_kind=self.info.kind,
            owner_scope=spec.owner_scope,
            state=HostIpcLifecycleState.WRITABLE,
            size_bytes=shm.size,
            shared_name=shm_name,
            path=f"{_SHM_ROOT}/{shm_name}",
            shape=spec.shape,
            dtype_name=spec.dtype_name,
            access_mode=spec.access_mode,
            metadata=spec.metadata,
        )

    def attach_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        """Attach to an existing POSIX shm segment (non-creator).

        Suppresses the multiprocessing resource tracker to prevent the
        Python bug where non-creator processes unlink the segment on exit.
        """
        name = metadata.shared_name
        if name is None:
            raise HostIpcStaleArtifactError(
                f"Artifact '{metadata.artifact_id}' has no shared_name"
            )
        with _mock_patch(
            "multiprocessing.resource_tracker.register", _noop
        ):
            shm = SharedMemory(name=name, create=False)
        self._handles[name] = shm
        assert_valid_transition(metadata.state, HostIpcLifecycleState.ATTACHED)
        return replace(metadata, state=HostIpcLifecycleState.ATTACHED)

    def attach_by_name(self, name: str) -> HostIpcArtifactMetadata:
        """Attach to an existing POSIX shm segment by name only.

        Convenience wrapper for non-creators that know only the segment
        name (e.g. pinned-cache workers, GGUF readers).  Constructs
        minimal metadata internally and delegates to :meth:`attach_artifact`.
        """
        stub = HostIpcArtifactMetadata(
            artifact_id=name,
            backend_name=self.info.name,
            backend_kind=self.info.kind,
            owner_scope="remote",
            state=HostIpcLifecycleState.WRITABLE,
            size_bytes=0,
            shared_name=name,
        )
        return self.attach_artifact(stub)

    def attach_with_retry(
        self,
        name: str,
        *,
        retries: int = 50,
        delay_s: float = 0.2,
    ) -> HostIpcArtifactMetadata:
        """Attach to *name* with retries for cross-rank timing.

        Used when a non-creator rank needs to wait for rank 0 to finish
        creating the segment.  Raises :class:`RuntimeError` on timeout.
        """
        import time

        for attempt in range(retries):
            try:
                return self.attach_by_name(name)
            except FileNotFoundError:
                if attempt < retries - 1:
                    time.sleep(delay_s)
        total_s = retries * delay_s
        raise RuntimeError(
            f"Timed out waiting for shm '{name}' after {total_s:.0f}s"
        )

    def seal_artifact(self, metadata: HostIpcArtifactMetadata) -> HostIpcArtifactMetadata:
        assert_valid_transition(metadata.state, HostIpcLifecycleState.SEALED)
        return replace(metadata, state=HostIpcLifecycleState.SEALED)

    def release_artifact(self, metadata: HostIpcArtifactMetadata, *, unlink: bool = False) -> None:
        """Close the local handle and optionally unlink the segment."""
        name = metadata.shared_name
        if name is None:
            return
        shm = self._handles.pop(name, None)
        if shm is not None:
            try:
                shm.close()
            except BufferError:
                # Exported pointers (e.g. torch tensors via frombuffer)
                # still reference the mmap — suppress and let __del__
                # retry once those references are freed.
                pass
            if unlink:
                try:
                    shm.unlink()
                except Exception:
                    pass
        elif unlink:
            # No local handle — try to unlink by name anyway.
            _weak_unlink_shm(name)

    def get_handle(self, name: str) -> SharedMemory | None:
        """Return the live ``SharedMemory`` handle for *name*, or ``None``."""
        return self._handles.get(name)

    def iter_cleanup_patterns(self, prefixes: Iterable[str]) -> Iterable[HostIpcCleanupPattern]:
        for prefix in prefixes:
            yield HostIpcCleanupPattern(
                pattern=f"{_SHM_ROOT}/{prefix}_*",
                backend_name=self.info.name,
                recursive=False,
            )
