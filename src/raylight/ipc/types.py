"""Serializable types and enums for Host-managed IPC contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class HostIpcBackendKind(str, Enum):
    """Backend families supported by the Host IPC layer."""

    FILE_MMAP = "file_mmap"
    NAMED_SHARED_MEMORY = "named_shared_memory"


class HostIpcLifecycleState(str, Enum):
    """Lifecycle states for a shared host artifact."""

    ALLOCATED = "allocated"
    ATTACHED = "attached"
    WRITABLE = "writable"
    SEALED = "sealed"
    RELEASED = "released"


class HostIpcAccessMode(str, Enum):
    """Access policy for an IPC artifact."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class HostIpcBackendInfo:
    """Static capabilities for an IPC backend implementation."""

    name: str
    kind: HostIpcBackendKind
    ram_backed: bool
    supports_file_paths: bool
    supports_pinned_registration: bool
    description: str = ""


@dataclass(frozen=True)
class HostIpcCleanupPattern:
    """Backend-specific cleanup pattern emitted for stale artifact pruning."""

    pattern: str
    backend_name: str
    recursive: bool = False


@dataclass(frozen=True)
class HostIpcBufferSpec:
    """Logical specification for a Host IPC buffer to be created."""

    prefix: str
    logical_name: str
    owner_scope: str
    size_bytes: int
    shape: Optional[Tuple[int, ...]] = None
    dtype_name: Optional[str] = None
    access_mode: HostIpcAccessMode = HostIpcAccessMode.READ_WRITE
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HostIpcArtifactMetadata:
    """Serializable metadata exchanged between creators and attachers."""

    artifact_id: str
    backend_name: str
    backend_kind: HostIpcBackendKind
    owner_scope: str
    state: HostIpcLifecycleState
    size_bytes: int
    path: Optional[str] = None
    shared_name: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype_name: Optional[str] = None
    access_mode: HostIpcAccessMode = HostIpcAccessMode.READ_WRITE
    metadata: Mapping[str, Any] = field(default_factory=dict)