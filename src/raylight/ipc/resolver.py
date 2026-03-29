"""Backend resolution helpers for Host-managed IPC."""

from __future__ import annotations

from pathlib import Path
import platform

from .errors import HostIpcBackendUnavailableError, HostIpcBackendValidationError
from .file_mmap import FileMmapHostIpcBackend
from .posix_shm import PosixShmBackend
from .service_impl import DefaultHostIpcService

_LINUX_RAM_BACKED_FS_TYPES = {"tmpfs", "ramfs", "hugetlbfs"}


def _find_mount_fs_type(path: Path) -> str | None:
    mounts_path = Path("/proc/mounts")
    if not mounts_path.exists():
        return None

    best_match: tuple[int, str] | None = None
    resolved = path.resolve()
    with open(mounts_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            mount_point = Path(parts[1].replace("\\040", " "))
            fs_type = parts[2]
            try:
                resolved.relative_to(mount_point)
            except ValueError:
                continue
            candidate = (len(str(mount_point)), fs_type)
            if best_match is None or candidate[0] > best_match[0]:
                best_match = candidate
    return best_match[1] if best_match is not None else None


def is_ram_backed_path(path: str | Path) -> bool:
    resolved = Path(path).resolve()
    if platform.system() != "Linux":
        return False
    fs_type = _find_mount_fs_type(resolved)
    return fs_type in _LINUX_RAM_BACKED_FS_TYPES


def resolve_file_mmap_root(mode: str = "auto", override_path: str | None = None) -> Path:
    if mode not in {"auto", "explicit"}:
        raise HostIpcBackendValidationError(
            f"Unsupported Host IPC mode '{mode}'. Expected 'auto' or 'explicit'."
        )

    if mode == "explicit":
        if not override_path:
            raise HostIpcBackendValidationError(
                "Host IPC mode 'explicit' requires a host_ipc_path override."
            )
        root = Path(override_path).resolve()
        if not is_ram_backed_path(root):
            raise HostIpcBackendValidationError(
                f"Configured Host IPC path is not on a verified RAM-backed mount: {root}"
            )
        return root

    default_root = Path("/dev/shm")
    if default_root.exists() and is_ram_backed_path(default_root):
        return default_root

    raise HostIpcBackendUnavailableError(
        "No verified RAM-backed Host IPC backend is available. Configure a RAM-backed mount via host_ipc_path."
    )


def build_file_mmap_backend(mode: str = "auto", override_path: str | None = None) -> FileMmapHostIpcBackend:
    root = resolve_file_mmap_root(mode=mode, override_path=override_path)
    backend = FileMmapHostIpcBackend(root)
    backend.validate()
    return backend


def build_posix_shm_backend() -> PosixShmBackend:
    """Build and validate the POSIX shared-memory backend."""
    backend = PosixShmBackend()
    backend.validate()
    return backend


def build_default_host_ipc_service(
    mode: str = "auto", override_path: str | None = None
) -> DefaultHostIpcService:
    """Build the default Host IPC service for the configured backend."""

    return DefaultHostIpcService(
        build_file_mmap_backend(mode=mode, override_path=override_path)
    )