"""Backend-agnostic memory usage collection for Host IPC.

Used by :mod:`utils.memory` as the canonical stats provider instead of
direct glob scans scattered across consumer modules.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from dataclasses import dataclass

from .backends.backend import HostIpcBackend
from .services import HostIpcService


@dataclass(frozen=True)
class MemorySourceStats:
    """Usage stats for one category of shared-memory artifacts."""

    kind_name: str
    artifact_bytes: int
    artifact_count: int


def _sum_glob_sizes(pattern: str) -> tuple[int, int]:
    """Return ``(bytes, count)`` for all paths matching *pattern*, best-effort."""
    total_bytes = 0
    count = 0
    for path in glob.glob(pattern):
        try:
            total_bytes += os.path.getsize(path)
            count += 1
        except OSError:
            pass
    return total_bytes, count


def collect_backend_stats(
    backend: HostIpcBackend,
    prefixes: Iterable[str],
    kind_name: str,
) -> MemorySourceStats:
    """Collect stats for *prefixes* via a backend's cleanup patterns."""
    total_bytes = 0
    total_count = 0
    for cleanup_pattern in backend.iter_cleanup_patterns(prefixes):
        b, c = _sum_glob_sizes(cleanup_pattern.pattern)
        total_bytes += b
        total_count += c
    return MemorySourceStats(
        kind_name=kind_name,
        artifact_bytes=total_bytes,
        artifact_count=total_count,
    )


def collect_ipc_artifact_stats(
    service: HostIpcService,
    prefixes: Iterable[str],
) -> MemorySourceStats:
    """Collect file-mmap artifact stats via IPC service cleanup patterns."""
    total_bytes = 0
    total_count = 0
    for cleanup_pattern in service.iter_cleanup_patterns(prefixes):
        b, c = _sum_glob_sizes(cleanup_pattern.pattern)
        total_bytes += b
        total_count += c
    return MemorySourceStats(
        kind_name="file_mmap",
        artifact_bytes=total_bytes,
        artifact_count=total_count,
    )


def collect_legacy_pt_stats() -> MemorySourceStats:
    """Collect stats for legacy ``.pt`` scratch files in ``/dev/shm``."""
    b, c = _sum_glob_sizes("/dev/shm/raylight_*.pt")
    return MemorySourceStats(kind_name="legacy_pt", artifact_bytes=b, artifact_count=c)


def collect_all_stats(ipc_service: HostIpcService) -> list[MemorySourceStats]:
    """Return per-source memory usage stats.

    Builds all available backends and collects stats through their cleanup
    patterns.  Returns a list of :class:`MemorySourceStats` with
    ``kind_name`` values ``"file_mmap"``, ``"pinned_shm"``,
    ``"gguf_dequant_shm"``, and ``"legacy_pt"``.
    """
    from .cleanup import GGUF_DEQUANT_PREFIX, PINNED_CACHE_PREFIX
    from .resolver import build_posix_shm_backend
    from .vae_ipc import VAE_CLEANUP_PREFIXES

    results: list[MemorySourceStats] = [
        collect_ipc_artifact_stats(ipc_service, VAE_CLEANUP_PREFIXES),
    ]

    try:
        shm_backend = build_posix_shm_backend()
        results.append(
            collect_backend_stats(shm_backend, [PINNED_CACHE_PREFIX], "pinned_shm")
        )
        results.append(
            collect_backend_stats(shm_backend, [GGUF_DEQUANT_PREFIX], "gguf_dequant_shm")
        )
    except Exception:
        pass

    results.append(collect_legacy_pt_stats())
    return results
