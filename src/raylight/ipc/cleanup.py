"""Stale artifact cleanup for all Raylight host-memory categories."""

from __future__ import annotations

import glob
import logging
import os
import time
from collections.abc import Iterable

from .backend import HostIpcBackend

log = logging.getLogger(__name__)

# Prefixes used by the POSIX shm backend for cleanup pattern generation.
PINNED_CACHE_PREFIX = "raylight_pc"
GGUF_DEQUANT_PREFIX = "raylight_gguf"
SHM_CLEANUP_PREFIXES: list[str] = [PINNED_CACHE_PREFIX, GGUF_DEQUANT_PREFIX]


def cleanup_stale(
    backend: HostIpcBackend,
    prefixes: Iterable[str],
    *,
    older_than_s: float | None = None,
) -> None:
    """Remove orphaned host-memory artifacts for a single backend.

    Queries *backend* for its cleanup patterns using *prefixes*, then
    removes all matching filesystem entries.  When *older_than_s* is set,
    only artifacts whose mtime predates the threshold are removed.
    """
    for cleanup_pattern in backend.iter_cleanup_patterns(prefixes):
        for path in glob.glob(cleanup_pattern.pattern):
            if older_than_s is not None and _is_too_young(path, older_than_s):
                continue
            try:
                os.remove(path)
            except OSError:
                pass


def cleanup_all_stale(*, older_than_s: float | None = None) -> None:
    """Remove orphaned artifacts from all available backends.

    Builds the file-mmap and POSIX shm backends, skipping any that are
    unavailable on the current platform.  Safe to call at startup (before
    workers exist) or during actor teardown.
    """
    from .resolver import build_file_mmap_backend, build_posix_shm_backend
    from .vae_ipc import VAE_CLEANUP_PREFIXES

    # File-mmap VAE artifacts.
    try:
        cleanup_stale(
            build_file_mmap_backend(), VAE_CLEANUP_PREFIXES,
            older_than_s=older_than_s,
        )
    except Exception:
        log.debug("File-mmap backend unavailable for cleanup", exc_info=True)

    # POSIX shm segments (pinned-cache + GGUF dequant).
    try:
        cleanup_stale(
            build_posix_shm_backend(), SHM_CLEANUP_PREFIXES,
            older_than_s=older_than_s,
        )
    except Exception:
        log.debug("POSIX shm backend unavailable for cleanup", exc_info=True)


def _is_too_young(path: str, older_than_s: float) -> bool:
    """Return True if *path* is younger than *older_than_s* seconds."""
    try:
        return time.time() - os.path.getmtime(path) < older_than_s
    except OSError:
        return True
