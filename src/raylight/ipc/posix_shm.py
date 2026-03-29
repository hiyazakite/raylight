"""Shim pointing at implementation in :mod:`raylight.ipc.backends.posix_shm`.

Keeping this top-level module avoids breaking imports; the real
implementation now lives in `ipc.backends.posix_shm`.
"""

from .backends.posix_shm import PosixShmBackend  # re-export

__all__ = ["PosixShmBackend"]
