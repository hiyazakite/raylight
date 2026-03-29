"""Shim pointing at implementation in :mod:`raylight.ipc.backends.file_mmap`.

Keeping this top-level module avoids breaking imports; the real
implementation now lives in `ipc.backends.file_mmap`.
"""

from .backends.file_mmap import FileMmapHostIpcBackend  # re-export

__all__ = ["FileMmapHostIpcBackend"]