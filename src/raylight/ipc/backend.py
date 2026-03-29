"""Top-level shim for backend implementations moved into `ipc.backends`.

This file re-exports the protocol from the `backends` package so existing
imports continue to work while the concrete implementations live under
`ipc/backends`.
"""

from .backends.backend import HostIpcBackend  # re-export

__all__ = ["HostIpcBackend"]