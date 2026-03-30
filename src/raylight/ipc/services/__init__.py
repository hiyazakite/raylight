"""Host IPC service protocol and default implementation."""

from .service import HostIpcService
from .service_impl import DefaultHostIpcService

__all__ = ["HostIpcService", "DefaultHostIpcService"]
