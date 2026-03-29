"""Host-managed IPC error hierarchy."""


class HostIpcError(RuntimeError):
    """Base class for Host IPC failures."""


class HostIpcBackendValidationError(HostIpcError):
    """Raised when a backend candidate fails capability validation."""


class HostIpcBackendUnavailableError(HostIpcError):
    """Raised when no suitable RAM-backed Host IPC backend is available."""


class HostIpcArtifactAttachTimeoutError(HostIpcError):
    """Raised when an attacher times out waiting for an artifact."""


class HostIpcStaleArtifactError(HostIpcError):
    """Raised when metadata points to an artifact that is no longer valid."""


class HostIpcArtifactLifecycleError(HostIpcError):
    """Raised when an invalid lifecycle transition is attempted."""