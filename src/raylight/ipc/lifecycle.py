"""Lifecycle helpers for Host-managed IPC artifacts."""

from __future__ import annotations

from .errors import HostIpcArtifactLifecycleError
from .types import HostIpcLifecycleState


_VALID_TRANSITIONS = {
    HostIpcLifecycleState.ALLOCATED: {
        HostIpcLifecycleState.ATTACHED,
        HostIpcLifecycleState.WRITABLE,
        HostIpcLifecycleState.RELEASED,
    },
    HostIpcLifecycleState.ATTACHED: {
        HostIpcLifecycleState.WRITABLE,
        HostIpcLifecycleState.RELEASED,
    },
    HostIpcLifecycleState.WRITABLE: {
        HostIpcLifecycleState.ATTACHED,
        HostIpcLifecycleState.SEALED,
        HostIpcLifecycleState.RELEASED,
    },
    HostIpcLifecycleState.SEALED: {
        HostIpcLifecycleState.RELEASED,
    },
    HostIpcLifecycleState.RELEASED: set(),
}


def can_transition(current: HostIpcLifecycleState, new_state: HostIpcLifecycleState) -> bool:
    """Return True when a lifecycle transition is allowed."""

    return new_state in _VALID_TRANSITIONS[current]


def assert_valid_transition(current: HostIpcLifecycleState, new_state: HostIpcLifecycleState) -> None:
    """Raise when an invalid lifecycle transition is attempted."""

    if can_transition(current, new_state):
        return
    raise HostIpcArtifactLifecycleError(
        f"Invalid Host IPC lifecycle transition: {current.value} -> {new_state.value}"
    )