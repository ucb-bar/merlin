"""Managed native client and public Chia hooks, independent of Chia internals.

The driver explicitly connects to a provisioned service. Pass setup/cleanup and
an Invitation through Chia's public hook kwargs; the task body calls run(). This
module itself neither dispatches Ray tasks nor changes accounting/task receipts.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..frozen_python import active_source_identity
from . import _protocol as protocol

_state = threading.local()


class CleanupIncomplete(RuntimeError):
    """Native lifecycle evidence is absent or incomplete, regardless of Ray state."""


@dataclass(frozen=True)
class Invitation:
    endpoint: str
    supervisor: str
    invocation: str
    session: str = field(repr=False)
    source: dict[str, str] | None = None


def _exchange(endpoint, message, fds=(), *, timeout=10):
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError("managed native control deadline expired")
    deadline = time.monotonic() + timeout
    with protocol.connect(endpoint, timeout=timeout) as connection:
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        protocol.send(connection, message, fds)
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        response, attachments = protocol.receive(connection)
        protocol.close_fds(attachments)
        if "error" in response:
            raise RuntimeError("managed supervisor refused request: " + response["error"])
        return response


class Session:
    """One driver's explicit lease; only its own invocations may be cancelled."""

    def __init__(self, endpoint: str | Path, *, expected_source: dict[str, str] | None = None):
        self.endpoint = str(Path(endpoint).absolute())
        if expected_source is not None and (
            not isinstance(expected_source, dict)
            or set(expected_source) != {"path", "sha256"}
            or not isinstance(expected_source["path"], str)
            or not Path(expected_source["path"]).is_absolute()
            or not isinstance(expected_source["sha256"], str)
            or len(expected_source["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in expected_source["sha256"])
        ):
            raise ValueError("managed expected source must be an explicit snapshot seal pin")
        active = active_source_identity()
        if active is not None and expected_source is not None and expected_source != active:
            raise ValueError("managed service source differs from the frozen caller")
        selected = expected_source if expected_source is not None else active
        self.source = dict(selected) if selected is not None else None
        fd = protocol.self_pidfd()
        try:
            response = _exchange(
                self.endpoint, {"operation": "session", "host": protocol.host_identity(), "source": self.source}, [fd]
            )
        finally:
            os.close(fd)
        self._token = response["session"]
        self.supervisor = response["supervisor"]
        if response.get("source") != self.source:
            raise RuntimeError("managed supervisor source identity mismatch")
        self._invocations = []
        self._receipts = {}
        self._closed = False

    def reserve(self) -> Invitation:
        if self._closed:
            raise RuntimeError("managed session is closed")
        response = _exchange(self.endpoint, {"operation": "reserve", "session": self._token})
        invitation = Invitation(self.endpoint, self.supervisor, response["invocation"], self._token, self.source)
        self._invocations.append(invitation)
        return invitation

    def _owned(self, invitation):
        if invitation not in self._invocations:
            raise ValueError("invocation is not owned by this session")

    def cancel(self, invitation: Invitation, *, timeout: float = 8) -> None:
        self._owned(invitation)
        _exchange(
            self.endpoint,
            {"operation": "cancel", "session": self._token, "invocation": invitation.invocation},
            timeout=timeout,
        )

    def receipt(self, invitation: Invitation, *, timeout: float = 8) -> dict:
        self._owned(invitation)
        if invitation.invocation in self._receipts:
            return self._receipts[invitation.invocation]
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CleanupIncomplete("native lifecycle receipt unavailable")
            try:
                response = _exchange(
                    self.endpoint,
                    {"operation": "receipt", "session": self._token, "invocation": invitation.invocation},
                    timeout=remaining,
                )
            except TimeoutError as error:
                raise CleanupIncomplete("native lifecycle receipt unavailable") from error
            receipt = response.get("receipt")
            if receipt is not None:
                if receipt.get("supervisor") != self.supervisor or receipt.get("invocation") != invitation.invocation:
                    raise CleanupIncomplete("native receipt identity mismatch")
                if receipt.get("source") != self.source:
                    raise CleanupIncomplete("native receipt source identity mismatch")
                self._receipts[invitation.invocation] = receipt
                return receipt
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))

    def close(self) -> None:
        if self._closed:
            return
        deadline = time.monotonic() + 8
        errors = []
        for invitation in self._invocations:
            try:
                self.cancel(invitation, timeout=min(1, max(0, deadline - time.monotonic())))
            except Exception as error:
                errors.append(f"cancel:{invitation.invocation}:{type(error).__name__}")
        for invitation in self._invocations:
            try:
                if not self.receipt(invitation, timeout=max(0, deadline - time.monotonic()))["cleanup_complete"]:
                    errors.append(f"incomplete:{invitation.invocation}")
            except Exception as error:
                errors.append(f"observe:{invitation.invocation}:{type(error).__name__}")
        if errors:
            raise CleanupIncomplete("managed session cleanup remains unresolved: " + ", ".join(errors))
        _exchange(self.endpoint, {"operation": "release", "session": self._token}, timeout=deadline - time.monotonic())
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _kind, primary, _traceback):
        try:
            self.close()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"Managed native cleanup also failed: {type(error).__name__}")


def setup(invitation: Invitation) -> None:
    """Public Chia setup hook; no subprocess is created in the worker."""
    if getattr(_state, "connection", None) is not None:
        raise RuntimeError("managed native invocation already active on this thread")
    active = active_source_identity()
    if active is not None and active != invitation.source:
        raise RuntimeError("managed native invitation differs from frozen worker source")
    fd = protocol.self_pidfd()
    connection = None
    try:
        connection = protocol.connect(invitation.endpoint)
        protocol.send(
            connection,
            {
                "operation": "attach",
                "session": invitation.session,
                "invocation": invitation.invocation,
                "host": protocol.host_identity(),
            },
            [fd],
        )
        response, attachments = protocol.receive(connection)
        protocol.close_fds(attachments)
        if response != {"state": "ready", "supervisor": invitation.supervisor, "source": invitation.source}:
            raise RuntimeError("managed guardian did not become ready on the selected supervisor")
        connection.settimeout(None)
        _state.connection = connection
        _state.invitation = invitation
    except BaseException:
        if connection is not None:
            connection.close()
        raise
    finally:
        os.close(fd)


def cleanup(invitation: Invitation) -> None:
    """Fallback hook closes only its own control channel; driver observes evidence."""
    if getattr(_state, "invitation", None) != invitation:
        return
    connection = getattr(_state, "connection", None)
    _state.connection = None
    _state.invitation = None
    if connection is not None:
        connection.close()


def run(argv: list[str], *, cwd: str | Path, env: dict[str, str] | None = None) -> int:
    """Run exactly one native command with original worker stdio and explicit env."""
    connection = getattr(_state, "connection", None)
    invitation = getattr(_state, "invitation", None)
    if connection is None or invitation is None:
        raise RuntimeError("native execution requires a managed Chia setup hook")
    try:
        protocol.send(
            connection,
            {"operation": "start", "argv": argv, "cwd": str(cwd), "env": dict(os.environ) if env is None else env},
            [0, 1, 2],
        )
        # Ray delivers cooperative task cancellation as a Python exception.
        # An unbounded native recvmsg can defer that exception indefinitely;
        # return to Python periodically without imposing a native runtime limit.
        connection.settimeout(0.1)
        while True:
            try:
                response, attachments = protocol.receive(connection)
                break
            except TimeoutError:
                continue
        protocol.close_fds(attachments)
        receipt = response.get("receipt") or {}
        if (
            response.get("state") != "finished"
            or receipt.get("supervisor") != invitation.supervisor
            or receipt.get("invocation") != invitation.invocation
            or receipt.get("source") != invitation.source
            or not receipt.get("cleanup_complete")
            or not (receipt.get("guardian") or {}).get("native_started")
        ):
            raise CleanupIncomplete("native command lacks complete managed lifecycle evidence")
        return int(receipt["guardian"]["returncode"])
    finally:
        cleanup(invitation)
