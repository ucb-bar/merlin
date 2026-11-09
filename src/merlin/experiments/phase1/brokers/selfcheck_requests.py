"""Durable self-check request leases, cancellation and broker stop conditions.

The dispatcher and its workers use the same protocol decisions across restarts.
Deadline translation stays in selfcheck_deadlines; this module only decides
whether an existing request is pending or must stop, without grading it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .selfcheck_deadlines import _request_deadline


def _pid_namespace() -> str:
    """The broker's Linux PID-namespace identity, comparable across a shared file channel."""
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def _requester_alive(request: dict) -> bool | None:
    """Whether the process waiting for this request is still the same process.

    ``None`` means a legacy request supplied no identity, so its historical deadline contract remains
    authoritative.  A protocol-v3 request with identity is a lease: once that exact PID incarnation is
    gone, replay can help nobody and may block a new client behind hours of full-suite work.
    """
    identity = request.get("requester")
    if not isinstance(identity, dict):
        # Protocol-v3 shims before the explicit start-tick lease still embedded their PID as the first
        # request-id field.  Absence is conclusive; presence is not (the PID may have been reused), so
        # retain the deadline contract in the ambiguous case.  This drains old dead-client requests on
        # the first upgraded broker restart without weakening compatibility for a possibly live client.
        if int(request.get("protocol", 1)) >= 3:
            prefix = str(request.get("request_id") or "").partition("_")[0]
            if prefix.isdecimal() and not Path(f"/proc/{prefix}").exists():
                return False
        return None
    requester_namespace = identity.get("pid_namespace")
    # A numeric PID and /proc start tick are only comparable within one PID namespace.  Returning False
    # for a sandbox-local PID looked up in the broker's host /proc discarded every live self-check.  A
    # foreign namespace (and protocol-v3 identities written before this field existed) retains the
    # request's durable deadline lease; only a same-namespace identity may be declared dead here.
    if not requester_namespace or requester_namespace != _pid_namespace():
        return None
    try:
        pid = int(identity["pid"])
        expected_start = int(identity["start_ticks"])
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        state, actual_start = tail[0], int(tail[19])
    except FileNotFoundError:
        return False
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return False
    return state != "Z" and actual_start == expected_start


def _detached(request: Path) -> bool:
    """Whether the request's client returned on purpose and left it to its deadline lease.

    A client that submits with a short wait budget and collects later with ``--attach`` exits while
    its request is still queued or running. Judged by that process alone, a same-namespace request
    would be dropped or aborted as abandoned the moment the client returned. The marker is written
    before the client exits, so its absence still means a client that died mid-wait.
    """
    rid = request.name[len("req_") : -len(".json")]
    return bool(rid) and (request.parent / f"detached_{rid}").exists()


def _pending_requests(ch: Path, seen: set[str], *, now: float | None = None) -> list[Path]:
    """Outstanding, live requests in arrival order.

    ``seen`` is process-local, while completion markers and deadlines survive a broker restart.  All
    three are needed: relying only on ``seen`` caused a restarted broker to rerun every historical
    full-corpus grade before it reached the request whose client was actually waiting.
    """
    now = time.time() if now is None else now
    pending: list[tuple[float, Path]] = []
    for req in ch.glob("req_*.json"):
        if req.name in seen:
            continue
        rid = req.stem[len("req_") :]
        if (ch / f"done_{rid}").exists() and (ch / f"resp_{rid}.json").exists():
            continue
        from ..joint_gsim_measurement import REQUEST_PREFIX as MEASUREMENT_PREFIX
        from ..joint_gsim_measurement import long_request

        if rid.startswith(MEASUREMENT_PREFIX) and long_request(rid, running=True):
            continue  # a long measurement outlived an earlier broker and is still being served
        try:
            body = json.loads(req.read_text(encoding="utf-8"))
            if _requester_alive(body) is False and not _detached(req):
                continue
            if now > _request_deadline(req):
                continue
            arrival = req.stat().st_mtime
        except FileNotFoundError:
            continue
        pending.append((arrival, req))
    return [request for _, request in sorted(pending, key=lambda row: row[0])]


def _abort_reason(
    ch: Path,
    orig_ppid: int,
    request: Path,
    deadline: float,
    *,
    request_stop: Path | None = None,
    survive_stop: bool = False,
) -> str | None:
    """Why the self-check in flight must stop now, or ``None`` to let it run.

    A CANCEL CANCELS. A client cancels by renaming its request away; the broker had already claimed
    it and ran to completion regardless, holding the one slot. Measured on one run: 4,609 seconds of
    oracle time spent on a request the agent stopped reading 37 minutes earlier, while four later
    requests waited behind it and were never served.

    ``survive_stop`` is set only for a long gSIM measurement the idle lane authorized
    (:func:`..joint_gsim_measurement.long_request`): the broker's STOP, its parent's exit and the
    dispatcher's stop then do not end it -- it runs to its own deadline, which is bounded by the
    campaign's end -- unless a process is waiting for a shared compute slot while none is free (final
    grading must never wait for a measurement's slot). Cancellation, an unreadable request and the
    deadline still stop it.
    """
    stop = _should_stop(ch, orig_ppid)
    if not stop and request_stop is not None and request_stop.exists():
        stop = "dispatcher stopped"
    if stop:
        if not survive_stop:
            return stop
        from ..joint_gsim_measurement import shared_pool_contended

        if shared_pool_contended():
            return f"{stop}; a shared compute slot is wanted while none is free"
    if not request.exists():
        return "the requester cancelled it"
    try:
        if _requester_alive(json.loads(request.read_bytes())) is False and not _detached(request):
            return "requester exited"
    except (OSError, ValueError, AttributeError):
        return "request changed or became unreadable"
    if time.monotonic() > deadline:
        return "timed out"
    return None


def _should_stop(ch: Path, orig_ppid: int) -> str | None:
    """Why the broker must exit now, or None to keep serving.

    Checked BEFORE each request and while one is in flight. Two conditions, both observed to matter:
    the driver's STOP sentinel (previously only tested between batches, so a ~10-minute self-check kept
    running after the round ended), and the death of the QA loop that started us — an orphaned broker
    was seen reparented to init, still spawning self-checks and RTL sims to answer requests whose
    channel no one was reading any more.
    """
    if (ch / "STOP").exists():
        return "STOP"
    if os.getppid() != orig_ppid:
        return "parent exited"
    return None
