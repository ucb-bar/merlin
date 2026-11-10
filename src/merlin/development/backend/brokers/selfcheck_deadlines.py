"""Translate durable request deadlines and enforce the selected campaign boundary."""

from __future__ import annotations

import json
import time
from pathlib import Path

COMMON_ACTIONS = frozenset({"common-transform", "common-check"})


def _request_deadline(req: Path) -> float:
    """Client deadline for protocol-v2 and legacy requests.

    Legacy shims did not put their deadline in the payload, but their wait contract has always been
    ``request mtime + timeout + 240 seconds``.  Recovering it here preserves those clients while making
    their abandoned files finite work rather than a permanent restart backlog.
    """
    try:
        body = json.loads(req.read_text(encoding="utf-8"))
        explicit = body.get("deadline_unix_ns")
        if explicit is not None:
            return int(explicit) / 1_000_000_000
        timeout = max(0, int(body.get("timeout", 1800)))
    except Exception:
        timeout = 1800
    return req.stat().st_mtime + timeout + 240


def _child_deadline_monotonic(
    request: dict, *, now_wall: float | None = None, now_monotonic: float | None = None
) -> float:
    """Translate the client's durable wall-clock deadline onto this process's monotonic clock.

    The request deadline accounts for the whole suite.  Reconstructing a shorter
    ``per_capsule_timeout + 180`` deadline here used to kill full-corpus checks while their later worker
    waves were still healthy.
    """
    wall = time.time() if now_wall is None else now_wall
    monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    explicit = request.get("deadline_unix_ns")
    if explicit is not None:
        deadline = monotonic + max(0.0, int(explicit) / 1_000_000_000 - wall)
    else:
        deadline = monotonic + max(0, int(request.get("timeout", 1800))) + 240
    # A conservative suite-sized client budget cannot outlive its selected
    # campaign. This also bounds setup before the evaluator process exists.
    from merlin.development.backend import joint_control

    campaign = joint_control.selected()
    # Tool readiness exercises the common transformation broker before author
    # launch. Its client deadline still applies, but no campaign clock exists
    # yet. This exception cannot admit capsule evaluation before launch.
    readiness = (
        campaign is not None
        and "started_at_epoch" in campaign
        and campaign["started_at_epoch"] is None
        and request.get("action") in COMMON_ACTIONS
    )
    if campaign is not None and not readiness:
        deadline = min(deadline, monotonic + joint_control.remaining(now=wall))
    return deadline
