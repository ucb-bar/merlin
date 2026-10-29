"""How long one Spike timing observation needs, from the case's declared work and this run's own rate.

A timed observation runs the case twice on the functional engine (plain and annotated) and costs time
in proportion to the work the kernel does (measured offline: about 5 s plus 0.19 ms per accelerator
command). Each case has its own floor budget (the selection's ``per_observation_seconds``, else
``per_request_seconds``); this module lets a case whose predicted need exceeds that floor use what it
needs where its request has the time, and say what it needs where it does not, rather than start an
observation that is bound to be cut off:

* :func:`work` is the case's declared work -- its structural cycle floor (compute floor or ideal
  request service), already derived from the target facts in the case's structural roofline;
* :func:`record` adds each completed observation's ``(work, wall seconds)`` to a run-side ledger beside
  the joint campaign state (every grade child of the campaign shares it);
* :func:`need_seconds` fits ``seconds = setup + work x rate`` over those observations (the same
  fit and upper-quantile margin the gSIM lane uses) and predicts what one case needs; ``None`` until
  enough observations exist, and then the selected floor applies unchanged;
* :func:`protocol_ceiling` is the bound the selected count runtime itself keeps (its per-command wall
  bound for each of the two commands); a case needing more is reported, not started.

The selected ``per_observation_seconds`` (or ``per_request_seconds``) stays the floor. Nothing here
raises into a grade: an unreadable ledger is no evidence, never a guess.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

LEDGER = "joint_observation_rate.json"
#: Observations kept in the ledger (the newest); older ones describe older code and add nothing.
_KEEP = 2048
_MIN_OBSERVATIONS = 4
_QUANTILE = 0.95
#: Commands one count observation runs (plain functional, then annotated), each under the runtime's bound.
_COMMANDS = 2


def _ledger_path() -> Path | None:
    from .. import joint_control

    state = os.environ.get(joint_control.ENV, "").strip()
    return Path(state).with_name(LEDGER) if state else None


def work(structural) -> float | None:
    """The case's declared work in structural floor cycles, or ``None`` when the roofline states none."""
    from ..joint_gsim_measurement import structural_floor

    floor, _ = structural_floor(structural)
    return float(floor) if floor is not None and floor > 0 else None


def _read(path: Path | None) -> list:
    if path is None:
        return []
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    rows = document.get("observations") if isinstance(document, dict) else None
    return [r for r in rows or () if isinstance(r, dict)]


@contextmanager
def _locked(path: Path):
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        try:
            document = json.loads(raw) if raw.strip() else {}
        except ValueError:
            document = {}
        if not isinstance(document, dict) or not isinstance(document.get("observations"), list):
            document = {"version": 1, "observations": []}
        yield document
        handle.seek(0)
        handle.truncate()
        json.dump(document, handle, sort_keys=True)
        handle.flush()


def record(structural, observation, *, path: Path | None = None) -> None:
    """Add one completed observation's declared work and wall seconds to the run's ledger; never raises."""
    try:
        amount, seconds = work(structural), (observation or {}).get("wall_seconds")
        if (observation or {}).get("status") != "observed" or amount is None:
            return
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            return
        path = _ledger_path() if path is None else path
        if path is None:
            return
        with _locked(path) as document:
            document["observations"] = [*document["observations"], {"work": amount, "seconds": seconds}][-_KEEP:]
    except (OSError, ValueError, TypeError, KeyError):
        pass


def fit(*, path: Path | None = None) -> dict | None:
    """``{setup_s, seconds_per_work, margin, observations}`` over the ledger, or ``None`` with too few."""
    from ..joint_gsim_measurement import _affine

    rows = _read(_ledger_path() if path is None else path)
    points = sorted(
        (float(r["work"]), float(r["seconds"]))
        for r in rows
        if type(r.get("work")) in (int, float) and type(r.get("seconds")) in (int, float)
        and r["work"] > 0 and r["seconds"] > 0
    )
    if len(points) < _MIN_OBSERVATIONS:
        return None
    result = _affine(points, quantile=_QUANTILE)
    if result is None:
        return None
    setup, slope, margin = result
    return {"setup_s": setup, "seconds_per_work": slope, "margin": margin, "observations": len(points)}


def need_seconds(structural, *, rate: dict | None = None, path: Path | None = None) -> int | None:
    """Seconds one observation of a case with this roofline is predicted to need (upper margin)."""
    amount = work(structural)
    rate = fit(path=path) if rate is None else rate
    if amount is None or rate is None:
        return None
    return max(1, math.ceil(rate["margin"] * (rate["setup_s"] + amount * rate["seconds_per_work"])))


def protocol_ceiling(runtime) -> int | None:
    """Seconds the selected count runtime lets one observation run (its per-command bound per command)."""
    try:
        bound = runtime.record()["observation_resources"]["per_command_wall_seconds"]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return _COMMANDS * int(bound) if type(bound) in (int, float) and bound > 0 else None


def deferred_reason(need: int, available: float) -> str:
    """Why a case's observation did not run in this request, with what it needs."""
    return (
        f"observation budget too small (needed ~{need} s, {max(0, int(available))} s available here); "
        "the background timing lane observes it with a budget that fits"
    )


def coverage_limit(performance: dict) -> list[str]:
    """The timing model's own unresolved events of an UNKNOWN estimate (its coverage limit), deduplicated.

    Events the model reports as ``<part>:<event>`` repeat a bare event; only the bare names are kept."""
    if not isinstance(performance, dict) or performance.get("estimated_whole_soc_cycles") is not None:
        return []
    events = [str(e) for e in performance.get("unresolved_events") or () if isinstance(e, str) and e]
    bare = [e for e in events if ":" not in e]
    return list(dict.fromkeys(bare or events))


__all__ = [
    "LEDGER",
    "coverage_limit",
    "deferred_reason",
    "fit",
    "need_seconds",
    "protocol_ceiling",
    "record",
    "work",
]
