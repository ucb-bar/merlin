"""When a measured run stops, and what a session actually ran: plateau, quota, model and infra evidence.

A long measured run ends on EVIDENCE, never on a round count: a plateau in wall time (no eligible
improvement for ``plateau_hours`` over at least ``plateau_min_sessions`` sessions), a certified best
reaching the bar when the run asked for that, or a streak of infrastructure refusals.  Each of these
was once read wrongly:

* **The plateau lives in the STORE** (``plateau.json``), not the launcher, so a relaunch keeps the
  time since the last improvement instead of silently starting again from zero.  Its clock restarts
  only on an explicit reset that states its reason.
* **An exhausted authoring account is not a plateau.**  A round the CLI refused before it started a
  turn looks, to a transcript audit, exactly like a parser false positive; 61 such rounds once ended
  a run "on evidence".  :func:`account_exhausted` reads the driver's own summary, and the session that
  never ran is ABANDONED (with the account's message) so it never counts.
* **The model that ran is verified**, not assumed: an alias can silently fall through to a driver
  default.  :func:`verify_round_model` refuses a round whose own transcript reports another model.
* **An infra streak stops the run** (:func:`circuit_breaker_check`) instead of burning rounds against a
  broken harness, and says so in a file beside the stage.
* **An operator stops a run at a session boundary** (:func:`request_stop`): a request file in the run
  directory, read at the top of every session before anything of it starts.  Signalling the launcher
  instead interrupts a round mid-flight, and a frozen or killed round is recorded as a failure, not as
  a stop (on the old line a SIGSTOP that outlasted its round timeout ended a run outright).

:func:`run_sessions` composes these around a declared ``run_round`` callable (the authoring driver),
so the session loop's stopping and accounting rules are one tested owner whatever drives the rounds.
"""

from __future__ import annotations

import getpass
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import gates as G

PLATEAU_FILE = "plateau.json"
PLATEAU_SCHEMA = "merlin_whole_model_plateau_v1"
SEQUENCE_SCHEMA = "merlin.phase2.whole_model_measured.sessions.v1"
INFRA_CIRCUIT_BREAKER_LIMIT = 3
INFRA_ACCOUNT_EXHAUSTED = "infra_account_exhausted"
#: A round's status when its driver raised (a killed agent, a crashed driver).
ROUND_FAILED = "round_failed"
#: The round statuses that ended with a clean authoring audit.
CLEAN_ROUND_STATUSES = ("authored",)
#: (count, seconds): that many consecutive failed rounds, each shorter than that, is a crash loop.
CRASH_LOOP = (5, 60.0)

#: Substrings of an authoring driver's own reported error that mean the PROVIDER ACCOUNT declined to
#: start the turn at all.  Matched case-insensitively against the driver's own ``errors`` list, never
#: against transcript content the agent could have produced.
ACCOUNT_EXHAUSTION_MARKERS = ("hit your weekly limit", "hit your usage limit", "hit your rate limit")

#: An operator's request that a run stop at its NEXT session boundary, in the run directory.
OPERATOR_STOP_FILE = "stop_requested.json"
OPERATOR_STOP_SCHEMA = "merlin.phase2.whole_model_measured.stop_request.v1"
#: The stop kind an operator's request records (never evidence: the run did not converge).
OPERATOR_STOP = "operator"


def _write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    staging.replace(path)


def _stamp(epoch: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch))


# --------------------------------------------------------------- the plateau record
def plateau_rule(config: Mapping[str, Any] | None) -> dict[str, float] | None:
    """``{hours, min_sessions}`` when the objective config declares a wall-time plateau, else None."""
    config = config or {}
    if not config.get("plateau_hours"):
        return None
    return {"hours": float(config["plateau_hours"]), "min_sessions": int(config.get("plateau_min_sessions") or 0)}


def plateau_path(objective: Any) -> Path | None:
    root = getattr(getattr(objective, "screen", None), "root", None)
    return Path(root) / PLATEAU_FILE if root else None


def _load(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        record = None
    return record if isinstance(record, dict) else {"schema": PLATEAU_SCHEMA, "trace": []}


def _is_reset(row: Mapping[str, Any]) -> bool:
    """An EXPLICIT reset: a ``reset`` entry that states its reason."""
    return bool(row.get("reset")) and bool(str(row.get("reason") or "").strip())


def _is_abandoned(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("abandoned") or "").strip())


def plateau_anchor(trace: Sequence[Mapping[str, Any]]) -> int | None:
    """Index of the entry the plateau counts from: the later of the last explicit reset and the last
    session at which the eligible best got LOWER than every earlier reading."""
    lowest, at = None, None
    for index, row in enumerate(trace):
        if row.get("reset"):
            if _is_reset(row):
                at = index
            continue
        if _is_abandoned(row):
            continue
        cycles = row.get("cycles")
        if cycles is None:
            continue
        if lowest is None:
            lowest = int(cycles)
            if at is None:
                at = index
        elif int(cycles) < lowest:
            lowest, at = int(cycles), index
    return at


def reset_plateau(path: str | Path, *, reason: str, now: float | None = None) -> dict[str, Any]:
    """Append an explicit reset: the plateau counts from here.  ``reason`` is required."""
    if not str(reason or "").strip():
        raise ValueError("a plateau reset states its reason")
    path = Path(path)
    now = time.time() if now is None else now
    record = _load(path)
    entry = {"reset": True, "reset_at": _stamp(now), "epoch": now, "reason": str(reason).strip()}
    record["trace"].append(entry)
    record["trace"].sort(key=lambda row: float(row.get("epoch") or 0.0))
    _write(path, record)
    return entry


def abandon_session(path: str | Path, *, run: str, session: int, reason: str) -> dict[str, Any]:
    """Mark one recorded session as not run, so it does not count toward the plateau."""
    if not str(reason or "").strip():
        raise ValueError("an abandoned session states its reason")
    path = Path(path)
    record = _load(path)
    matches = [r for r in record["trace"] if not r.get("reset") and r.get("run") == run and r.get("session") == session]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} recorded session(s) match run {run!r} session {session}")
    matches[0]["abandoned"] = str(reason).strip()
    _write(path, record)
    return matches[0]


def backfill_plateau(path: str | Path, rows: Sequence[Mapping[str, Any]], *, reason: str) -> list[dict[str, Any]]:
    """Insert historical session rows (``{epoch, run, session, cycles, package_sha256}``) that a run
    recorded elsewhere, each marked ``backfilled`` with ``reason``; an already-recorded (run, session)
    is never duplicated."""
    if not str(reason or "").strip():
        raise ValueError("a plateau backfill states its reason")
    path = Path(path)
    record = _load(path)
    have = {(r.get("run"), r.get("session")) for r in record["trace"] if not r.get("reset")}
    added = []
    for row in rows:
        if "epoch" not in row or "run" not in row or "session" not in row:
            raise ValueError(f"a backfilled row needs epoch, run and session: {dict(row)}")
        if (row["run"], row["session"]) in have:
            continue
        entry = {**dict(row), "at": _stamp(float(row["epoch"])), "backfilled": str(reason).strip()}
        record["trace"].append(entry)
        added.append(entry)
    record["trace"].sort(key=lambda r: float(r.get("epoch") or 0.0))
    _write(path, record)
    return added


def best_now(objective: Any) -> tuple[str | None, int | None]:
    try:
        objective.poll()
        best = (objective.summary() or {}).get("best") or {}
    except Exception:  # noqa: BLE001 -- an unreadable objective never stops a run
        return None, None
    return best.get("package_sha256"), best.get("screen_whole_window_cycles")


def record_session(
    objective: Any,
    *,
    session: int,
    run: str,
    now: float | None = None,
    driver: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Append this session's start to the store's plateau record; return the stop evidence, or None.

    ``driver``/``model`` are kept on the row (the model that ACTUALLY ran, see
    :func:`verify_round_model`) so a trace spanning a switch is never read as one continuous search."""
    rule = plateau_rule(getattr(objective, "config", None))
    path = plateau_path(objective)
    if rule is None or path is None:
        return None
    now = time.time() if now is None else now
    digest, cycles = best_now(objective)
    record = _load(path)
    record["rule"] = rule
    record["trace"].append(
        {
            "at": _stamp(now),
            "epoch": now,
            "run": run,
            "session": session,
            "package_sha256": digest,
            "cycles": cycles,
            **({"driver": str(driver)} if driver else {}),
            **({"model": str(model)} if model else {}),
        }
    )
    trace = record["trace"]
    index = plateau_anchor(trace)
    index = index if index is not None else 0
    since = trace[index]
    hours = (now - float(since["epoch"])) / 3600.0
    sessions = sum(1 for row in trace[index + 1 :] if not row.get("reset") and not _is_abandoned(row))
    record["last_improvement"] = {**since, "hours_since": round(hours, 3), "sessions_since": sessions}
    _write(path, record)
    if cycles is None or hours < rule["hours"] or sessions < rule["min_sessions"]:
        return None
    return {
        "reason": f"plateau: no eligible improvement for {hours:.1f} h of wall time over {sessions} sessions "
        f"(rule: {rule['hours']:g} h and at least {rule['min_sessions']} sessions)",
        "kind": "plateau",
        "rule": rule,
        "hours_since_improvement": round(hours, 3),
        "sessions_since_improvement": sessions,
        "plateau_record": str(path),
        "counted_from": "an explicit reset" if since.get("reset") else "the last eligible improvement",
    }


# --------------------------------------------------------------- the operator's stop request
def request_stop(
    run_dir: str | Path, *, why: str, operator: str | None = None, now: float | None = None
) -> dict[str, Any]:
    """Ask the run in ``run_dir`` to stop at its next session boundary; return the request on disk.

    The FIRST request stands: a repeated one is returned unchanged, so the reason a run stopped is the
    reason it was first asked to.  ``why`` is required."""
    if not str(why or "").strip():
        raise ValueError("a stop request states why")
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"no run directory at {run_dir}")
    path = run_dir / OPERATOR_STOP_FILE
    if path.is_file() and not path.is_symlink():
        existing = _load_request(path)
        return {**existing, "already_requested": True}
    now = time.time() if now is None else now
    document = {
        "schema": OPERATOR_STOP_SCHEMA,
        "requested_at": _stamp(now),
        "epoch": now,
        "operator": operator or getpass.getuser(),
        "why": str(why).strip(),
    }
    _write(path, document)
    return document


def _load_request(path: Path) -> dict[str, Any]:
    try:
        request = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {"why": "unreadable request file"}
    return request if isinstance(request, dict) else {"why": "unreadable request file", "value": request}


def operator_stop(request: str | Path | None) -> dict[str, Any] | None:
    """Stop evidence for the operator's request file ``request``, or None when no request was made.

    An unreadable request still stops the run: someone wrote it to stop the run."""
    if request is None:
        return None
    path = Path(request)
    if not (path.is_file() or path.is_symlink()):
        return None
    document = _load_request(path) if not path.is_symlink() else {"why": "the request file is a link"}
    return {
        "kind": OPERATOR_STOP,
        "reason": f"an operator asked the run to stop at its next session boundary: {document.get('why')}",
        "request": document,
        "request_path": str(path),
    }


# --------------------------------------------------------------- what a round actually ran
def account_exhausted(summaries: Sequence[Path]) -> str | None:
    """The driver's own account-exhaustion message among ``summaries`` (its per-round summary files),
    or None."""
    for path in summaries:
        if Path(path).is_symlink() or not Path(path).is_file():
            continue
        try:
            summary = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(summary, Mapping):
            continue
        for message in summary.get("errors") or ():
            if isinstance(message, str) and any(m in message.lower() for m in ACCOUNT_EXHAUSTION_MARKERS):
                return message
    return None


def verify_round_model(transcript: Path, expected_model: str) -> str | None:
    """A refusal when this round's own transcript REPORTS having run a model other than
    ``expected_model``; None when it agrees or reports nothing checkable (only a positive mismatch
    refuses)."""
    try:
        event = json.loads(Path(transcript).read_text(encoding="utf-8", errors="replace").splitlines()[0])
    except (OSError, IndexError, ValueError):
        return None
    if not isinstance(event, Mapping):
        return None
    ran = str(event.get("model") or "")
    if not ran or ran == expected_model:
        return None
    return f"round ran model {ran!r}, not the requested {expected_model!r} (transcript {transcript})"


def circuit_breaker_check(objective: Any, stage_root: Path, *, limit: int = INFRA_CIRCUIT_BREAKER_LIMIT) -> dict | None:
    """Stop evidence when the screen's latest ``limit`` candidates were each refused for an
    infrastructure reason; the alert is written beside ``stage_root`` so it is visible without the store."""
    screen = getattr(objective, "screen", None)
    if screen is None or not hasattr(screen, "history"):
        return None
    reason = G.infra_circuit_breaker(screen.history(), limit=limit)
    if reason is None:
        return None
    alert = {
        "schema": "merlin_whole_model_infra_circuit_breaker_v1",
        "at": _stamp(time.time()),
        "reason": reason,
        "limit": limit,
    }
    Path(stage_root).mkdir(parents=True, exist_ok=True)
    (Path(stage_root) / "infra_circuit_breaker.json").write_text(json.dumps(alert, indent=1) + "\n", encoding="utf-8")
    return {"reason": reason, "kind": "infra_circuit_breaker"}


def bar_reached(objective: Any) -> dict[str, Any] | None:
    """Stop evidence when the run asked to stop at the bar and a CERTIFIED best reached it."""
    config = getattr(objective, "config", None) or {}
    if not config.get("stop_at_bar", True):
        return None
    try:
        best = (objective.summary() or {}).get("best") or {}
    except Exception:  # noqa: BLE001
        return None
    gap = best.get("screen_gap_cycles")
    if isinstance(gap, int) and gap <= 0 and (best.get("certification") or {}).get("state") == "certified":
        return {"reason": "a certified best reached the screening machine's bar", "kind": "bar", "best": best}
    return None


# --------------------------------------------------------------- the session loop
def run_sessions(
    objective: Any,
    *,
    run_round: Callable[..., Mapping[str, Any]],
    stage_root: Path,
    run: str,
    max_sessions: int,
    total_seconds: float,
    driver: str | None = None,
    model: str | None = None,
    clock: Callable[[], float] = time.time,
    crash_loop: tuple[int, float] = CRASH_LOOP,
    stop_request: Path | None = None,
    first_session: int = 1,
) -> dict[str, Any]:
    """Run authoring sessions until evidence, budget or an operator stops them, and record why.

    A round that ends WITHOUT a clean authoring audit -- the agent timed out, was killed, or the
    driver raised -- is recorded and the next session starts: one bad round is never the run's end
    (it once ended a live loop overnight).  Only a CRASH LOOP stops it: ``crash_loop = (count,
    seconds)`` consecutive failed rounds, each shorter than ``seconds``, which no next session fixes.

    ``run_round(session=, stage_root=)`` drives one authoring session and returns ``{"status",
    "transcript"?, "summaries"?}``.  Before each session: the operator's ``stop_request`` file (read
    FIRST, so a stop starts nothing of the session -- not even its plateau row), the stagnation mark,
    the circuit breaker, the plateau record and the bar.  After it: an exhausted account abandons the
    session it recorded and stops; a transcript that ran another model is refused and stops.

    ``first_session`` > 1 continues a run a previous launcher of the SAME run directory started (see
    :func:`.rounds.next_session`): ``max_sessions`` still bounds the run's sessions, not this launch's."""
    stage_root = Path(stage_root)
    started = clock()
    rows: list[dict[str, Any]] = []
    stop: dict[str, Any] | None = None
    consecutive_brief_failures = 0
    for session in range(max(1, int(first_session)), int(max_sessions) + 1):
        stop = operator_stop(stop_request)
        if stop is not None:
            break  # A CLEAN STOP AT A SESSION BOUNDARY: the last session finished and was recorded.
        if clock() - started >= float(total_seconds):
            stop = {"reason": "the declared authoring budget is spent", "kind": "budget"}
            break
        mark = getattr(objective, "mark_session", None)
        if callable(mark):
            mark()
        stop = circuit_breaker_check(objective, stage_root)
        if stop is None:
            stop = record_session(objective, session=session, run=run, now=clock(), driver=driver, model=model)
        if stop is None:
            stop = bar_reached(objective)
        if stop is not None:
            break
        round_started = clock()
        try:
            outcome = dict(run_round(session=session, stage_root=stage_root) or {})
        except Exception as exc:  # noqa: BLE001 -- a killed or crashed round is recorded, never the run's end
            outcome = {"status": ROUND_FAILED, "failure": f"{type(exc).__name__}: {str(exc)[:500]}"}
        row: dict[str, Any] = {"session": session, "status": outcome.get("status"), "at": _stamp(clock())}
        if outcome.get("failure"):
            row["failure"] = outcome["failure"]
        failed = outcome.get("status") not in CLEAN_ROUND_STATUSES
        brief = clock() - round_started < crash_loop[1]
        consecutive_brief_failures = consecutive_brief_failures + 1 if (failed and brief) else 0
        exhausted = account_exhausted([Path(p) for p in outcome.get("summaries") or ()])
        if exhausted:
            path = plateau_path(objective)
            if path is not None and path.is_file():
                try:
                    abandon_session(path, run=run, session=session, reason=f"{INFRA_ACCOUNT_EXHAUSTED}: {exhausted}")
                except ValueError:
                    pass  # no plateau row for this session (no plateau rule): nothing to exclude
            row["status"] = INFRA_ACCOUNT_EXHAUSTED
            rows.append(row)
            stop = {"reason": f"{INFRA_ACCOUNT_EXHAUSTED}: {exhausted}", "kind": INFRA_ACCOUNT_EXHAUSTED}
            break
        if model and outcome.get("transcript"):
            mismatch = verify_round_model(Path(outcome["transcript"]), model)
            if mismatch:
                row.update(status="refused", refusal=mismatch)
                rows.append(row)
                stop = {"reason": mismatch, "kind": "model_mismatch"}
                break
        rows.append(row)
        if consecutive_brief_failures >= crash_loop[0]:
            stop = {
                "reason": f"{consecutive_brief_failures} consecutive round(s) failed in under {crash_loop[1]:g} s each "
                f"(latest: {row.get('failure') or row.get('status')}); a crash loop, not a search",
                "kind": "crash_loop",
            }
            break
    else:
        stop = {"reason": "the declared session limit is reached", "kind": "budget"}
    document = {
        "schema": SEQUENCE_SCHEMA,
        "run": run,
        "driver": driver,
        "model": model,
        "sessions": rows,
        "stopped": stop,
        "stopped_on_evidence": bool(stop and stop.get("kind") in ("plateau", "bar")),
    }
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "sessions.json").write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return document


__all__ = [
    "ACCOUNT_EXHAUSTION_MARKERS",
    "INFRA_ACCOUNT_EXHAUSTED",
    "OPERATOR_STOP",
    "OPERATOR_STOP_FILE",
    "PLATEAU_FILE",
    "abandon_session",
    "account_exhausted",
    "backfill_plateau",
    "bar_reached",
    "circuit_breaker_check",
    "operator_stop",
    "plateau_anchor",
    "record_session",
    "request_stop",
    "reset_plateau",
    "run_sessions",
    "verify_round_model",
]
