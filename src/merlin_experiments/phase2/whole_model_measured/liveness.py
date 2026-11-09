"""Is a measured run still making progress?  A heartbeat it writes, and a verdict read from its records.

Infrastructure failures present as "nothing happened": a launcher killed by a closed pipe, a SIGSTOPped
agent, a board queue starved by solo repeats -- each looked like a slow loop for hours.  So a run says
what it is doing, and ``status`` says when that has stopped.

THE HEARTBEAT (``<run_dir>/heartbeat.json``, :data:`HEARTBEAT_SCHEMA`) is written by the launcher's own
process (:class:`Heartbeat`, ticked from the objective's poll and at every session start, at most once
per ``min_interval``):

* ``launcher`` -- ``pid``, its ``start_ticks`` (the kernel's start time of that pid, so a reused pid is
  never mistaken for the launcher), ``argv0`` and ``first_beat_at``;
* ``last_activity`` -- ``at`` (``YYYYmmddTHHMMSSZ``), ``epoch`` and ``what`` (``poll``, ``session 3``);
* ``last_measured`` -- the newest candidate measurement any of the run's stores holds (every attempt
  counted): ``at``, ``epoch``, ``package_sha256``, ``timing_status``, ``whole_window_cycles``;
* ``beats``, ``updated_at``.

THE VERDICT (:func:`assess`) is ``STOPPED`` when the run recorded a stop or an operator asked for one;
otherwise ``STALLED`` when the launcher is gone (its launch record's pid no longer runs the run, or the
heartbeat's pid is not the process that wrote it) or when no candidate was measured within
``stall_hours`` (counted from the run's launch when nothing was measured yet); otherwise ``LIVE``.  Ages
come from the records' own timestamps, never a file's modification time.

THE WATCHDOG records each stall once per episode in ``<run_dir>/liveness_events.jsonl``
(:data:`EVENT_SCHEMA`: ``event`` STALLED or RECOVERED, ``at``, ``reasons``, ``notify``) and, when
configured, runs a notify command (an argv; ``{run_dir}`` and ``{reason}`` are substituted, no shell).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from .identity import epoch_of, now, read_json, write_json_atomic

HEARTBEAT = "heartbeat.json"
HEARTBEAT_SCHEMA = "merlin.phase2.whole_model_measured.heartbeat.v1"
EVENTS = "liveness_events.jsonl"
EVENT_SCHEMA = "merlin.phase2.whole_model_measured.liveness_event.v1"
LIVE, STALLED, STOPPED = "LIVE", "STALLED", "STOPPED"
#: Hours without a measured candidate after which a run is STALLED (the dashboard's default too).
DEFAULT_STALL_HOURS = 6.0
#: The shortest interval between two heartbeat writes from one process.
DEFAULT_BEAT_SECONDS = 60.0
MEASURED_STATUSES = (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID)


def start_ticks(pid: int) -> str | None:
    """The kernel's start time of ``pid`` (``/proc/<pid>/stat`` field 22), or None when it is not running.
    A pid plus its start time names one process; a pid alone may have been reused."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rpartition(")")[2].split()
    except (OSError, ValueError):
        return None
    return fields[19] if len(fields) > 19 else None


def last_measured(roots: Iterable[Path]) -> dict[str, Any] | None:
    """The newest candidate measurement (MEASURED or MEASURED_INVALID) any of ``roots`` holds, any attempt."""
    from . import noise as NOISE

    newest: tuple[float, dict[str, Any]] | None = None
    for root in roots:
        for path in NOISE.result_paths(Path(root)):
            document = read_json(path) or {}
            if document.get("timing_status") not in MEASURED_STATUSES:
                continue
            epoch = epoch_of(document.get("finished_at"))
            if epoch is None or (newest is not None and epoch <= newest[0]):
                continue
            newest = (
                epoch,
                {
                    "at": document.get("finished_at"),
                    "epoch": epoch,
                    "package_sha256": document.get("package_sha256"),
                    "timing_status": document.get("timing_status"),
                    "whole_window_cycles": (document.get("verdict") or {}).get("whole_window_cycles"),
                },
            )
    return newest[1] if newest else None


class Heartbeat:
    """The launcher's heartbeat writer for one run (see the module doc)."""

    def __init__(
        self,
        run_dir: Path,
        *,
        stores: Callable[[], Sequence[Path]] | Sequence[Path] = (),
        min_interval: float = DEFAULT_BEAT_SECONDS,
        clock: Callable[[], float] = time.time,
        pid: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._stores = stores
        self.min_interval = float(min_interval)
        self.clock = clock
        self.pid = int(pid if pid is not None else os.getpid())
        self._last = 0.0

    def stores(self) -> list[Path]:
        found = self._stores() if callable(self._stores) else self._stores
        return [Path(p) for p in found or () if p]

    def tick(self, what: str, *, force: bool = False) -> dict[str, Any] | None:
        """Write the heartbeat when ``min_interval`` has passed (or ``force``); never raises -- a heartbeat
        that cannot be written must not stop the run it reports on (its absence is what ``assess`` reads)."""
        moment = self.clock()
        if not force and moment - self._last < self.min_interval:
            return None
        self._last = moment
        try:
            previous = read_json(self.run_dir / HEARTBEAT) or {}
            kept = previous.get("launcher") or {}
            ticks = start_ticks(self.pid)
            same = kept.get("pid") == self.pid and kept.get("start_ticks") == ticks
            launcher = (
                kept
                if same
                else {
                    "pid": self.pid,
                    "start_ticks": ticks,
                    "argv0": sys.argv[0] if sys.argv else None,
                    "first_beat_at": _stamp(moment),
                }
            )
            document = {
                "schema": HEARTBEAT_SCHEMA,
                "run_dir": str(self.run_dir),
                "launcher": launcher,
                "last_activity": {"at": _stamp(moment), "epoch": moment, "what": str(what)},
                "last_measured": last_measured(self.stores()),
                "beats": int(previous.get("beats") or 0) + 1 if same else 1,
                "updated_at": _stamp(moment),
            }
            write_json_atomic(self.run_dir / HEARTBEAT, document)
            return document
        except Exception:  # noqa: BLE001 -- see the docstring: a stale heartbeat is itself the signal
            return None


def _stamp(epoch: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch))


def _heartbeat_alive(heartbeat: Mapping[str, Any]) -> bool | None:
    launcher = heartbeat.get("launcher") or {}
    pid = launcher.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    ticks = start_ticks(pid)
    return ticks is not None and ticks == launcher.get("start_ticks")


def assess(
    run_dir: Path,
    *,
    stall_hours: float = DEFAULT_STALL_HOURS,
    stores: Iterable[Path] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """The run's liveness verdict (see the module doc), from its own records only."""
    from . import launch as LAUNCH
    from . import sessions as SES

    run_dir = Path(run_dir)
    moment = clock()
    heartbeat = read_json(run_dir / HEARTBEAT) or {}
    launch = LAUNCH.record(run_dir) or {}
    if stores is None:
        stores = [Path(p) for p in ((read_json(run_dir / "resumed_seed.json") or {}).get("store_roots") or {}).values()]
    measured = last_measured(stores) or heartbeat.get("last_measured")
    stopped = (read_json(run_dir / "stage" / "sessions.json") or {}).get("stopped")
    requested = SES.operator_stop(run_dir / SES.OPERATOR_STOP_FILE)
    document: dict[str, Any] = {
        "state": LIVE,
        "reasons": [],
        "stall_hours": float(stall_hours),
        "last_measured": measured,
        "last_activity": heartbeat.get("last_activity"),
        "heartbeat": {"pid": (heartbeat.get("launcher") or {}).get("pid"), "updated_at": heartbeat.get("updated_at")}
        if heartbeat
        else None,
    }
    if isinstance(stopped, Mapping) and stopped.get("kind"):
        document.update(state=STOPPED, reasons=[f"the run stopped ({stopped.get('kind')}): {stopped.get('reason')}"])
        return document
    if requested is not None and LAUNCH.launcher_alive(run_dir) is not True:
        document.update(state=STOPPED, reasons=[f"an operator asked the run to stop: {requested.get('reason')}"])
        return document
    alive = LAUNCH.launcher_alive(run_dir)
    if alive is None and heartbeat:
        alive = _heartbeat_alive(heartbeat)
    document["launcher_alive"] = alive
    reasons = []
    if alive is False:
        pid = launch.get("pid") or (heartbeat.get("launcher") or {}).get("pid")
        reasons.append(f"the launcher (pid {pid}) is not running and the run recorded no stop")
    since = (measured or {}).get("epoch")
    basis = "the last measured candidate"
    if since is None:
        since = epoch_of(launch.get("started_at")) or epoch_of((heartbeat.get("launcher") or {}).get("first_beat_at"))
        basis = "the launch (no candidate measured since)"
    if since is not None:
        hours = (moment - float(since)) / 3600.0
        document["hours_since_measured"] = round(hours, 2)
        if hours >= float(stall_hours):
            reasons.append(f"{hours:.1f} h since {basis}, at or over the {float(stall_hours):g} h stall threshold")
    if reasons:
        document.update(state=STALLED, reasons=reasons)
    return document


def notify(command: Sequence[str], *, run_dir: Path, reasons: Sequence[str], timeout: float = 60.0) -> dict[str, Any]:
    """Run the configured notify ``command`` (an argv, ``{run_dir}``/``{reason}`` substituted; no shell)."""
    reason = "; ".join(reasons)
    argv = [str(token).replace("{run_dir}", str(run_dir)).replace("{reason}", reason) for token in command]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return {"argv": argv, "returncode": done.returncode, "output": (done.stdout + done.stderr)[-500:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def events(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / EVENTS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_transition(
    run_dir: Path, verdict: Mapping[str, Any], *, notify_command: Sequence[str] | None = None
) -> dict[str, Any] | None:
    """Append a STALLED event when the run has just stalled (or RECOVERED when it has just recovered), once
    per episode, running the notify command on a stall.  Returns the event written, or None."""
    run_dir = Path(run_dir)
    history = events(run_dir)
    last = history[-1]["event"] if history else None
    state = verdict.get("state")
    if state == STALLED and last != STALLED:
        event: dict[str, Any] = {
            "schema": EVENT_SCHEMA,
            "event": STALLED,
            "at": now(),
            "reasons": list(verdict.get("reasons") or ()),
            "last_measured": verdict.get("last_measured"),
        }
        if notify_command:
            event["notify"] = notify(notify_command, run_dir=run_dir, reasons=event["reasons"])
    elif state == LIVE and last == STALLED:
        event = {"schema": EVENT_SCHEMA, "event": "RECOVERED", "at": now(), "reasons": []}
    else:
        return None
    with (run_dir / EVENTS).open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    return event


__all__ = [
    "DEFAULT_STALL_HOURS",
    "EVENTS",
    "EVENT_SCHEMA",
    "HEARTBEAT",
    "HEARTBEAT_SCHEMA",
    "Heartbeat",
    "LIVE",
    "STALLED",
    "STOPPED",
    "assess",
    "events",
    "last_measured",
    "notify",
    "record_transition",
    "start_ticks",
]
