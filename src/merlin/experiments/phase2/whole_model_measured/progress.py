"""What a measured run is doing, read from its own records: one status document and a follow loop.

:func:`run_status` gathers, read-only, everything an operator used to piece together from scratch
scripts: what the run is (``run.json``, ``resumed_seed.json``), whether its launcher is alive (its own
``launch.json``), whether a stop was asked for and whether it stopped (``stop_requested.json``,
``stage/sessions.json``), each round's record and each round still open, the stores' job states and
holds (a disk hold, a board outage), the plateau clock, and the objective's bar, best and recent
history.  ``poll=True`` is the one side effect, asked for by name: it advances the objective exactly
as the run's own loop does (dispatching pending jobs, promoting a best), for a run whose launcher is
gone but whose measurements should still land.

:func:`follow` prints one line per change -- a job's state, a new best, a round's end, the launcher's
exit, a stop request, a hold -- until the run is over or a deadline passes.  Nothing here reads a
process name or a file's age: liveness is the launch record's pid with a command line naming the run.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import batch as BATCH
from . import capabilities as CAP
from . import jobs as J
from . import launch as LAUNCH
from . import liveness as LIVE
from . import rounds as RND
from . import service as SERVICE
from . import sessions as SES
from .identity import read_json

STATUS_SCHEMA = "merlin.phase2.whole_model_measured.status.v1"
#: The job states that mean a measurement is still owed.
OPEN_STATES = (J.PENDING, J.RUNNING, J.SCREENING, J.BOARD)


def _rounds(stage: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rounds_dir = stage / "rounds"
    ended, opened = [], []
    if not rounds_dir.is_dir():
        return ended, opened
    for path in sorted(rounds_dir.glob(f"round_*{RND.ROUND_SUFFIX}")):
        record = read_json(path) or {}
        ended.append(
            {
                "round": record.get("round"),
                "status": record.get("status"),
                "why": str(record.get("why") or "")[:240] or None,
                "agent_exit_code": record.get("agent_exit_code"),
                "carried_forward": record.get("carried_forward"),
                "requested": len(record.get("requested") or ()),
                "final_package_sha256": record.get("final_package_sha256"),
                "final_timing_status": record.get("final_timing_status"),
            }
        )
    for path in sorted(rounds_dir.glob(f"round_*{RND.OPEN_SUFFIX}")):
        marker = read_json(path) or {}
        opened.append(
            {
                "round": marker.get("round"),
                "started_at": marker.get("started_at"),
                "pid": marker.get("pid"),
                "requested": len(marker.get("requested") or ()),
            }
        )
    return ended, opened


def store_state(root: Path) -> dict[str, Any]:
    """One store, read from disk: its job states, its holds and its plateau clock."""
    root = Path(root)
    states: dict[str, int] = {}
    jobs: dict[str, str] = {}
    for path in sorted(root.glob("*/job.json")) if root.is_dir() else ():
        job = read_json(path) or {}
        state = str(job.get("state"))
        states[state] = states.get(state, 0) + 1
        jobs[path.parent.name] = state
    plateau = (read_json(root / SES.PLATEAU_FILE) or {}).get("last_improvement")
    return {
        "root": str(root),
        "jobs": states,
        "job_states": jobs,
        "disk_hold": read_json(root / SERVICE.DISK_HOLD),
        "board_outage": BATCH.board_outage(root) if root.is_dir() else None,
        "control_preflight": read_json(root / BATCH.CONTROL_PREFLIGHT),
        "plateau": plateau,
    }


def _free(path: Path) -> int | None:
    try:
        return SERVICE.build_free_bytes(path)
    except OSError:
        return None


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "package_sha256",
        "replicate",
        "label",
        "attribution",
        "state",
        "timing_status",
        "objective_cycles",
        "package_groups",
        "package_priced_share",
        "eligible",
    )
    out = {k: row.get(k) for k in keys}
    for key in ("refusal", "notice"):
        if row.get(key):
            out[key] = str(row[key])[:200]
    return out


def run_status(
    run_dir: Path,
    *,
    objective: Any = None,
    poll: bool = False,
    history: int = 10,
    stall_hours: float | None = None,
) -> dict[str, Any]:
    """The run's status document (see the module docstring).  ``objective`` is the run's objective when
    the caller has one; without it the stores are still read, and the objective's view is omitted.
    ``liveness`` is :func:`.liveness.assess`: STALLED when the launcher is gone or no candidate was
    measured within ``stall_hours`` (default :data:`.liveness.DEFAULT_STALL_HOURS`)."""
    from . import cli as CLI

    run_dir = Path(run_dir)
    record = read_json(run_dir / "run.json") or {}
    seed = read_json(run_dir / "resumed_seed.json") or {}
    stage = run_dir / "stage"
    ended, opened = _rounds(stage)
    sessions = read_json(stage / "sessions.json") or {}
    request = SES.operator_stop(run_dir / SES.OPERATOR_STOP_FILE)
    launch = LAUNCH.record(run_dir) or {}
    roots = {name: Path(path) for name, path in (seed.get("store_roots") or {}).items()}
    document: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "run_dir": str(run_dir),
        "target": record.get("target"),
        "method": record.get("method"),
        "prohibited_instruction_roles": record.get("prohibited_instruction_roles"),
        "seed": {
            "package_sha256": seed.get("seed_package_sha256"),
            "lineage_kind": seed.get("lineage_kind"),
            "resumed_from_run": seed.get("resumed_from_run"),
        },
        "launcher": {
            "pid": launch.get("pid"),
            "alive": LAUNCH.launcher_alive(run_dir),
            "started_at": launch.get("started_at"),
            "log": launch.get("log"),
        },
        "stop_requested": (request or {}).get("request"),
        "machine_warnings": list((read_json(run_dir / CAP.RECORD) or {}).get("warnings") or ()),
        "stopped": sessions.get("stopped"),
        "rounds": ended,
        "open_rounds": opened,
        "stores": {name: store_state(root) for name, root in roots.items()},
        "liveness": LIVE.assess(
            run_dir,
            stall_hours=LIVE.DEFAULT_STALL_HOURS if stall_hours is None else float(stall_hours),
            stores=list(roots.values()),
        ),
        "resumed_into": CLI.successors(run_dir),
        "disk_free_bytes": {
            "store": _free(roots["screen"]) if "screen" in roots else None,
            "tmpdir": _free(Path(os.environ.get("TMPDIR") or tempfile.gettempdir())),
        },
    }
    if objective is not None:
        if poll:
            objective.poll()
        summary = objective.summary()
        document["objective"] = {
            "bar": summary.get("bar"),
            "best": summary.get("best"),
            "screen_basis": summary.get("screen_basis"),
            "board": summary.get("board"),
            "noise": summary.get("noise"),
            "coverage_floor": summary.get("coverage_floor"),
            "history": [_compact(row) for row in (summary.get("history") or [])[-int(history) :]],
        }
    for store in document["stores"].values():
        store.pop("job_states")
    return document


def _authored(authored: Mapping[str, Any]) -> str:
    share = authored.get("priced_share")
    return f"{authored.get('groups_answered')}/{authored.get('groups_total')} groups, " + (
        "-" if share is None else f"{100 * float(share):.1f}%"
    )


def _cycles(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def format_status(document: Mapping[str, Any]) -> str:
    """The status document as a few lines for a terminal."""
    lines = [f"{document['run_dir']}", f"  {document.get('target')} / {document.get('method')}"]
    liveness = document.get("liveness") or {}
    if liveness:
        measured = (liveness.get("last_measured") or {}).get("at") or "never"
        lines.append(
            f"  {liveness.get('state')}: last measured candidate {measured}"
            + (
                f" ({liveness['hours_since_measured']} h ago)"
                if liveness.get("hours_since_measured") is not None
                else ""
            )
            + "".join(f"\n    {reason}" for reason in liveness.get("reasons") or ())
        )
    for warning in document.get("machine_warnings") or ():
        lines.append(f"  MACHINE: {warning}")
    launcher = document.get("launcher") or {}
    alive = {True: "alive", False: "gone", None: "never launched here"}[launcher.get("alive")]
    lines.append(f"  launcher pid {launcher.get('pid')} {alive}; log {launcher.get('log')}")
    if document.get("stop_requested"):
        lines.append(f"  STOP REQUESTED: {document['stop_requested'].get('why')}")
    if document.get("stopped"):
        lines.append(f"  stopped ({document['stopped'].get('kind')}): {document['stopped'].get('reason')}")
    for successor in document.get("resumed_into") or ():
        lines.append(f"  resumed into {successor}")
    objective = document.get("objective") or {}
    noise = objective.get("noise") or {}
    if noise:
        lines.append(
            f"  noise margin {100 * float(noise.get('margin') or 0):.2f}% ({noise.get('basis')})"
            + (f"  [{noise['flag']}]" if noise.get("flag") else "")
        )
    if objective:
        bar = (objective.get("bar") or {}).get("screen_whole_window_cycles")
        best = objective.get("best") or {}
        authored = best.get("package_authored") or {}
        lines.append(
            f"  vendor bar (context only) {_cycles(bar)}  best {_cycles(best.get('screen_whole_window_cycles'))} "
            f"({str(best.get('package_sha256') or '-')[:12]}, ratio {best.get('screen_ratio_to_bar')}"
            + (f", package-authored {_authored(authored)}" if authored else "")
            + ")"
        )
    for name, store in (document.get("stores") or {}).items():
        jobs = ", ".join(f"{k} {v}" for k, v in sorted((store.get("jobs") or {}).items())) or "no jobs"
        lines.append(f"  {name} store: {jobs}")
        if store.get("disk_hold"):
            lines.append(f"    DISK HOLD: {store['disk_hold'].get('reason')}")
        if store.get("board_outage"):
            lines.append(f"    BOARD OUTAGE since {store['board_outage'].get('opened_at')}")
        if store.get("control_preflight"):
            lines.append(f"    BATCHES HELD: {store['control_preflight'].get('reason')}")
        plateau = store.get("plateau") or {}
        if plateau:
            lines.append(
                f"    plateau: {plateau.get('hours_since')} h / {plateau.get('sessions_since')} sessions since "
                "the last improvement"
            )
    for row in document.get("rounds") or ():
        lines.append(f"  round {row['round']}: {row['status']} -- {row.get('why') or ''}"[:200])
    for row in document.get("open_rounds") or ():
        lines.append(f"  round {row['round']}: OPEN since {row.get('started_at')} ({row['requested']} requested)")
    for row in objective.get("history") or ():
        share = row.get("package_priced_share")
        authored = (
            f"pkg {row.get('package_groups')} grp {100 * float(share):.1f}%" if share is not None else "pkg -"
        ) + ("" if row.get("eligible", True) else " INELIGIBLE")
        lines.append(
            f"    {str(row.get('package_sha256'))[:12]} r{row.get('replicate')} {row.get('state'):>14} "
            f"{row.get('timing_status') or '-':>10} {_cycles(row.get('objective_cycles')):>14}  {authored:<22} "
            f"{row.get('label') or ''}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- follow
def snapshot(run_dir: Path, *, objective: Any = None) -> dict[str, Any]:
    """The facts :func:`follow` compares from one tick to the next."""
    run_dir = Path(run_dir)
    seed = read_json(run_dir / "resumed_seed.json") or {}
    jobs: dict[str, str] = {}
    holds: dict[str, bool] = {}
    for name, root in (seed.get("store_roots") or {}).items():
        state = store_state(Path(root))
        jobs.update({f"{name}/{key}": value for key, value in state["job_states"].items()})
        holds[f"{name} disk hold"] = bool(state["disk_hold"])
        holds[f"{name} board outage"] = bool(state["board_outage"])
    ended, opened = _rounds(run_dir / "stage")
    best = None
    if objective is not None:
        found = objective.summary().get("best") or {}
        if found:
            share = (found.get("package_authored") or {}).get("priced_share")
            best = (found.get("package_sha256"), found.get("screen_whole_window_cycles"), share)
    return {
        "jobs": jobs,
        "rounds": {row["round"]: row["status"] for row in ended},
        "open_rounds": sorted(row["round"] for row in opened),
        "launcher_alive": LAUNCH.launcher_alive(run_dir),
        "stop_requested": (run_dir / SES.OPERATOR_STOP_FILE).exists(),
        "stopped": ((read_json(run_dir / "stage" / "sessions.json") or {}).get("stopped") or {}).get("kind"),
        "holds": holds,
        "best": best,
        "liveness": LIVE.assess(run_dir, stores=[Path(p) for p in (seed.get("store_roots") or {}).values()])["state"],
    }


def changes(before: Mapping[str, Any] | None, after: Mapping[str, Any]) -> list[str]:
    """One line per difference between two snapshots (every fact is new when ``before`` is None)."""
    before = before or {"jobs": {}, "rounds": {}, "open_rounds": [], "holds": {}}
    lines = []
    for key, state in sorted(after["jobs"].items()):
        was = before["jobs"].get(key)
        if was != state:
            lines.append(f"job {key.partition('/')[0]}/{key.partition('/')[2][:12]} {was or 'new'} -> {state}")
    for index, status in sorted(after["rounds"].items(), key=lambda kv: int(kv[0] or 0)):
        if before["rounds"].get(index) != status:
            lines.append(f"round {index} ended: {status}")
    for index in after["open_rounds"]:
        if index not in before["open_rounds"]:
            lines.append(f"round {index} started")
    if after.get("best") != before.get("best") and after.get("best"):
        digest, cycles, share = (list(after["best"]) + [None])[:3]
        authored = "-" if share is None else f"{100 * float(share):.1f}%"
        lines.append(f"best {str(digest)[:12]} at {_cycles(cycles)} cycles (package-authored {authored})")
    for name, held in sorted(after["holds"].items()):
        if held != bool(before["holds"].get(name)):
            lines.append(f"{name} {'opened' if held else 'closed'}")
    if after.get("liveness") != before.get("liveness") and after.get("liveness"):
        lines.append(f"liveness: {after['liveness']}")
    for key, word in (("launcher_alive", "launcher"), ("stop_requested", "stop requested"), ("stopped", "stopped")):
        if after.get(key) != before.get(key) and (after.get(key) or before.get(key) is not None):
            lines.append(f"{word}: {after.get(key)}")
    return lines


def _over(now: Mapping[str, Any]) -> bool:
    """The run is over when its launcher is gone (or it recorded a stop) and no measurement is owed."""
    owed = any(state in OPEN_STATES for state in now["jobs"].values())
    return (now.get("launcher_alive") is False or bool(now.get("stopped"))) and not owed


def follow(
    run_dir: Path,
    *,
    objective: Any = None,
    interval: float = 60.0,
    max_seconds: float | None = None,
    out: Callable[[str], None] = print,
    take: Callable[[], Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Print each change until the run is over (:func:`_over`) or ``max_seconds`` pass."""
    take = take or (lambda: snapshot(run_dir, objective=objective))
    started, previous, printed = clock(), None, 0
    while True:
        now = take()
        stamp = time.strftime("%H:%M:%SZ", time.gmtime(clock()))
        for line in changes(previous, now):
            out(f"{stamp} {line}")
            printed += 1
        previous = now
        if _over(now):
            return {"ended": "the run is over", "lines": printed}
        if max_seconds is not None and clock() - started >= max_seconds:
            return {"ended": "the follow deadline passed", "lines": printed}
        sleep(interval)


__all__ = ["STATUS_SCHEMA", "changes", "follow", "format_status", "run_status", "snapshot", "store_state"]
