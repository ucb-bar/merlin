"""Run integrity and infrastructure: errors the harness logged, broker and grade health, store, disk, neighbours.

Read from the campaign log (the runner's standard output while it runs, else the chain's launch log;
counted by :data:`.insight_common.LOG_RULES`: snapshot marker mismatches, grade errors, skipped
promotions, broker and supervisor lines, dropped writes, a full disk, tracebacks), the run's
``feedback_health.json`` (the self-check broker: starts, expired, stranded) and ``grade_health.json``
(consecutive identical grade failures), the runner's ``MERLIN_BUNDLE_CAS`` (``/proc`` environment,
read-only) and its child processes, and the :mod:`.machine` section (volume, other runs' allocations
and the busy cores they hold). Example log lines can carry test-case names, so they appear only with
``operator_private``. Display telemetry.

Panel fields beyond the shared shape: ``log`` {path, lines, counts, complete}, ``broker`` {...},
``grade_health`` {...}, ``store`` {path, scope}, ``children`` {alive, defunct}, ``neighbours`` [...].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..io import clean, epoch, num, obj, rows
from .insight_common import campaign_log, line, log_scan, panel, proc_env, row, table

#: ``(code, label, level when present)`` of the log counts shown, in order.
SHOWN = (
    ("marker", "snapshot marker mismatches", "bad"),
    ("grade_error", "grade errors", "bad"),
    ("promotion_skipped", "promotions skipped", "warn"),
    ("snapshot_unavailable", "snapshot unavailable", "warn"),
    ("broker", "broker lines", "info"),
    ("restart", "restart / supervisor lines", "warn"),
    ("dropped", "dropped writes", "bad"),
    ("enospc", "disk full (ENOSPC)", "bad"),
    ("traceback", "tracebacks", "warn"),
)
MAX_PROC = 4096


def _children(pid: Any) -> dict[str, int] | None:
    """Direct children of ``pid``: alive and defunct (zombie), from ``/proc/<n>/stat``."""
    if type(pid) is not int:
        return None
    alive = defunct = 0
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for name in names[:MAX_PROC]:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as stream:
                text = stream.read(1024).decode(errors="replace")
        except OSError:
            continue
        fields = text.rpartition(")")[2].split()
        if len(fields) < 2 or fields[1] != str(pid):
            continue
        if fields[0] == "Z":
            defunct += 1
        else:
            alive += 1
    return {"alive": alive, "defunct": defunct}


def _store_scope(path: str | None, experiment: dict) -> str | None:
    if not path:
        return None
    run_id = str(experiment.get("run_id") or "")
    task = str(experiment.get("task") or "")
    if run_id and run_id in Path(path).parts:
        return "private to this run"
    if task and path.startswith(task.rstrip("/") + "/"):
        return "shared within the task"
    return "shared outside the task"


def read(ctx) -> dict[str, Any]:
    run_dir = Path(ctx.experiment.get("run_dir") or "")
    path = campaign_log(ctx)
    log = log_scan(ctx, path)
    counts = log.counts if log else {}
    broker = obj(ctx.reader.document(run_dir / "feedback_health.json", limit=1 << 20))
    health = obj(ctx.reader.document(run_dir / "grade_health.json", limit=1 << 20))
    pid = ctx.experiment.get("runner_pid")
    store = proc_env(pid, "MERLIN_BUNDLE_CAS") if ctx.live else None
    scope = _store_scope(store, ctx.experiment)
    children = _children(pid) if ctx.live else None
    machine = obj(ctx.shared.get("machine"))
    volume = obj(machine.get("volume"))
    run_id = ctx.experiment.get("run_id")
    mine = set(machine.get("experiment_cpus") or [])
    neighbours = []
    for allocation in rows(machine.get("allocations")):
        if allocation.get("kind") != "run" or allocation.get("name") == run_id:
            continue
        name = allocation.get("name")
        held = [c for c in rows(machine.get("cores"), 1024) if name in (c.get("holders") or [])]
        # Busy cores are a live sample: a finished run's panel does not churn with the host.
        busy = sum((num(c.get("util_pct")) or 0) for c in held) / 100 if ctx.live else None
        overlap = sum(1 for c in held if c.get("cpu") in mine)
        neighbours.append(
            {
                "name": clean(name, 48),
                "cpus": clean(allocation.get("cpus"), 24),
                "busy_cores": round(busy, 1) if busy is not None else None,
                "overlap": overlap,
            }
        )

    def count(code: str) -> int | None:
        return counts.get(code, 0) if log else None

    bad_log = sum(counts.get(c, 0) for c, _, level in SHOWN if level == "bad")
    starts = num(broker.get("broker_starts"))
    summary = [
        line(
            "log errors",
            f"{bad_log} (marker, grade, dropped, disk)" if log else None,
            "bad" if bad_log else "ok",
            key="log",
        ),
        line(
            "promotions skipped",
            f"{count('promotion_skipped')} · snapshot unavailable {count('snapshot_unavailable')}" if log else None,
            "warn" if counts.get("promotion_skipped") else None,
        ),
        line(
            "self-check broker",
            f"{'healthy' if broker.get('healthy') else 'unhealthy'} · starts {starts} · "
            f"expired {num(broker.get('expired'))} · stranded {num(broker.get('stranded'))}"
            if broker
            else None,
            None if not broker else ("ok" if broker.get("healthy") and (starts or 0) <= 1 else "warn"),
        ),
        line(
            "grade health",
            ("alert: " + clean(health.get("error_class") or health.get("error"), 60))
            if health.get("alert")
            else f"ok · {num(health.get('consecutive_identical_failures'))} identical failures"
            if health
            else None,
            "bad" if health.get("alert") else ("ok" if health else None),
        ),
        line(
            "content store",
            f"{scope} ({Path(store).name})" if store else ("not visible" if ctx.live else "run not live"),
            "warn" if scope == "shared outside the task" else None,
        ),
        line(
            "runner children",
            f"{children['alive']} alive · {children['defunct']} defunct" if children else None,
            "warn" if children and children["defunct"] else None,
        ),
        line(
            "disk free",
            f"{volume.get('free_pct')}% ({(num(volume.get('free_bytes')) or 0) / 2**30:.0f} GiB)"
            if volume.get("free_pct") is not None
            else None,
            {"ok": "ok", "warn": "warn", "bad": "bad"}.get(volume.get("level")),
        ),
        line(
            "other runs",
            ", ".join(
                f"{n['name']}"
                + (f" {n['busy_cores']} cores" if n["busy_cores"] is not None else "")
                + (" (on our CPUs!)" if n["overlap"] else "")
                for n in neighbours
            )
            or "none",
            "bad" if any(n["overlap"] for n in neighbours) else ("info" if neighbours else None),
        ),
    ]
    table_rows = []
    for code, label, level in SHOWN:
        n = count(code)
        example = obj(log.last if log else {}).get(code)
        detail = example[1] if example and ctx.operator_private else ("" if not example else f"line {example[0]}")
        table_rows.append(row([label, n, detail], level if n else None, key=code))
    notes = []
    if log and not log.complete:
        notes.append(f"log read so far: {log.lines} lines (catching up)")
    if not path:
        notes.append("no campaign log found for this run")
    return panel(
        "INTEGRITY",
        summary,
        table_=table(["log signal", "n", "last"], table_rows),
        notes=notes,
        log={
            "path": path if ctx.operator_private else None,
            "lines": log.lines if log else None,
            "counts": dict(counts),
            "complete": log.complete if log else None,
        },
        broker={
            k: broker.get(k)
            for k in ("healthy", "broker_starts", "expired", "stranded", "in_flight", "max_queue_depth")
        }
        if broker
        else None,
        grade_health={
            "alert": health.get("alert"),
            "consecutive": num(health.get("consecutive_identical_failures")),
            "last_completed_at": epoch(health.get("last_completed_at")),
        }
        if health
        else None,
        store={"path": store if ctx.operator_private else None, "scope": scope},
        children=children,
        neighbours=neighbours,
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    section = obj(ctx.shared.get("integrity"))
    log = obj(section.get("log"))
    lines = [f"{code}: {n}" for code, n in sorted(obj(log.get("counts")).items())]
    return {
        "title": "campaign log signals",
        "sections": [{"title": f"{log.get('lines')} lines read", "lines": lines}],
        "paths": [log["path"]] if log.get("path") else [],
    }
