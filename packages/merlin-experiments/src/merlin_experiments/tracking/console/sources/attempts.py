"""Every attempt of the run's campaign: when it started and ended, why it stopped, and the time it cost.

The chain comes from the campaign manifest (:func:`.insight_common.chain`); without one it is this run
alone. Per attempt: the start is the first timestamped line of its launch log (else the run
directory's modification time); the runtime, end and exit are the service accounting block the
manifest names (``accounting_log`` / ``accounting_entry``, the ``Service runtime`` and ``Main
processes terminated with`` lines), else the launch log's modification time; the stop detail is the
last error line of its launch log; the author startup span is ``rounds/author_startup.jsonl``; grades
count ``qa_history/verdict_fast_*.json`` and the first verdict is the earliest one's modification time.
An attempt that delivered no verdict counts as lost time. ``PREVIOUS_SESSION.md`` is the continuation
handoff staged in the workspace. Archived attempts are read once. Display telemetry.

Panel fields beyond the shared shape: ``attempts`` [{label, live, start, end, runtime_s, exit,
stop_detail, startup_s, grades, first_verdict_s, handoff}], ``totals`` {wall_s, lost_s, attempts}.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..io import clean, epoch, is_file, mtime, num, obj, scan_dir
from .insight_common import ago, chain, duration, line, log_scan, panel, row, table

UNITS = (("ms", 0.001), ("us", 1e-6), ("min", 60.0), ("d", 86400.0), ("h", 3600.0), ("s", 1.0))


def runtime_seconds(text: str) -> float | None:
    """``1d 5h 28min 55.000s`` / ``25min 670ms`` as seconds (``None`` when nothing parses)."""
    total, seen = 0.0, False
    for token in text.split():
        for suffix, scale in UNITS:
            if token.endswith(suffix):
                value = token[: -len(suffix)]
                try:
                    total += float(value) * scale
                    seen = True
                except ValueError:
                    pass
                break
    return total if seen else None


def accounting_blocks(text_lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for raw in text_lines:
        key, _, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Running as unit":
            blocks.append({"unit": value})
        elif blocks and key == "Finished with result":
            blocks[-1]["result"] = value
        elif blocks and key == "Main processes terminated with":
            blocks[-1]["exit"] = value.replace("code=", "").replace("status=", "")
        elif blocks and key == "Service runtime":
            blocks[-1]["runtime_s"] = runtime_seconds(value)
    return blocks


def _text(ctx, path: Path, nbytes: int) -> list[str] | None:
    """The tail lines of a small text file, or ``None`` when this refresh's byte budget cannot hold it."""
    try:
        size = os.stat(path).st_size
    except OSError:
        return []
    if min(size, nbytes) > ctx.reader.available():
        return None
    return ctx.reader.text_tail(path, nbytes)


def _startup(ctx, run_dir: Path) -> tuple[float | None, bool]:
    """``(seconds, pending)``: author startup time of an attempt, summed over its turns.

    Each turn's startup is the span of its own stage records (``round`` and ``turn`` of
    ``merlin.author_startup_stage.v1``), so the time between two turns is never counted."""
    path = run_dir / "rounds" / "author_startup.jsonl"
    if not is_file(path):
        return None, False
    text = _text(ctx, path, 256 << 10)
    if text is None:
        return None, True
    spans: dict[tuple, list[float]] = {}
    for raw in text:
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        at = epoch(obj(record).get("at"))
        if at:
            spans.setdefault((record.get("round"), record.get("turn"), record.get("pid")), []).append(at)
    if not spans:
        return None, False
    return sum(max(v) - min(v) for v in spans.values()), False


def _grades(run_dir: Path) -> tuple[int, float | None]:
    times = [
        mtime(e.path)
        for e in scan_dir(run_dir / "qa_history", 8192)
        if e.name.startswith("verdict_fast_") and e.name.endswith(".json")
    ]
    times = [t for t in times if t]
    return len(times), (min(times) if times else None)


def _attempt(ctx, attempt: dict, accounting: list[dict] | None, now: float) -> dict[str, Any]:
    """One attempt's record; ``reading`` while its log or accounting is not read whole yet."""
    run_dir = Path(attempt.get("run_dir") or "")
    log = log_scan(ctx, attempt.get("launch_log"))
    log_pending = log is not None and not log.complete
    start = (log.first_at if log else None) or (num(ctx.experiment.get("started_at")) if attempt.get("live") else None)
    if not start and not log_pending:
        start = mtime(run_dir)
    index = attempt.get("accounting_entry")
    block = None
    if isinstance(index, int) and accounting:
        try:
            block = accounting[index]
        except IndexError:
            block = None
    if attempt.get("live"):
        block = None  # the live attempt's block is still open
    runtime = obj(block).get("runtime_s")
    end = (start + runtime) if (start and runtime is not None) else None
    if attempt.get("live"):
        end = None
        runtime = (now - start) if start else None
    elif end is None:
        end = mtime(attempt.get("launch_log"))
        runtime = (end - start) if (end and start) else None
    detail = None
    if log:
        candidates = [log.last.get(c) for c in ("grade_error", "error_line", "marker", "enospc")]
        candidates = [c for c in candidates if c]
        if candidates:
            detail = max(candidates)[1]
    startup_s, startup_pending = _startup(ctx, run_dir)
    grades, first = _grades(run_dir)
    workspace = attempt.get("workspace")
    return {
        "label": attempt.get("label"),
        "live": bool(attempt.get("live")),
        "start": start,
        "end": end,
        "runtime_s": runtime,
        "exit": obj(block).get("exit"),
        "result": obj(block).get("result"),
        "stop_detail": detail,
        "startup_s": startup_s,
        "grades": grades,
        "first_verdict_s": (first - start) if (first and start) else None,
        "handoff": is_file(Path(workspace) / "PREVIOUS_SESSION.md") if workspace else None,
        "reading": log_pending or startup_pending or accounting is None,
    }


def read(ctx) -> dict[str, Any]:
    now = ctx.now
    info = chain(ctx)
    cache: dict = ctx.state.setdefault("archived", {})
    accounting_cache: dict = ctx.state.setdefault("accounting", {})
    attempts = []
    for attempt in info["attempts"]:
        key = attempt.get("run_dir")
        if not attempt.get("live") and key in cache:
            attempts.append(cache[key])
            continue
        accounting: list[dict] | None = []
        source = attempt.get("accounting_log")
        if source:
            stamp = mtime(source)
            cached = accounting_cache.get(source)
            if not cached or cached[0] != stamp:
                text = _text(ctx, Path(source), 1 << 20)
                cached = (stamp, accounting_blocks(text)) if text is not None else None
                if cached:
                    accounting_cache[source] = cached
            accounting = cached[1] if cached else None
        record = _attempt(ctx, attempt, accounting, now)
        if not attempt.get("live") and not record["reading"]:
            cache[key] = record
        attempts.append(record)

    wall = sum(a["runtime_s"] or 0 for a in attempts)
    lost = sum(a["runtime_s"] or 0 for a in attempts if not a["live"] and not a["grades"])
    startup = sum(a["startup_s"] or 0 for a in attempts)
    current = attempts[info["current"]] if attempts else {}
    summary = [
        line("attempts", f"{len(attempts)} (this is {current.get('label') or 'the only one'})"),
        line("wall time, all attempts", duration(wall) if wall else None),
        line("lost (no verdict)", duration(lost) if attempts else None, "warn" if lost > 1800 else None),
        line("author startup, all", duration(startup) if startup else None),
        line("first verdict (this)", duration(current.get("first_verdict_s"))),
        line(
            "PREVIOUS_SESSION.md",
            {True: "present", False: "absent", None: None}[current.get("handoff")],
            "warn" if current.get("handoff") is False and len(attempts) > 1 else None,
        ),
    ]
    table_rows = []
    for a in reversed(attempts):
        stop = "running" if a["live"] else ("reading…" if a["reading"] else (a["exit"] or "not recorded"))
        level = None if a["live"] else ("warn" if not a["grades"] else None)
        table_rows.append(
            row(
                [
                    a["label"],
                    ago(now, a["start"]),
                    duration(a["runtime_s"]),
                    stop,
                    duration(a["startup_s"]),
                    a["grades"],
                    duration(a["first_verdict_s"]),
                ],
                level,
                key=a["label"],
            )
        )
    notes = [] if info["manifest"] else ["no campaign manifest names this run; showing it alone"]
    reading = sum(1 for a in attempts if a["reading"])
    if reading:
        notes.append(f"{reading} attempt(s) still being read; totals grow as they are")
    return panel(
        "ATTEMPTS",
        summary,
        table_=table(["attempt", "started", "ran", "stop", "startup", "grades", "1st verdict"], table_rows),
        notes=notes,
        attempts=attempts,
        totals={"wall_s": wall, "lost_s": lost, "startup_s": startup, "attempts": len(attempts)},
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    for a in obj(ctx.shared.get("attempts")).get("attempts") or []:
        if a.get("label") == key:
            return {
                "title": f"attempt {key}",
                "sections": [{"title": "record", "lines": [f"{k}: {clean(v, 160)}" for k, v in a.items()]}],
                "paths": [],
            }
    return {"title": f"attempt {key}", "sections": [], "paths": [], "notice": "not in the chain"}
