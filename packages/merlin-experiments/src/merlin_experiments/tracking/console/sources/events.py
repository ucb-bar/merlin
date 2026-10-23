"""One time-ordered feed of what happened: agent messages and failures, checks, simulations,
construction actions, round ends, stops and the launch log's summary lines.

Every row carries the record it came from. Times are the record's own when it has one; a
self-check or simulation response without one uses its file modification time (labelled), and
launch-log lines, which carry no time, have ``t: None`` and sort after timed rows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..io import bounded_glob, clean, epoch, is_file, mtime, num, obj, rows, scan_dir
from .agents import index_for

LIMIT = 300
CHANNEL_SCAN = 20000
RESPONSES = 60
LOG_TAIL = 256 << 10
LOG_PREFIXES = ("[round", "[joint]", "[promote] L2:", "[promote] L3:", "[promote] L2 pass")


def _event(t, kind, severity, text, source, agent=None) -> dict[str, Any]:
    return {"t": t, "kind": kind, "severity": severity, "agent": agent, "text": clean(text, 300), "source": source}


def _channel(ctx, workspace: Path) -> list[tuple[float, Path]]:
    """Newest response files, re-listed only when the directory changed."""
    channel = workspace / ".qa_channel"
    try:
        stamp = os.stat(channel).st_mtime_ns
    except OSError:
        return []
    cached = ctx.state.get("channel")
    if cached and cached[0] == stamp:
        return cached[1]
    found = []
    for entry in scan_dir(channel, CHANNEL_SCAN):
        if entry.name.endswith(".json") and entry.name.startswith(("resp_", "simresp_")):
            try:
                found.append((entry.stat().st_mtime, Path(entry.path)))
            except OSError:
                continue
    found.sort(reverse=True)
    resp = [f for f in found if f[1].name.startswith("resp_")][:RESPONSES]
    sim = [f for f in found if f[1].name.startswith("simresp_")][:RESPONSES]
    ctx.state["channel"] = (stamp, resp + sim)
    return resp + sim


def _responses(ctx, workspace: Path, out: list) -> None:
    private = ctx.operator_private
    for modified, path in _channel(ctx, workspace):
        doc = ctx.reader.document(path, limit=8 << 20)
        if doc is None:
            continue
        source = str(path) if private else path.name
        if path.name.startswith("resp_"):
            t = epoch(doc.get("graded_at_unix_ns")) or modified
            if doc.get("error"):
                out.append(_event(t, "selfcheck", "warn", f"self-check error: {clean(doc['error'], 120)}", source))
                continue
            passed, total = doc.get("n_passed"), doc.get("n_capsules") or doc.get("n_result_rows")
            verdict = "pass" if doc.get("all_pass") is True else "fail" if doc.get("all_pass") is False else "unknown"
            count = f" {passed}/{total}" if type(passed) is int and type(total) is int else ""
            out.append(
                _event(t, "selfcheck", "info" if verdict == "pass" else "warn", f"self-check {verdict}{count}", source)
            )
        else:
            per = rows(doc.get("per_capsule"))
            passed = sum(1 for r in per if r.get("pass") is True)
            text = (
                f"{clean(doc.get('sim') or 'sim', 12)} {clean(doc.get('barrier_tier'), 6)}: {passed}/{len(per)} passed"
            )
            if private and per:
                text += (
                    " (" + ", ".join(clean(r.get("capsule"), 48) for r in per[:3]) + ("…" if len(per) > 3 else "") + ")"
                )
            out.append(
                _event(modified, "sim", "info" if per and passed == len(per) else "warn", text + " [file time]", source)
            )


def _construction(ctx, workspace: Path | None, run: Path, out: list) -> None:
    feed = ctx.state.setdefault("construction", [])
    for stream in (workspace / "construction_study" / "events.jsonl" if workspace else None,):
        if stream is None:
            continue

        def consume(row, stream=stream):
            code = row.get("returncode")
            failed = row.get("state") in ("failed", "error") or (type(code) is int and code != 0)
            text = (
                f"action {clean(row.get('action_id'), 40)} {clean(row.get('state'), 16)}"
                f" [{clean(row.get('boundary'), 24)}] {clean(row.get('capability'), 40)}"
            )
            if type(code) is int:
                text += f" rc={code}"
            feed.append(
                _event(
                    epoch(row.get("time_ns")),
                    "action",
                    "warn" if failed else "info",
                    text,
                    str(stream) if ctx.operator_private else stream.name,
                )
            )
            del feed[:-LIMIT]

        ctx.reader.poll(stream, consume, feed.clear)
    out.extend(feed)


def _run_records(ctx, run: Path, out: list) -> None:
    stop = run / "joint_epoch_stop.json"
    doc = ctx.reader.document(stop, limit=1 << 20)
    if doc is not None:
        t = epoch(doc.get("stopped_at")) or epoch(doc.get("at")) or mtime(stop)
        out.append(
            _event(t, "stop", "warn", f"epoch stop: {clean(doc.get('reason') or 'no reason recorded', 200)}", stop.name)
        )
    for summary in sorted(bounded_glob(run / "rounds", "round_*.codex_summary.json", 32)):
        doc = obj(ctx.reader.document(summary, limit=4 << 20))
        if not doc:
            continue
        label = summary.name.split(".")[0]
        if doc.get("operator_interrupted"):
            text, severity = f"{label} interrupted: {clean(doc.get('interruption_reason'), 120)}", "warn"
        elif doc.get("timed_out"):
            text, severity = f"{label} timed out", "warn"
        else:
            text, severity = f"{label} ended (exit {doc.get('exit_code')})", "info"
        wall = num(doc.get("wall_s"))
        if wall is not None:
            text += f" after {wall / 3600:.1f} h"
        out.append(_event(mtime(summary), "round", severity, text + " [file time]", summary.name))


def _log(ctx, out: list) -> None:
    launch = ctx.experiment.get("launch_dir")
    if not launch:
        return
    log = Path(launch) / "process" / "native.log"
    if not is_file(log):
        return
    lines = [line.strip() for line in ctx.reader.text_tail(log, LOG_TAIL)]
    picked = [line for line in lines if line.startswith(LOG_PREFIXES) and "invalidated by" not in line][-10:]
    for line in reversed(picked):
        severity = "warn" if "NOT CONFORMANT" in line or "fail" in line.lower() else "info"
        text = line if ctx.operator_private else line.split("—")[0].split(":")[0]
        out.append(_event(None, "log", severity, text, "native.log" if not ctx.operator_private else str(log)))


def read(ctx) -> dict[str, Any]:
    run = Path(ctx.experiment.get("run_dir") or "")
    workspace = Path(ctx.experiment["workspace"]) if ctx.experiment.get("workspace") else None
    index = index_for(ctx)
    names = {tid: t.get("name") or t.get("path") or tid[:8] for tid, t in index.threads.items()}
    events: list[dict[str, Any]] = []
    for row in index.feed:
        text = row["text"]
        if not ctx.operator_private:
            text = "agent message" if row["kind"] == "message" else text.split(":")[0] + " (command hidden)"
        events.append(
            _event(
                row["t"],
                row["kind"],
                row["severity"],
                text,
                "agent stream",
                clean(names.get(row["agent"], row["agent"]), 40),
            )
        )
    if workspace:
        _responses(ctx, workspace, events)
    _construction(ctx, workspace, run, events)
    _run_records(ctx, run, events)
    _log(ctx, events)
    events.sort(key=lambda e: (e["t"] is None, -(e["t"] or 0)))
    notes = []
    if not events:
        notes.append("No events recorded yet.")
    if workspace is None:
        notes.append("Workspace not recorded; self-check and simulation responses unavailable.")
    return {"events": events[:LIMIT], "notes": notes}


__all__ = ["read"]
