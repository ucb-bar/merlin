"""One time-ordered feed of what happened, as milestones a reader can follow.

``kind`` says which part of the run the event belongs to:

* ``author`` -- a turn admitted, its startup finished, the host interrupted the author, the session
  ended (:mod:`.lifecycle`);
* ``agent`` -- a helper started or finished; failed commands, grouped per agent and minute (with the
  command text in operator-private mode); agent messages (operator-private mode only);
* ``check`` -- a self-check response; ``sim`` -- a simulation response; ``grade`` -- a recorded grade
  of the submission (passing count); ``action`` -- a construction action; ``stop`` -- an epoch stop or
  round end.

Every row carries the record it came from. Times are the record's own when it has one; a self-check
or simulation response without one uses its file modification time.
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
                out.append(_event(t, "check", "warn", f"self-check error: {clean(doc['error'], 120)}", source))
                continue
            passed, total = doc.get("n_passed"), doc.get("n_capsules") or doc.get("n_result_rows")
            verdict = "pass" if doc.get("all_pass") is True else "fail" if doc.get("all_pass") is False else "unknown"
            count = f" {passed}/{total}" if type(passed) is int and type(total) is int else ""
            out.append(
                _event(t, "check", "info" if verdict == "pass" else "warn", f"self-check {verdict}{count}", source)
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
            out.append(_event(modified, "sim", "info" if per and passed == len(per) else "warn", text, source))


def _construction(ctx, workspace: Path | None, run: Path, out: list) -> None:
    feed = ctx.state.setdefault("construction", [])
    for stream in (workspace / "construction_study" / "events.jsonl" if workspace else None,):
        if stream is None:
            continue

        last_state = ctx.state.setdefault("action_state", {})

        def consume(row, stream=stream):
            # One event per state change of an action: a poller re-recording "pending" is not news.
            action, state = row.get("action_id"), row.get("state")
            if last_state.get(action) == state:
                return
            last_state[action] = state
            code = row.get("returncode")
            failed = state in ("failed", "error") or (
                state in ("done", "completed") and type(code) is int and code != 0
            )
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

        ctx.reader.poll(stream, consume, lambda: (feed.clear(), last_state.clear()))
    out.extend(feed)


def _run_records(ctx, run: Path, out: list) -> None:
    covered = {t.get("round") for t in rows(obj(ctx.shared.get("lifecycle")).get("turns")) if t.get("agent_end")}
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
        digits = summary.name[len("round_") :].split(".", 1)[0]
        label = f"round {int(digits)}" if digits.isdigit() else summary.name.split(".")[0]
        if digits.isdigit() and int(digits) in covered:
            continue  # The lifecycle already says how this round's turns ended.
        errors = [e for e in doc.get("unrecovered_errors") or doc.get("errors") or [] if isinstance(e, str)]
        if doc.get("turns_started") == 0:
            reason = errors[0] if errors else doc.get("interruption_reason") or "no reason recorded"
            out.append(
                _event(
                    mtime(summary),
                    "stop",
                    "warn",
                    f"{label} ended before any agent turn: {clean(reason, 160)}",
                    summary.name,
                )
            )
            continue
        if doc.get("operator_interrupted"):
            text, severity = f"{label} interrupted: {clean(doc.get('interruption_reason'), 120)}", "warn"
        elif doc.get("timed_out"):
            text, severity = f"{label} timed out", "warn"
        else:
            text, severity = f"{label} ended (exit {doc.get('exit_code')})", "info"
        wall = num(doc.get("wall_s"))
        if wall is not None:
            text += f" after {wall / 3600:.1f} h"
        out.append(_event(mtime(summary), "stop", severity, text, summary.name))


def _lifecycle(ctx, out: list) -> None:
    life = obj(ctx.shared.get("lifecycle"))
    for turn in rows(life.get("turns")):
        name = f"round {turn.get('round')} turn {turn.get('turn')}"
        if turn.get("admitted_at"):
            allowance = num(turn.get("allowance"))
            text = f"{name} admitted ({turn.get('phase') or 'phase not recorded'}"
            text += f", {allowance / 1e6:.1f}M token allowance)" if allowance else ")"
            out.append(_event(turn["admitted_at"], "author", "info", text, "transcript"))
        if turn.get("startup_end") and turn.get("startup_s") is not None:
            slow = max(
                (
                    st
                    for st in rows(turn.get("stages"))
                    if st.get("depth") and isinstance(st.get("wall_s"), int | float)
                ),
                key=lambda st: st["wall_s"],
                default=None,
            )
            text = f"{name} startup finished in {_dur(turn['startup_s'])}"
            if slow is not None:
                text += f" (slowest: {slow['stage']} {_dur(slow['wall_s'])})"
            severity = "warn" if turn["startup_s"] >= 300 else "info"
            out.append(_event(turn["startup_end"], "author", severity, text, "author_startup.jsonl"))
        if turn.get("agent_end") and turn.get("end_reason") and turn["end_reason"] != "next turn started":
            worked = turn.get("agent_s")
            text = f"{name} ended: {turn['end_reason']}"
            text += f" (agent worked {_dur(worked)})" if worked is not None else " (the agent never ran)"
            out.append(_event(turn["agent_end"], "author", "warn", text, "transcript"))


def _helpers(ctx, out: list) -> None:
    for agent in rows(obj(ctx.shared.get("agents")).get("agents")):
        if agent.get("role") != "subagent":
            continue
        name = clean(agent.get("name") or agent.get("path") or "helper", 40)
        where = f" on {clean(agent.get('path'), 60)}" if agent.get("path") else ""
        if agent.get("started_at"):
            out.append(_event(agent["started_at"], "agent", "info", f"helper started{where}", "agent stream", name))
        if agent.get("state") in ("completed", "aborted", "stopped") and agent.get("last_activity_at"):
            verb = {"completed": "finished", "aborted": "aborted", "stopped": "stopped (author interrupted)"}
            severity = "info" if agent["state"] == "completed" else "warn"
            out.append(
                _event(
                    agent["last_activity_at"], "agent", severity, f"helper {verb[agent['state']]}", "agent stream", name
                )
            )


def _grades(ctx, out: list) -> None:
    series = obj(obj(ctx.shared.get("progress")).get("series"))
    total = num(obj(ctx.shared.get("progress")).get("n_selected"))
    previous = None
    for t, passed in zip(series.get("t") or [], series.get("n_passed") or []):
        if t is None or passed is None:
            continue
        if passed == previous:
            continue
        change = "" if previous is None else f" ({passed - previous:+d})"
        of = f"/{int(total)}" if total else ""
        out.append(_event(t, "grade", "info", f"graded: {passed}{of} capsules pass{change}", "qa_history"))
        previous = passed


def _dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{seconds:.0f}s" if seconds < 90 else f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def _agent_feed(ctx, index, out: list) -> None:
    """Agent stream rows: failures grouped per agent and minute; messages only in operator-private mode."""
    names = {
        tid: "lead" if t.get("role") == "lead" and not t.get("name") else t.get("name") or t.get("path") or tid[:8]
        for tid, t in index.threads.items()
    }
    groups: dict[tuple, list[dict]] = {}
    for row in index.feed:
        who = clean(names.get(row["agent"], row["agent"]), 40)
        if row["kind"] == "message":
            if ctx.operator_private:
                out.append(_event(row["t"], "agent", "info", row["text"], "agent stream", who))
            continue
        minute = int((row["t"] or 0) // 60)
        groups.setdefault((who, minute), []).append(row)
    for (who, _), group in groups.items():
        last = max(r["t"] or 0 for r in group)
        if len(group) == 1:
            text = group[0]["text"] if ctx.operator_private else group[0]["text"].split(":")[0]
            text = f"command failed ({text})"
        else:
            codes = sorted({r["text"].split(":")[0] for r in group})
            text = f"{len(group)} commands failed ({', '.join(codes)})"
            if ctx.operator_private:
                text += ": " + group[-1]["text"].split(": ", 1)[-1]
        out.append(_event(last or None, "agent", "warn", text, "agent stream", who))


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
    events: list[dict[str, Any]] = []
    _agent_feed(ctx, index, events)
    _helpers(ctx, events)
    _lifecycle(ctx, events)
    _grades(ctx, events)
    if workspace:
        _responses(ctx, workspace, events)
    _construction(ctx, workspace, run, events)
    _run_records(ctx, run, events)
    # Launch-log summary lines carry no time; they belong to the inspector, not the timed feed.
    events = [e for e in events if e["t"] is not None]
    events.sort(key=lambda e: -e["t"])
    notes = []
    if not events:
        notes.append("No events recorded yet.")
    if workspace is None:
        notes.append("Workspace not recorded; self-check and simulation responses unavailable.")
    return {"events": events[:LIMIT], "notes": notes}


__all__ = ["read"]
