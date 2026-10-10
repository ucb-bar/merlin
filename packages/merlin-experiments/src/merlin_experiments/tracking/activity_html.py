"""The agent-activity section of a run page: what it is doing now, a feed, and per-round tokens/time."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import activity as A
from . import records as R
from .html import NR, details, esc, hours, mono, table, text_or_nr, tile, when

_FEED_KINDS = ("message", "reasoning", "error", "turn failed", "edit", A.SELFCHECK, A.TEST, A.BUILD, A.WAIT)


def _ago(at: float | None, now: float) -> str:
    if at is None:
        return NR
    minutes = max(0.0, (now - at) / 60.0)
    return f"{minutes:.0f} min ago" if minutes < 120 else hours(minutes / 60.0) + " ago"


def now_panel(activity: Mapping[str, Any], now: float) -> str:
    current = activity["now"]
    parts = [f"<p><b>Last event</b> {when(current.get('last_event'))} ({_ago(current.get('last_event'), now)}).</p>"]
    running = current.get("running") or []
    if running:
        rows = "".join(
            f"<li>{esc(c.get('label') or c['type'])}: {mono(A._clip(c.get('command') or '', 240))} "
            f"since {when(c['start'])} ({_ago(c['start'], now)})</li>"
            for c in running
        )
        parts.append(f"<p><b>Running now</b> (started, no completion recorded):</p><ul>{rows}</ul>")
    else:
        parts.append("<p><b>Running now:</b> no tool call is open in the stream.</p>")
    message = current.get("last_message")
    if message:
        parts.append(
            f"<p><b>Latest assistant message</b> ({when(message['at'])}):</p>"
            f'<pre class="note">{esc(message["text"])}</pre>'
        )
    thought = current.get("last_reasoning")
    if thought:
        parts.append(
            f"<p><b>Latest reasoning summary</b> ({when(thought['at'])}):</p>"
            f'<pre class="note">{esc(thought["text"])}</pre>'
        )
    todo = current.get("todo")
    if todo:
        items = "".join(f"<li>{'&#10003;' if t['completed'] else '&#9744;'} {esc(t['text'])}</li>" for t in todo)
        parts.append(f"<p><b>Its plan (latest todo list)</b>:</p><ul>{items}</ul>")
    return '<div class="now">' + "".join(parts) + "</div>"


def feed_list(activity: Mapping[str, Any], limit: int = A.FEED_LIMIT) -> str:
    rows = [f for f in activity["feed"] if f["kind"] in _FEED_KINDS or f.get("error")][-limit:]
    if not rows:
        return f"<p>Activity feed: {NR}.</p>"
    items = []
    for f in reversed(rows):
        cls = ' class="err"' if f.get("error") else ""
        detail = f' <span class="sub">{esc(f["detail"])}</span>' if f.get("detail") else ""
        items.append(
            f'<li><span class="t">{esc(R.stamp(f["at"]) or "no time")}</span><span class="kind">{esc(f["kind"])}</span>'
            f"<span{cls}>{esc(f['text'])}</span>{detail}</li>"
        )
    return f'<ul class="feed">{"".join(items)}</ul>'


def _tiles(activity: Mapping[str, Any]) -> str:
    counts = activity["command_counts"]
    tokens_out = sum(r["output_tokens"] for r in activity["rounds"])
    tokens_in = sum(r["input_tokens"] for r in activity["rounds"])
    usage = sum(r["usage_turns"] for r in activity["rounds"])
    errors = sum(r["errors"] for r in activity["rounds"])
    tiles = [
        tile("Rounds with a stream", str(len(activity["rounds"])), f"{len(activity['turns'])} turns"),
        tile("Commands run", f"{activity['n_commands']:,}", f"{activity['failed_commands']:,} exited non-zero"),
        tile("Self-checks", f"{counts[A.SELFCHECK]:,}", f"{counts[A.WAIT]:,} verdict waits"),
        tile("Tests / builds", f"{counts[A.TEST]:,} / {counts[A.BUILD]:,}", "labelled by the command's words"),
        tile("Files edited", f"{len(activity['edits']):,}", f"{sum(activity['edits'].values()):,} changes"),
        tile("Errors", f"{errors:,}", "error events and failed turns"),
        tile(
            "Tokens (reported turns)",
            f"{tokens_in:,} in / {tokens_out:,} out" if usage else NR,
            f"{usage} turns reported usage",
        ),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def rounds_table(activity: Mapping[str, Any]) -> str:
    rows = []
    for r in activity["rounds"]:
        span = (r["last"] - r["first"]) / 3600.0 if r["first"] is not None and r["last"] is not None else None
        usage = r["usage_turns"] > 0
        rows.append(
            [
                str(r["round"]),
                when(r["first"]),
                hours(span),
                str(r["turns"]),
                str(r["calls"]),
                str(r["failed_calls"]),
                f"{r['input_tokens']:,}" if usage else NR,
                f"{r['cached_input_tokens']:,}" if usage else NR,
                f"{r['output_tokens']:,}" if usage else NR,
            ]
        )
    headers = ["round", "first event", "span", "turns", "tool calls", "failed", "input tok", "cached", "output tok"]
    return table(headers, rows, numeric=(3, 4, 5, 6, 7, 8))


def edits_table(activity: Mapping[str, Any], limit: int = 40) -> str:
    rows = [[mono(path), str(n)] for path, n in list(activity["edits"].items())[:limit]]
    return table(["file", "changes"], rows, numeric=(1,))


def section(activity: Mapping[str, Any] | None, now: float) -> str:
    out = '<section id="activity"><h2>Agent activity</h2>'
    if not activity:
        return out + (
            '<p>Agent event stream (<span class="mono">rounds/round_NN.codex_events.timestamped.jsonl</span>): '
            f"{NR}.</p></section>"
        )
    out += _tiles(activity)
    out += "<h3>What it is working on now</h3>" + now_panel(activity, now)
    out += (
        f"<h3>Activity feed (newest first, last {A.FEED_LIMIT})</h3>"
        '<p class="sub">Messages, reasoning summaries, edits, self-checks, tests, builds and every error. '
        "Command labels come from the words a command runs; they are a reading aid, not a classification "
        "any grade uses.</p>" + feed_list(activity)
    )
    other = [
        [
            when(c["start"]),
            esc(c.get("label") or c["type"]),
            mono(A._clip(c.get("command") or "", 200)),
            text_or_nr(c.get("exit_code")),
            "running" if c["end"] is None else esc(c.get("status") or ""),
        ]
        for c in reversed(activity["calls"][-200:])
        if c["type"] == "command_execution"
    ]
    out += details(
        f"All commands, newest 200 ({activity['n_commands']} in total)",
        table(["started", "label", "command", "exit", "status"], other),
    )
    out += "<h3>Files the agent edited</h3>" + edits_table(activity)
    out += "<h3>Per round: time, calls and tokens (from the stream's own usage)</h3>" + rounds_table(activity)
    if activity.get("unknown_events"):
        out += f'<p class="sub">{activity["unknown_events"]} events of a type this reader does not follow.</p>'
    return out + "</section>"


__all__ = ["feed_list", "now_panel", "section"]
