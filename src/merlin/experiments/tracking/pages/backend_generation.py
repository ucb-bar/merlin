"""The richer Phase 1 sections of a run page: timeline, compiler evolution, family pass rates, cost/time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.experiments.tracking.pages import charts as C
from merlin.experiments.tracking.pages.markup import (
    NR,
    details,
    esc,
    hours,
    mono,
    num,
    table,
    text_or_nr,
    tile,
    when,
)
from merlin.experiments.tracking.readers import activity as A
from merlin.experiments.tracking.readers import records as R

# ---- phase1_views ------------------------------------------------------------------------------

#: Gantt colours by kind (identity in fixed categorical order; pass/fail keep their status colours).
COLOURS = {
    "turn": "var(--c1)",
    A.SELFCHECK: "var(--c3)",
    A.TEST: "var(--c7)",
    A.BUILD: "var(--c4)",
    A.WAIT: "var(--muted)",
    A.READ: "var(--c5)",
    A.OTHER: "var(--c2)",
    "edit": "var(--c6)",
    "mcp_tool_call": "var(--c5)",
    "web_search": "var(--c5)",
    "sim": "var(--c4)",
    "grade": "var(--c1)",
    "pass": "var(--good)",
    "fail": "var(--critical)",
    "failed": "var(--critical)",
    "freeze": "var(--ink)",
}
NAMES = {
    "turn": "agent turn",
    A.SELFCHECK: "self-check command",
    A.TEST: "test command",
    A.BUILD: "build command",
    A.WAIT: "verdict wait",
    A.READ: "read command",
    A.OTHER: "other command",
    "edit": "file edit",
    "sim": "sim job (running or queued)",
    "grade": "grade (not all pass)",
    "pass": "grade all pass / job passed",
    "fail": "failed (job error, exit non-zero)",
    "freeze": "freeze / operator seal",
}


def timeline_section(activity: Mapping[str, Any] | None, detail: Mapping[str, Any], now: float) -> str:
    lanes = A.lanes(activity) + list(detail.get("lanes") or ())
    out = '<section id="timeline"><h2>Timeline: agent, grades, simulator jobs, freeze</h2>'
    out += (
        "<p class='sub'>Capsule L2/L3 points are grade milestones; unavailable means no oracle result. "
        "Simulator bars show recorded job spans, not individual capsule durations. "
        "Private capsules are omitted from named lanes.</p>"
    )
    out += C.gantt_legend(COLOURS, NAMES)
    out += "<h3>Whole run</h3>" + C.gantt(lanes, now=now, colours=COLOURS, label="run timeline")
    starts = [s["start"] for lane in lanes for s in lane.get("spans") or () if s.get("start") is not None]
    if starts and now - min(starts) > 3 * 3600:
        out += "<h3>Last two hours</h3>" + C.gantt(
            lanes, now=now, colours=COLOURS, label="recent timeline", window=(now - 2 * 3600, now)
        )
    jobs = detail.get("jobs") or []
    if jobs:
        sims = [j for j in jobs if j.get("kind") == "simjob"]
        active = [j for j in sims if j.get("state") == "running"]
        queued = sum(j.get("state") == "queued" for j in sims)
        timing = [j.get("resource_timing", {}) for j in sims]
        waits = [t["queue_wait_s"] for t in timing if "queue_wait_s" in t]
        runtimes = [t["worker_runtime_s"] for t in timing if "worker_runtime_s" in t]
        slots = [j.get("resource_timing", {}).get("reserved_verilator_slots") for j in active]
        occupancy = str(sum(slots)) if slots and all(v is not None for v in slots) else NR
        mean_wait = f"{sum(waits) / len(waits):.1f}s" if waits else NR
        mean_runtime = f"{sum(runtimes) / len(runtimes):.1f}s" if runtimes else NR
        out += (
            f"<p>Simulation jobs: {len(active)} active, {queued} queued; "
            f"legacy Verilator reservations: {occupancy}. Recorded mean queue wait: {mean_wait}; "
            f"mean completed worker runtime: {mean_runtime}. "
            "Actual gSIM occupancy is not recorded here; legacy reservations are not measured emulator utilization.</p>"
        )
        rows = [
            [
                esc(j["kind"]),
                esc(j["sim"]),
                mono(j["id"]),
                when(j.get("requested")),
                when(j.get("started")),
                when(j.get("ended")),
                esc(j["state"]),
                text_or_nr(j.get("resource_timing", {}).get("queue_wait_s")),
                text_or_nr(j.get("resource_timing", {}).get("worker_runtime_s")),
                text_or_nr(j.get("resource_timing", {}).get("reserved_verilator_slots")),
                text_or_nr(j.get("all_pass")),
                text_or_nr(j.get("capsules")),
            ]
            for j in reversed(jobs[-150:])
        ]
        out += details(
            f"Simulator jobs and self-check requests ({len(jobs)}; ends are marker-file times)",
            table(
                [
                    "kind",
                    "engine",
                    "id",
                    "requested",
                    "started",
                    "ended",
                    "state",
                    "queue wait (s)",
                    "worker runtime (s)",
                    "legacy Verilator slots",
                    "all pass",
                    "capsules",
                ],
                rows,
            ),
        )
    else:
        out += f"<p>Simulator jobs and self-check requests (<span class='mono'>.qa_channel</span>): {NR}.</p>"
    checks = detail.get("selfchecks")
    if checks is None:
        out += f"<p>Self-check log (<span class='mono'>selfcheck_log.jsonl</span>): {NR}.</p>"
    elif not checks["anchored"]:
        out += "<p class='sub'>Self-check log rows have offsets but no recorded authoring start to anchor them.</p>"
    return out + "</section>"


def _names(values: Sequence[str], limit: int = 30) -> str:
    if not values:
        return "&ndash;"
    shown = ", ".join(esc(v) for v in values[:limit])
    return shown + (f" &hellip; (+{len(values) - limit})" if len(values) > limit else "")


def _delta(d: Mapping[str, Any] | None) -> str:
    if not d:
        return NR
    if d.get("first"):
        return "first checkpoint"
    parts = []
    for key in ("passes", "ops", "dialects"):
        added, removed = d[key]["added"], d[key]["removed"]
        if added:
            parts.append(f"+{len(added)} {key} ({_names(added, 6)})")
        if removed:
            parts.append(f"&minus;{len(removed)} {key} ({_names(removed, 6)})")
    if d["files_added"]:
        parts.append(f"+{len(d['files_added'])} files")
    if d["files_removed"]:
        parts.append(f"&minus;{len(d['files_removed'])} files")
    parts.append(f"{d['loc_delta']:+,} LOC")
    return "; ".join(parts)


def compiler_section(evolution: Mapping[str, Any] | None) -> str:
    out = '<section id="compiler"><h2>Compiler evolution</h2>'
    checkpoints = (evolution or {}).get("checkpoints") or []
    if not checkpoints:
        return out + (
            "<p>Snapshots: "
            f"{NR} (no <span class='mono'>oot_commits.jsonl</span> commit and no "
            "<span class='mono'>submission/</span>)."
            "</p></section>"
        )
    out += (
        '<p class="sub">Parsed from each graded snapshot in the run\'s <span class="mono">oot/</span> history (git '
        'object reads only) and the final <span class="mono">submission/</span>: TableGen defs, Python classes '
        "(ast), C++ getArgument() names, manifest.yaml. A name the parser does not recognise is not listed; LOC "
        "counts are exact.</p>"
    )
    if evolution.get("sampled"):
        out += f'<p class="sub">More than {len(checkpoints)} checkpoints: an evenly spaced sample plus the latest.</p>'
    series = {
        "passes": [
            (c["at"], len(c["analysis"]["passes"]), f"{c['label']}: {len(c['analysis']['passes'])} passes")
            for c in checkpoints
            if c["analysis"] and c["at"]
        ],
        "ops": [
            (c["at"], len(c["analysis"]["ops"]), f"{c['label']}: {len(c['analysis']['ops'])} ops")
            for c in checkpoints
            if c["analysis"] and c["at"]
        ],
    }
    loc = {
        "LOC": [
            (c["at"], c["analysis"]["loc_total"], f"{c['label']}: {c['analysis']['loc_total']:,} LOC")
            for c in checkpoints
            if c["analysis"] and c["at"]
        ]
    }
    out += "<h3>Passes and ops over checkpoints</h3>" + C.time_series(
        series, label="passes and ops", step=True, y_fmt=lambda v: f"{v:g}"
    )
    out += "<h3>Lines of source over checkpoints</h3>" + C.time_series(loc, label="LOC", step=True)
    rows = []
    for c in reversed(checkpoints):
        a = c["analysis"]
        rows.append(
            [
                when(c["at"]),
                esc(c["label"]),
                mono((c.get("commit") or c["source"])[:12]),
                f"{c['n_passed']}/{c['n_capsules']}" if c.get("n_passed") is not None else NR,
                str(len(a["passes"])) if a else NR,
                str(len(a["ops"])) if a else NR,
                str(len(a["dialects"])) if a else NR,
                num(a["loc_total"]) if a else NR,
                _delta(c.get("diff")),
            ]
        )
    out += "<h3>Per checkpoint</h3>" + table(
        ["at", "checkpoint", "commit", "passed", "passes", "ops", "dialects", "LOC", "change since previous"],
        rows,
        (4, 5, 6, 7),
    )
    latest = next((c for c in reversed(checkpoints) if c["analysis"]), None)
    if latest:
        a = latest["analysis"]
        manifest = a.get("manifest") or {}
        out += f"<h3>Latest snapshot ({esc(latest['label'])})</h3>"
        out += '<div class="cols">'
        out += f"<div><h3>Passes ({len(a['passes'])})</h3><p>{_names(a['passes'], 200)}</p></div>"
        out += f"<div><h3>Dialects ({len(a['dialects'])})</h3><p>{_names(a['dialects'], 50)}</p></div>"
        out += f"<div><h3>Ops ({len(a['ops'])})</h3><p>{_names(a['ops'], 200)}</p></div>"
        out += (
            f"<div><h3>Manifest</h3><p>package {text_or_nr(manifest.get('package_id'))}, language "
            f"{text_or_nr(manifest.get('language'))}<br>entry points {esc(manifest.get('entrypoints') or {}) or NR}"
            f"<br>commands {_names(manifest.get('commands') or [])}"
            f"<br>components {_names(manifest.get('components') or [])}"
            f"<br>optimization surfaces {_names(manifest.get('surfaces') or [])}</p></div>"
            if a.get("manifest")
            else f"<div><h3>Manifest</h3><p>manifest.yaml: {NR}.</p></div>"
        )
        out += "</div>"
        biggest = sorted(a["loc"].items(), key=lambda kv: -kv[1])[:40]
        out += details(
            f"LOC by file ({a['files']} source files, {a['loc_total']:,} lines)",
            table(["file", "lines"], [[mono(p), f"{n:,}"] for p, n in biggest], (1,)),
        )
    return out + "</section>"


def family_section(detail: Mapping[str, Any], now: float) -> str:
    out = '<section id="families"><h2>Pass rate by capsule family over grades</h2>'
    rates = detail.get("family_rates") or {}
    families = detail.get("families") or {}
    if not rates:
        return out + f"<p>Family pass rates: {NR} (no graded capsules).</p></section>"
    out += (
        f'<p class="sub">Families from each capsule\'s own capsule.yaml under {len(detail.get("family_roots") or [])} '
        f'corpus root(s); {len(families)} capsules resolved, the rest are "not recorded".</p>'
    )
    series = {k: v for k, v in sorted(rates.items(), key=lambda kv: kv[0] == "not recorded")}
    return (
        out
        + C.time_series(
            series, label="pass share by family", y_max=1.0, y_fmt=lambda v: f"{v:.0%}", now=now, height=240
        )
        + "</section>"
    )


def cost_section(detail: Mapping[str, Any], activity: Mapping[str, Any] | None) -> str:
    cost = (detail.get("cost") or {}).get("cost")
    timing = (detail.get("cost") or {}).get("timing")
    out = '<section id="cost"><h2>Tokens and time</h2>'
    live_usage = (detail.get("cost") or {}).get("live_usage")
    if cost:
        tiles = [
            tile(
                "Tokens total",
                num(cost.get("tokens_total")),
                f"input {num(cost.get('tokens_input'))}, cached {num(cost.get('tokens_cached'))}, out "
                f"{num(cost.get('tokens_output'))}",
            ),
            tile("Tool calls", num(cost.get("tool_calls")), f"usage complete {text_or_nr(cost.get('usage_complete'))}"),
            tile(
                "Wall time",
                hours((cost.get("wall_time_seconds") or 0) / 3600) if cost.get("wall_time_seconds") else NR,
                f"active {text_or_nr(cost.get('active_wall_s'))} s, rate-limit wait "
                f"{text_or_nr(cost.get('rate_limit_wait_s'))} s",
            ),
            tile(
                "Cost",
                text_or_nr(cost.get("subscription_notional_usd") or cost.get("estimated_cost_usd")),
                f"{text_or_nr(cost.get('billing_mode'))} (USD)",
            ),
        ]
        out += '<div class="tiles">' + "".join(tiles) + "</div>"
    else:
        out += f"<p>cost_time_toolcalls.yaml: {NR}.</p>"
    if live_usage:
        out += "<p>Live cumulative provider usage: input includes cached tokens; reasoning is a subset of output. "
        out += "USD charge is not reported for this subscription run. Rates below describe client-observed "
        out += "usage-sample traffic, not server decode throughput or TTFT. Unmatched agents are excluded.</p>"
        rows = []
        for actor in live_usage["threads"]:
            tokens = actor["tokens"]
            rows.append(
                [
                    esc(actor["thread_id"]),
                    esc(", ".join(str(v) for v in actor["rounds"])),
                    num(tokens["input_total"]),
                    num(tokens["cache_read"]),
                    num(tokens["output"]),
                    num(tokens["reasoning"]),
                    text_or_nr(actor["input_tokens_per_client_observed_s"]),
                    text_or_nr(actor["output_tokens_per_client_observed_s"]),
                    text_or_nr(actor["reasoning_share_of_output"]),
                    esc(actor["sampled_through"]),
                    NR,
                ]
            )
        out += "<h3>Observed agent threads</h3>" + table(
            [
                "thread",
                "rounds",
                "input incl. cache",
                "cached",
                "output",
                "reasoning subset",
                "input tok/s",
                "output tok/s",
                "reasoning/output",
                "usage through",
                "USD charge",
            ],
            rows,
            (2, 3, 4, 5, 6, 7, 8),
        )
    if timing:
        out += (
            f"<p>Wall split ({text_or_nr(timing.get('method'))}): think/generate "
            f"{text_or_nr(timing.get('think_generate_s'))} s, tools and waiting "
            f"{text_or_nr(timing.get('tool_and_wait_s'))} s ({text_or_nr(timing.get('think_pct'))}% "
            "no outstanding tool; think/generate proxy); "
            f"{text_or_nr(timing.get('tool_calls_matched'))} calls matched, "
            f"{text_or_nr(timing.get('tool_calls_unterminated'))} unterminated.</p>"
        )
        tools = [
            [
                esc(name),
                *(
                    text_or_nr(v.get(k))
                    for k in ("calls_started", "errors", "duration_p50_s", "duration_p90_s", "seconds_sum")
                ),
            ]
            for name, v in (detail["cost"].get("by_tool") or {}).items()
        ]
        out += details(
            "By tool", table(["tool", "calls", "errors", "p50 s", "p90 s", "seconds"], tools, (1, 2, 3, 4, 5))
        )
    else:
        out += f"<p>timing_detailed.json: {NR}.</p>"
    if activity and activity.get("rounds"):
        bars_rows = [
            (
                f"round {r['round']}",
                r["output_tokens"],
                f"round {r['round']}: {r['output_tokens']:,} output, "
                f"{r['input_tokens']:,} input ({r['cached_input_tokens']:,} cached)",
            )
            for r in activity["rounds"]
            if r["usage_turns"]
        ]
        from merlin.experiments.tracking.pages.markup import bars

        out += "<h3>Output tokens per round (from the stream's turn usage)</h3>" + bars(bars_rows, "tokens per round")
    rounds = (detail.get("cost") or {}).get("rounds") or []
    if rounds:
        rows = [
            [
                text_or_nr(r.get("round")),
                text_or_nr(r.get("mode")),
                f"{r.get('n_passed')}/{r.get('n_capsules')}",
                text_or_nr(r.get("tool_calls")),
                text_or_nr(r.get("tokens_input")),
                text_or_nr(r.get("tokens_output")),
                text_or_nr(r.get("agent_rc")),
            ]
            for r in rounds
        ]
        out += details(
            f"qa_loop_state.yaml rounds ({len(rows)})",
            table(["round", "mode", "passed", "tool calls", "tokens in", "tokens out", "agent rc"], rows),
        )
    return out + "</section>"


# ---- activity_html -----------------------------------------------------------------------------
# The agent-activity section of a run page: what it is doing now, a feed, and per-round tokens/time.

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
    live = activity.get("live_usage")
    if live:
        tokens_in = live["tokens"]["tokens_input_total"]
        tokens_out = live["tokens"]["tokens_output"]
    tiles = [
        tile("Rounds with a stream", str(len(activity["rounds"])), f"{len(activity['turns'])} turns"),
        tile("Commands run", f"{activity['n_commands']:,}", f"{activity['failed_commands']:,} exited non-zero"),
        tile("Self-checks", f"{counts[A.SELFCHECK]:,}", f"{counts[A.WAIT]:,} verdict waits"),
        tile("Tests / builds", f"{counts[A.TEST]:,} / {counts[A.BUILD]:,}", "labelled by the command's words"),
        tile("Files edited", f"{len(activity['edits']):,}", f"{sum(activity['edits'].values()):,} changes"),
        tile("Errors", f"{errors:,}", "error events and failed turns"),
        tile(
            "Tokens (provider cumulative)" if live else "Tokens (reported turns)",
            f"{tokens_in:,} in / {tokens_out:,} out" if usage or live else NR,
            (
                "latest matched provider-thread cumulative usage; input includes cache"
                if live
                else f"{usage} turns reported usage"
            ),
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


__all__ = [
    "compiler_section",
    "cost_section",
    "family_section",
    "feed_list",
    "now_panel",
    "section",
    "timeline_section",
]
