"""The richer Phase 1 sections of a run page: timeline, compiler evolution, family pass rates, cost/time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import activity as A
from . import charts as C
from .html import NR, details, esc, hours, mono, num, table, text_or_nr, tile, when

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
    out += C.gantt_legend(COLOURS, NAMES)
    out += "<h3>Whole run</h3>" + C.gantt(lanes, now=now, colours=COLOURS, label="run timeline")
    starts = [s["start"] for lane in lanes for s in lane.get("spans") or () if s.get("start") is not None]
    if starts and now - min(starts) > 3 * 3600:
        out += "<h3>Last two hours</h3>" + C.gantt(
            lanes, now=now, colours=COLOURS, label="recent timeline", window=(now - 2 * 3600, now)
        )
    jobs = detail.get("jobs") or []
    if jobs:
        rows = [
            [
                esc(j["kind"]),
                esc(j["sim"]),
                mono(j["id"]),
                when(j.get("requested")),
                when(j.get("started")),
                when(j.get("ended")),
                esc(j["state"]),
                text_or_nr(j.get("all_pass")),
                text_or_nr(j.get("capsules")),
            ]
            for j in reversed(jobs[-150:])
        ]
        out += details(
            f"Simulator jobs and self-check requests ({len(jobs)}; ends are marker-file times)",
            table(["kind", "engine", "id", "requested", "started", "ended", "state", "all pass", "capsules"], rows),
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
    if cost:
        tiles = [
            tile(
                "Tokens total",
                num(cost.get("tokens_total")),
                f"in {num(cost.get('tokens_input'))}, cached {num(cost.get('tokens_cached'))}, out "
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
    if timing:
        out += (
            f"<p>Wall split ({text_or_nr(timing.get('method'))}): think/generate "
            f"{text_or_nr(timing.get('think_generate_s'))} s, tools and waiting "
            f"{text_or_nr(timing.get('tool_and_wait_s'))} s ({text_or_nr(timing.get('think_pct'))}% "
            f"thinking); {text_or_nr(timing.get('tool_calls_matched'))} calls matched, "
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
        from .html import bars

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


__all__ = ["compiler_section", "cost_section", "family_section", "timeline_section"]
