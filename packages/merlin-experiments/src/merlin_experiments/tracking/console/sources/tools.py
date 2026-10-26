"""Which of its tools the author agent actually uses: diagnostics, performance feedback, helpers.

Counted from the agents' own streams (completed commands, file edits and helper spawns; see
:class:`.insight_common.AgentActivity`) over the last 30 and 60 minutes and the whole run, plus the
helper concurrency and the lead's time split already derived by :mod:`.agents` and :mod:`.timeline`.
A count says a command mentioned a tool, not that the tool helped. Display telemetry.

Panel fields beyond the shared shape: ``counts`` {window: {category: n}}, ``helpers`` {working,
subagents, ceiling, mean_active}, ``split`` {think_pct, tool_pct, wait_pct}.
"""

from __future__ import annotations

from typing import Any

from ..io import num, obj
from .insight_common import activity, line, panel, row, table

#: ``(category, label, kind)``; ``perf`` categories are the performance tooling the guidance names.
SHOWN = (
    ("isa_debug", "isa_tools debug", "perf"),
    ("evaluate_variants", "evaluate_variants", "perf"),
    ("suggestions", "optimization suggestions", "perf"),
    ("ranked_list", "ranked list / perf brief", "perf"),
    ("perf_fields", "cycle / roofline fields", "perf"),
    ("ab_variants", "A/B variants", "perf"),
    ("gsim_check", "gSIM self-checks", "check"),
    ("scalability_check", "scalability self-checks", "check"),
    ("selfcheck", "all self-checks", "check"),
    ("sched_library", "scheduling library", "library"),
    ("iteration_notes", "iteration notes edits", "notes"),
    ("spawn", "helpers spawned", "helpers"),
    ("edit", "file edits", "edits"),
    ("command", "commands", "commands"),
)


def read(ctx) -> dict[str, Any]:
    tracker = activity(ctx)
    now = ctx.now
    counts = {"30m": tracker.counts(now, 1800), "60m": tracker.counts(now, 3600), "all": tracker.counts(now, None)}
    totals = obj(obj(ctx.shared.get("agents")).get("totals"))
    concurrency = obj(obj(ctx.shared.get("timeline")).get("concurrency"))
    active = [v for v in concurrency.get("helpers") or [] if isinstance(v, int | float)]
    recent = active[-20:]
    mean_active = round(sum(recent) / len(recent), 2) if recent else None
    lead = obj(obj(ctx.shared.get("timeline")).get("lead"))
    parts = {k: num(lead.get(k)) or 0 for k in ("think_s", "tool_s", "wait_s")}
    whole = sum(parts.values())
    split = {k.replace("_s", "_pct"): (100 * v / whole if whole else None) for k, v in parts.items()}
    ceiling = num(totals.get("helper_ceiling"))
    commands = counts["all"].get("command", 0)
    unused = [label for key, label, kind in SHOWN if kind == "perf" and counts["all"].get(key, 0) == 0]

    table_rows = []
    for key, label, kind in SHOWN:
        level = None
        if kind == "perf" and counts["all"].get(key, 0) == 0 and commands >= 50:
            level = "warn"
        table_rows.append(
            row(
                [label, counts["30m"].get(key, 0), counts["60m"].get(key, 0), counts["all"].get(key, 0)], level, key=key
            )
        )
    summary = [
        line("commands 60m / all", f"{counts['60m'].get('command', 0)} / {commands}"),
        line(
            "perf tools never used",
            ", ".join(unused) if unused else "none",
            "warn" if unused and commands >= 50 else "ok",
        ),
        line(
            "helpers working / ceiling",
            f"{num(totals.get('helpers_working'))} / {ceiling}" if ceiling else num(totals.get("helpers_working")),
            "warn" if ceiling and mean_active is not None and mean_active < max(2, ceiling / 4) else None,
        ),
        line("agents active (recent)", mean_active),
        line(
            "lead time split",
            f"outside commands {split['think_pct']:.0f}% · command lifetime {split['tool_pct']:.0f}% "
            f"· recorded wait {split['wait_pct']:.0f}%"
            if whole
            else None,
        ),
    ]
    notes = ["counts are commands or edits that mention the tool; streams: " + str(len(tracker.files))]
    return panel(
        "AGENT TOOLS",
        summary,
        table_=table(["tool", "30m", "60m", "all"], table_rows),
        notes=notes,
        counts=counts,
        helpers={
            "working": num(totals.get("helpers_working")),
            "subagents": num(totals.get("subagents")),
            "ceiling": ceiling,
            "mean_active": mean_active,
        },
        split=split,
        unused_perf_tools=unused,
        commands=commands,
    )
