"""Render a checkpointed paired Phase 2 experiment (:mod:`.phase2_paired`) as page sections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import charts as C
from .html import NR, bars, details, esc, mono, num, table, text_or_nr, tile, when
from .records import stamp

ACTION_COLOURS = {
    "profile-tuning-member": "var(--c1)",
    "profile-whole-model": "var(--c2)",
    "analyze-command-buffers": "var(--c3)",
    "profile-reduced-global-witness": "var(--c4)",
    "tuning-gsim-feedback": "var(--c5)",
    "analyze-whole-model": "var(--c6)",
    "inspect-optimization-surfaces": "var(--c7)",
    "failed": "var(--critical)",
    "baseline": "var(--c2)",
    "candidate": "var(--c1)",
    "event": "var(--ink)",
}


def _geomean(values: Sequence[float]) -> float | None:
    values = [v for v in values if isinstance(v, int | float) and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _member_names(paired: Mapping[str, Any], label: str) -> dict[str, str]:
    """Display names: tuning members by name; held-out members by ordinal unless operator-private."""
    members = sorted({m for key, cell in paired["cells"].items() if key.endswith(":" + label) for m in cell["members"]})
    if label == "tuning" or paired.get("operator_private"):
        return {m: m for m in members}
    return {m: f"{m.partition('/')[0]}/held-out #{i + 1}" for i, m in enumerate(members)}


def _cell(member: Mapping[str, Any] | None) -> tuple[str, str, str]:
    if member is None:
        return "cell-other", "&ndash;", "no row recorded"
    pairs = member.get("pairs") or []
    ratio = _geomean([p.get("baseline_over_candidate") for p in pairs])
    comparable = all(p.get("comparable") for p in pairs) if pairs else None
    base = [p.get("baseline_cycles") for p in pairs if p.get("baseline_cycles")]
    cand = [p.get("candidate_cycles") for p in pairs if p.get("candidate_cycles")]
    carried = sum(1 for p in pairs if p.get("baseline_carried") or p.get("candidate_carried"))
    marks = []
    if member.get("single"):
        marks.append("single obs")
    elif len(pairs) > 1:
        marks.append(f"{len(pairs)} reps")
    if carried:
        marks.append(f"{carried} carried")
    wrong = [r for r in member.get("results") or () if r.get("correct") is False]
    if wrong:
        return "gap", f"&#10005; {len(wrong)} incorrect", "a gSIM row recorded correct=false"
    if ratio is None:
        return "cell-other", "pending" if not pairs else "not comparable", "no comparable pair recorded yet"
    cls = "hit" if ratio > 1.0 else "gap" if ratio < 1.0 else "cell-other"
    text = (
        f"<b>{1 / ratio:.3f}</b> cand/base<br><span class='sub'>{ratio:.3f}x speedup &middot; "
        f"{num(int(sum(cand) / len(cand)) if cand else None)} / "
        f"{num(int(sum(base) / len(base)) if base else None)}</span>"
        f"<br><span class='sub'>{esc(', '.join(marks))}</span>"
    )
    tip = (
        f"baseline/candidate geomean {ratio:.4f} over {len(pairs)} pair(s)\ncandidate cycles {cand}\n"
        f"baseline cycles {base}\ncomparable {comparable}"
    )
    return cls, text, tip


def grid_section(paired: Mapping[str, Any]) -> str:
    out = "<h3>Cells: member &times; trial, per cohort</h3>"
    out += (
        '<p class="sub">Each cell: the candidate/baseline gSIM cycle ratio (geometric mean over replicates; '
        "below 1 means the candidate is faster) and its inverse, the speedup; mean candidate / baseline cycles; "
        "and whether the member had one gSIM "
        "observation (single obs) or carried a measurement from an earlier execution.</p>"
    )
    for label in paired["labels"]:
        names = _member_names(paired, label)
        status = []
        for trial in paired["trials"]:
            cell = paired["cells"].get(f"{trial}:{label}")
            if cell is None:
                status.append(f"<th>{esc(trial)}<br><span class='sub'>not started</span></th>")
            else:
                comp = cell.get("completion") or {}
                status.append(
                    f"<th>{esc(trial)}<br><span class='sub'>{esc(cell.get('status') or 'no manifest')} "
                    f"{comp.get('reported', '?')}/{comp.get('expected', '?')}</span></th>"
                )
        rows = []
        for member, shown in names.items():
            tds = []
            for trial in paired["trials"]:
                cell = paired["cells"].get(f"{trial}:{label}")
                cls, text, tip = (
                    _cell((cell or {}).get("members", {}).get(member))
                    if cell
                    else ("cell-other", "&ndash;", "cell not started")
                )
                tds.append(f'<td class="{cls}" data-tip="{esc(tip)}">{text}</td>')
            rows.append(f"<tr><th>{mono(shown)}</th>{''.join(tds)}</tr>")
        body = "".join(rows) or f"<tr><td colspan='{len(paired['trials']) + 1}'>{NR}</td></tr>"
        out += (
            f"<h3>{esc(label)}</h3><div class='scroll'><table><thead><tr><th>member</th>{''.join(status)}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )
    return out


def roofline_section(paired: Mapping[str, Any]) -> str:
    points, members = [], []
    for trial, stage in sorted(paired["stages"].items()):
        latest: dict[str, Mapping[str, Any]] = {}
        for row in stage.get("feedback") or ():
            latest[row["member"]] = row
        for member, row in latest.items():
            if member not in members:
                members.append(member)
            if isinstance(row.get("candidate_over_roofline"), int | float):
                points.append(
                    {
                        "x": member,
                        "y": row["candidate_over_roofline"],
                        "series": trial,
                        "tip": f"{trial} {member}\ncandidate/roofline {row['candidate_over_roofline']:.3f}\n"
                        f"baseline/roofline {row.get('baseline_over_roofline')}\nlimiter {row.get('limiter')}\n"
                        f"verdict {row.get('verdict')} (round {row.get('round')})",
                    }
                )
    out = "<h3>Position against the roofline (latest tuning feedback per member)</h3>"
    if not points:
        return out + (
            f"<p>candidate_over_roofline: {NR} (no tuning-gsim-feedback document with a derived roofline "
            "in the stage directories).</p>"
        )
    return out + C.scatter(
        points,
        x_label="member",
        y_label="candidate cycles / roofline cycles",
        label="roofline position",
        categories=members,
        reference=1.0,
        reference_label="roofline (1.0)",
    )


def actions_section(paired: Mapping[str, Any], now: float) -> str:
    out = "<h3>Profile and analysis actions</h3>"
    lanes = []
    for trial, stage in sorted(paired["stages"].items()):
        spans = [
            {
                "start": a["start"],
                "end": a["end"],
                "kind": "failed" if a.get("failed") else a["action"],
                "tip": f"{trial}: {a['action']}\n{a['command']}",
            }
            for a in stage.get("actions") or ()
        ]
        lanes.append({"lane": f"{trial} actions", "spans": spans})
    if any(lane["spans"] for lane in lanes):
        out += C.gantt_legend(ACTION_COLOURS, {k: k for k in ACTION_COLOURS if "-" in k})
        out += C.gantt(lanes, now=now, colours=ACTION_COLOURS, label="profile actions")
    else:
        out += f"<p>Action times: {NR} (no stage event stream names a broker action).</p>"
    rows = []
    for trial, stage in sorted(paired["stages"].items()):
        counts: dict[str, list] = {}
        for r in stage.get("receipts") or ():
            entry = counts.setdefault(str(r.get("action")), [0, 0, 0.0])
            entry[0] += 1
            entry[1] += r.get("state") != "complete" or r.get("returncode") not in (0, None)
            entry[2] += r.get("elapsed_s") or 0.0
        for action, (n, bad, seconds) in sorted(counts.items()):
            rows.append([esc(trial), esc(action), str(n), str(bad), f"{seconds:,.1f}"])
    return (
        out
        + "<h3>Broker receipts</h3>"
        + table(["trial", "action", "calls", "refused or failed", "elapsed s"], rows, (2, 3, 4))
    )


def holdout_section(paired: Mapping[str, Any], now: float) -> str:
    events = paired.get("holdout") or []
    out = "<h3>Holdout commit and reveal</h3>"
    if not events:
        return out + f"<p>Holdout records: {NR}.</p>"
    lanes = [
        {
            "lane": "holdout",
            "points": [
                {"at": e["at"], "kind": "event", "tip": f"{e['event']}\n{stamp(e['at']) or 'no time'} ({e['source']})"}
                for e in events
            ],
        }
    ]
    candidates = [c for c in paired["chain"] if str(c.get("stage")).startswith("candidate:")]
    lanes.append(
        {
            "lane": "candidates sealed",
            "points": [{"at": c["at"], "kind": "candidate", "tip": c["stage"]} for c in candidates],
        }
    )
    rows = [
        [
            esc(e["event"]),
            when(e["at"]),
            esc(e["source"]),
            text_or_nr(e.get("members")),
            mono(str(e.get("digest") or "")[:16]) if e.get("digest") else NR,
        ]
        for e in events
    ]
    return (
        out
        + C.gantt(lanes, now=now, colours=ACTION_COLOURS, label="holdout timeline")
        + table(["event", "at", "time source", "revealed members (count)", "digest"], rows)
    )


def slots_section(paired: Mapping[str, Any], now: float) -> str:
    lanes = []
    for key, cell in sorted(paired["cells"].items()):
        by_slot: dict[int, list] = {}
        for s in cell.get("slots") or ():
            by_slot.setdefault(s["slot"], []).append(
                {
                    "start": s["start"] if s["start"] is not None else s["end"],
                    "end": s["end"],
                    "kind": s.get("arm"),
                    "tip": f"{key} execution {s.get('execution')}: {s.get('arm')} {s.get('family')}/"
                    f"{'' if paired.get('operator_private') or key.endswith(':tuning') else 'held-out '}"
                    f"{s.get('capsule') if paired.get('operator_private') or key.endswith(':tuning') else ''} "
                    f"{s.get('replicate')}\nduration {s.get('duration')} s (end = raw record write time)",
                }
            )
        for slot, spans in sorted(by_slot.items()):
            lanes.append({"lane": f"{key} slot {slot}", "spans": spans})
    out = "<h3>Measurement slots (reconstructed)</h3>"
    if not lanes:
        return out + f"<p>Raw executions: {NR}.</p>"
    return (
        out + '<p class="sub">Not a recorded slot schedule: each execution ends at its raw record\'s write time, '
        "starts its recorded duration earlier, and is packed into the campaign's declared fan-out.</p>"
        + C.gantt_legend(ACTION_COLOURS, {"baseline": "baseline arm", "candidate": "candidate arm"})
        + C.gantt(lanes, now=now, colours=ACTION_COLOURS, label="measurement slots")
    )


def executed_section(paired: Mapping[str, Any]) -> str:
    rows = []
    for trial, stage in sorted(paired["stages"].items()):
        latest: dict[str, Mapping[str, Any]] = {}
        for row in stage.get("feedback") or ():
            latest[row["member"]] = row
        for member, row in sorted(latest.items()):
            for arm, e in sorted((row.get("executed") or {}).items()):
                memory = e.get("local_memory") or {}
                rows.append(
                    [
                        esc(trial),
                        mono(member),
                        esc(arm),
                        text_or_nr(e.get("status")),
                        num(e.get("accelerator_commands")),
                        num(e.get("retired_instructions")),
                        esc(", ".join(f"{k} {v}" for k, v in (e.get("by_class") or {}).items())) or NR,
                        esc(f"{memory.get('scratchpad_rows_high_water')}/{memory.get('scratchpad_rows_capacity')}")
                        if memory
                        else NR,
                        esc(f"{memory.get('accumulator_rows_high_water')}/{memory.get('accumulator_rows_capacity')}")
                        if memory
                        else NR,
                        text_or_nr(e.get("why")),
                    ]
                )
    return "<h3>Executed commands and local memory (latest tuning feedback)</h3>" + table(
        [
            "trial",
            "member",
            "arm",
            "status",
            "accel commands",
            "retired",
            "by class",
            "scratchpad hw/cap",
            "accumulator hw/cap",
            "why",
        ],
        rows,
        (4, 5),
    )


def tokens_section(paired: Mapping[str, Any]) -> str:
    rows, bar_rows = [], []
    for trial, stage in sorted(paired["stages"].items()):
        t = stage.get("tokens") or {}
        rows.append(
            [
                esc(trial),
                num(t.get("tokens_total")),
                num(t.get("tokens_input")),
                num(t.get("tokens_cached")),
                num(t.get("tokens_output")),
                num(t.get("tool_calls")),
                text_or_nr(t.get("wall_time_seconds")),
            ]
        )
        if isinstance(t.get("tokens_total"), int):
            bar_rows.append((trial, t["tokens_total"], f"{trial}: {t['tokens_total']:,} tokens"))
    return (
        "<h3>Tokens per trial</h3>"
        + bars(bar_rows, "tokens per trial")
        + table(["trial", "total", "input", "cached", "output", "tool calls", "wall s"], rows, (1, 2, 3, 4, 5))
    )


def reference_section(paired: Mapping[str, Any]) -> str:
    present = {k: c["reference"] for k, c in paired["cells"].items() if c.get("reference")}
    out = "<h3>Reference comparison (operator-only)</h3>"
    if not present:
        return out + "<p>reference_comparison.json: none beside any cell.</p>"
    if not paired.get("operator_private"):
        return out + (
            f"<p>Present beside {len(present)} cell(s); operator-only ratios are hidden in public mode "
            "(pass --operator-private).</p>"
        )
    rows = []
    for key, ref in sorted(present.items()):
        for r in ref.get("rows") or ():
            rows.append(
                [
                    esc(key),
                    mono(f"{r.get('family')}/{r.get('capsule')}"),
                    text_or_nr(r.get("replicate")),
                    num(r.get("candidate_cycles")),
                    num(r.get("reference_cycles")),
                    text_or_nr(r.get("candidate_over_reference")),
                    text_or_nr(r.get("state")),
                ]
            )
    return out + table(
        ["cell", "member", "rep", "candidate", "reference", "candidate/reference", "state"], rows, (3, 4, 5)
    )


def section(paired: Mapping[str, Any] | None, now: float) -> str:
    if not paired:
        return ""
    stats = paired.get("statistics") or {}
    aggregate = stats.get("aggregate") or {}
    done, expected = paired["done"], paired["expected"]
    complete = sum(1 for c in paired["cells"].values() if (c.get("completion") or {}).get("complete"))
    total_cells = len(paired["trials"]) * len(paired["labels"])
    tiles = [
        tile("Checkpoints", f"{len(done)} / {len(expected)}", esc(done[-1]) if done else "none yet"),
        tile("Cells complete", f"{complete} / {total_cells}", f"{len(paired['cells'])} started"),
        tile(
            "Median speedup",
            f"{aggregate['median_speedup']:.3f}x" if isinstance(aggregate.get("median_speedup"), int | float) else NR,
            "baseline/candidate, across trials",
        ),
        tile("Sealed", "yes" if paired.get("sealed") else "no", "experiment_manifest"),
        tile(
            "Mode",
            "operator-private" if paired.get("operator_private") else "public",
            "held-out names and reference ratios" + ("" if paired.get("operator_private") else " hidden"),
        ),
    ]
    at = {c.get("stage"): c.get("at") for c in paired["chain"]}
    chain = "".join(
        f"<li>{'&#10003;' if stage in done else '&#9744;'} {mono(stage)} "
        f"{when(at.get(stage)) if stage in done else ''}</li>"
        for stage in expected
    )
    family = [
        [
            esc(k),
            f"{v.get('median_speedup'):.3f}x" if isinstance(v.get("median_speedup"), int | float) else NR,
            esc(v.get("all_trial_speedups")),
        ]
        for k, v in sorted((aggregate.get("family_aggregate") or {}).items())
    ]
    return (
        '<section id="paired"><h2>Phase 2 &middot; paired trials &times; members &times; cohorts</h2>'
        f'<p class="sub">Experiment {text_or_nr(paired.get("experiment_id"))} at {mono(paired["root"])}. Times are '
        "file modification times of sealed records (the records hold no wall clock).</p>"
        + '<div class="tiles">'
        + "".join(tiles)
        + "</div>"
        + details("Checkpoint chain", f"<ul class='feed'>{chain}</ul>")
        + grid_section(paired)
        + ("<h3>Speedup by family</h3>" + table(["family", "median", "per trial"], family) if family else "")
        + roofline_section(paired)
        + actions_section(paired, now)
        + holdout_section(paired, now)
        + slots_section(paired, now)
        + executed_section(paired)
        + tokens_section(paired)
        + reference_section(paired)
        + "</section>"
    )


__all__ = ["section"]
