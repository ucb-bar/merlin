"""Render a tracking summary as ONE self-contained HTML page: inline CSS, inline SVG, a few lines of JS.

No network, no CDN, no external file: the page opens from disk or through an ssh port-forward, and a
copy mailed to someone renders the same.  Charts are SVG drawn here from the summary's numbers; the
only script shows a hover tooltip, so the page is complete with scripts disabled (every chart also
has a table view).  A value the records do not hold is printed as "not recorded" -- never a zero.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from merlin.experiments.tracking.readers import records as R

# ---- html --------------------------------------------------------------------------------------


def _phase2_tiles(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, badge, digest, esc, num, text_or_nr, tile, when

    live, best, bar = p2.get("liveness") or {}, p2.get("best") or {}, p2.get("bar") or {}
    counts, in_run = p2.get("counts") or {}, p2.get("counts_in_run")
    board = p2.get("board")
    ratio = (
        f" &middot; {best['cycles'] / bar['cycles']:.3f}x the bar" if best.get("cycles") and bar.get("cycles") else ""
    )
    if live.get("last_measured"):
        last = when(live["last_measured"])
    else:
        last = "none since the run started" if live.get("run_started") else NR
    failures = sum(counts.get(k, 0) for k in R.FAILURE_CLASSES)

    def this_run(value: int | None) -> str:
        return f"this run: {value}" if in_run is not None else "this run: not recorded"

    tiles = [
        tile("State", badge(live.get("state")), esc(live.get("detail") or "")),
        tile(
            "Last measured candidate (this run)",
            last,
            f"run started {when(live.get('run_started'))}; store's last {when(live.get('last_measured_store'))}",
        ),
        tile("Best so far (valid)", num(best.get("cycles")), digest(best.get("key")) + ratio),
        tile("Bar (reference, same machine)", num(bar.get("cycles")), text_or_nr(bar.get("label"))),
        tile("Measured", f"{counts.get(R.MEASURED, 0):,}", this_run((in_run or {}).get(R.MEASURED, 0))),
        tile("Failed or refused", f"{failures:,}", this_run(sum((in_run or {}).get(k, 0) for k in R.FAILURE_CLASSES))),
        tile(
            "Board queue",
            f"{board.get('queue', 0):,}" if board else NR,
            f"oldest waiting since {when(board.get('oldest_waiting'))}" if board else "",
        ),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _store_line(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, esc, mono

    store = p2.get("store") or {}
    if not store.get("root"):
        return (
            f'<p class="sub">Measurement store: {NR} (the run records no <span class="mono">store_roots.screen</span>;'
            ' pass <span class="mono">--store</span>).</p>'
        )
    line = f'<p class="sub">Measurement store {mono(store["root"])} (from {esc(store.get("source"))}).</p>'
    if not store.get("loaded"):
        line += '<p class="sub">The recorded store directory does not exist.</p>'
    return line


def _reading(c: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, num

    if c.get("cycles"):
        return num(c["cycles"])
    if c.get("whole_window"):
        return num(c["whole_window"]) + " (wrong)"
    return NR


def _in_run(c: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR

    return {True: "yes", False: "no"}.get(c.get("in_run"), NR)


def _candidates_table(candidates: Sequence[Mapping[str, Any]]) -> str:
    from merlin.experiments.tracking.pages.markup import _at, details, digest, esc, share, table, text_or_nr, when

    rows = [
        [
            when(_at(c) if c.get("finished") or c.get("requested") else None),
            digest(c["key"], 14),
            text_or_nr(c.get("label")),
            esc(c["class"]),
            _reading(c),
            share((c.get("coverage") or {}).get("priced_share")),
            _in_run(c),
            text_or_nr(c.get("reason")),
        ]
        for c in reversed(candidates[-80:])
    ]
    headers = ["finished", "job", "label", "class", "cycles", "pkg-authored", "this run", "reason"]
    return details("Table view (newest 80 store jobs)", table(headers, rows, numeric=(4, 5)))


def _outcomes(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import _at, bars, details, digest, esc, table, text_or_nr, when

    candidates = p2.get("candidates") or []
    counts, in_run = p2.get("counts") or {}, p2.get("counts_in_run")
    rows = []
    for name in (R.MEASURED, *R.FAILURE_CLASSES, R.OPEN, R.SUPERSEDED):
        tip = f"{counts.get(name, 0)} of {len(candidates)} store jobs"
        if in_run is not None:
            tip += f"; {in_run.get(name, 0)} in this run"
        rows.append((name, counts.get(name, 0), tip))
    failures = [
        [when(_at(c)), digest(c["key"], 14), esc(c["class"]), text_or_nr(c.get("reason"))]
        for c in reversed([c for c in candidates if c["class"] in R.FAILURE_CLASSES][-30:])
    ]
    return bars(rows, "store jobs by outcome class") + details(
        "Latest failures with their recorded reason", table(["at", "job", "class", "reason"], failures)
    )


def _roofline(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, details, digest, esc, num, table, text_or_nr

    roofline = p2.get("roofline")
    if not roofline:
        return (
            f"<p>Per-form roofline gap: {NR} (no result.json in the store carries "
            '<span class="mono">diagnostics.per_group</span> roofline rows).</p>'
        )
    kinds = [
        [
            esc(k["kind"]),
            str(k["groups"]),
            num(k["ours"]),
            num(k["roofline"]),
            f"{k['ours'] / k['roofline']:.2f}x" if k["roofline"] else NR,
        ]
        for k in roofline["by_kind"]
    ]
    groups = [
        [
            esc(g["group"]),
            text_or_nr(g.get("kind")),
            num(g["ours"]),
            num(g["roofline"]),
            f"{g['over_roofline']}x",
            text_or_nr(g.get("limiter")),
        ]
        for g in roofline["gaps"]
    ]
    return (
        '<p class="sub">From the lowest measured candidate whose result records '
        f'<span class="mono">diagnostics.per_group</span> rooflines ({digest(roofline["key"])}).</p>'
        + table(["form (kind)", "groups", "ours", "roofline", "over roofline"], kinds, numeric=(1, 2, 3, 4))
        + details(
            "Per group (largest gap first)",
            table(["group", "kind", "ours", "roofline", "over", "limiter"], groups, numeric=(2, 3, 4)),
        )
    )


def _plateau(p2: Mapping[str, Any], now: float) -> str:
    from merlin.experiments.tracking.pages.markup import details, digest, hours, missing, num, table, text_or_nr, when

    plateau = p2.get("plateau")
    if not plateau:
        return missing("Plateau record", "plateau.json")
    rule = plateau.get("rule") or {}
    last = plateau.get("last_improvement") or {}
    anchor = R.epoch(last.get("epoch")) or R.epoch(last.get("at"))
    since_now = (now - anchor) / 3600.0 if anchor else None
    sessions = plateau.get("sessions") or []
    trace = [
        [
            when(s["at"]),
            text_or_nr(s.get("run")),
            text_or_nr(s.get("session")),
            num(s.get("cycles")),
            text_or_nr(s.get("model")),
            "reset" if s.get("reset") else text_or_nr(s.get("abandoned")),
        ]
        for s in reversed(sessions[-60:])
    ]
    return (
        f"<p>Rule: no eligible improvement for <b>{text_or_nr(rule.get('hours'))} h</b> over at least "
        f"<b>{text_or_nr(rule.get('min_sessions'))}</b> sessions. Last improvement {when(anchor)} "
        f"({num(last.get('cycles'))} cycles, {digest(last.get('package_sha256'))}); as recorded "
        f"{text_or_nr(last.get('hours_since'))} h / {text_or_nr(last.get('sessions_since'))} sessions since; "
        f"now {hours(since_now)}.</p>"
        + details(
            f"Session trace ({len(sessions)} rows, newest first)",
            table(["at", "run", "session", "best cycles at start", "model", "note"], trace, numeric=(3,)),
        )
    )


def _sessions(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import details, digest, missing, table, text_or_nr

    sessions = p2.get("sessions")
    if sessions:
        stopped = sessions.get("stopped") or {}
        out = (
            f"<p>Sessions: {len(sessions.get('rows') or [])} recorded; driver {text_or_nr(sessions.get('driver'))}, "
            f"model {text_or_nr(sessions.get('model'))}; stopped: {text_or_nr(stopped.get('kind'))} "
            f"{text_or_nr(stopped.get('reason'))}.</p>"
        )
    else:
        out = missing("Session record", "stage/sessions.json")
    breaker = p2.get("circuit_breaker")
    if breaker:
        out += (
            f"<p>Infra circuit breaker tripped {text_or_nr(breaker.get('at'))}: {text_or_nr(breaker.get('reason'))}</p>"
        )
    if p2.get("rounds"):
        rows = [
            [
                text_or_nr(r["round"]),
                text_or_nr(r["status"]),
                text_or_nr(r["why"]),
                text_or_nr(r["requested"]),
                digest(r["final_package"]),
                text_or_nr(r["final_timing_status"]),
            ]
            for r in p2["rounds"]
        ]
        headers = ["round", "status", "why", "requested", "final package", "final status"]
        out += details(f"Authoring rounds ({len(rows)})", table(headers, rows))
    return out


def _board(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, details, digest, hours, num, table, text_or_nr, when

    board = p2.get("board")
    if not board:
        return f"<p>Board queue and outages: {NR}.</p>"
    runner = board.get("batch_runner") or {}
    if runner:
        present = "process present now" if runner.get("process_present") else "no such process at generation time"
        runner_text = f"pid {text_or_nr(runner.get('pid'))} started {when(runner.get('started'))}; {present}"
    else:
        runner_text = NR
    summary = (
        f"<p>Waiting for the board: <b>{board.get('queue', 0)}</b> (oldest since {when(board.get('oldest_waiting'))});"
        f" running {board.get('running', 0)}; pending/screening {board.get('pending', 0)}. "
        f"Solo streak {num(board.get('solo_streak'))}. Batches run: {board.get('batches', 0)}. "
        f"Batch runner: {runner_text}.</p>"
    )
    outage = board.get("open_outage")
    if outage:
        summary += (
            f"<p><b>Board outage open</b> since {when(outage['opened'])}: {outage['failures']} failures; next try "
            f"{when(outage['retry_after'])}. Last: {text_or_nr(outage['last_reason'])}</p>"
        )
    else:
        summary += "<p>No board outage open.</p>"
    closed = [
        [
            when(o["opened"]),
            when(o["closed"]),
            hours((o["closed"] - o["opened"]) / 3600 if o["closed"] and o["opened"] else None),
            str(o["failures"]),
            text_or_nr(o["last_reason"]),
        ]
        for o in reversed(board.get("closed_outages") or [])
    ]
    batches = [
        [
            digest(b["id"], 40),
            str(b["jobs"]),
            text_or_nr(b["control_ok"]),
            text_or_nr(b["control_ratio"]),
            "yes" if b["board_unavailable"] else "no",
            text_or_nr(b["failure"]),
        ]
        for b in reversed(board.get("recent_batches") or [])
    ]
    return (
        summary
        + table(["outage opened", "closed", "duration", "failures", "last reason"], closed, numeric=(3,))
        + details(
            "Recent batches",
            table(["batch", "jobs", "control ok", "control ratio", "board unavailable", "failure"], batches),
        )
    )


def _node(key: str, value: str) -> str:
    from merlin.experiments.tracking.pages.markup import esc

    return f'<div class="node"><div class="k">{esc(key)}</div>{value}</div>'


def _chain(nodes: Sequence[str]) -> str:
    return '<div class="chain">' + '<span class="arrow">&rarr;</span>'.join(nodes) + "</div>"


def _phase2_lineage(p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, digest, missing, mono, num, text_or_nr, when

    lineage, ledger = p2.get("lineage"), p2.get("ledger")
    parts = ['<section id="lineage"><h2>Champion lineage</h2>']
    if lineage:
        nodes = [
            _node(
                "resumed from",
                f"{mono(row['run'].rstrip('/').rpartition('/')[2])}<br>seed {digest(row.get('seed_package'))}",
            )
            for row in reversed(lineage["chain"])
        ]
        kind = lineage.get("lineage_kind") or "kind not recorded"
        origin = text_or_nr(lineage.get("origin_kind"))
        nodes.append(_node(f"this run's seed ({kind})", f"{digest(lineage.get('seed_package'))}<br>origin {origin}"))
        for event in (ledger or {}).get("best") or ():
            nodes.append(
                _node(
                    f"best moved (candidate {event.get('n')})",
                    f"{digest(event.get('package'))}<br>{when(event.get('at'))}",
                )
            )
        index_rows = lineage.get("index_rows")
        for row in (index_rows or {}).get("champions") or ():
            cycles = num((row.get("firesim") or {}).get("cycles"))
            nodes.append(_node("exported champion", f"{text_or_nr(row.get('package_id'))}<br>{cycles} cycles"))
        parts.append(_chain(nodes))
        parts.append(f'<p class="sub">Why this run exists: {text_or_nr(lineage.get("why"))}</p>')
        if index_rows is None:
            parts.append(f"<p>INDEX.yaml rows citing this run: {NR}.</p>")
    else:
        parts.append(missing("Seed lineage", "resumed_seed.json"))
    if ledger is None:
        parts.append(missing("OOT ledger", "iterations.jsonl"))
    else:
        parts.append(
            f"<p>OOT ledger: {ledger['candidates']} candidate commits, {len(ledger['measured'])} measured tags, "
            f"{len(ledger['best'])} best moves.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def phase2_section(summary: Mapping[str, Any], p2: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import coverage_chart, cycles_chart

    now = summary["generated"]
    return (
        '<section id="phase2"><h2>Phase 2 &middot; measured candidates</h2>'
        + _phase2_tiles(p2)
        + _store_line(p2)
        + "<h3>Cycles per candidate over time</h3>"
        + cycles_chart(p2, now)
        + _candidates_table(p2.get("candidates") or [])
        + "<h3>Package-authored coverage (priced share) beside the cycles</h3>"
        + coverage_chart(p2, now)
        + "<h3>Outcome by class</h3>"
        + _outcomes(p2)
        + "<h3>Distance to the roofline, per form</h3>"
        + _roofline(p2)
        + "<h3>Plateau and liveness</h3>"
        + _plateau(p2, now)
        + _sessions(p2)
        + "<h3>Board queue and outages</h3>"
        + _board(p2)
        + "</section>"
        + _phase2_lineage(p2)
    )


def _tier_colour(index: int, count: int) -> str:
    return "var(--t3)" if count <= 1 else f"var(--t{round(index * 6 / (count - 1))})"


def progression_chart(grades: Sequence[Mapping[str, Any]]) -> str:
    from merlin.experiments.tracking.pages.markup import (
        NR,
        Scale,
        _mark,
        _svg,
        _x_time_axis,
        _y_axis,
        esc,
        legend,
        nice_ticks,
    )

    points = [g for g in grades if g.get("at") is not None and g.get("n_passed") is not None]
    if not points:
        return f"<p>Score progression: {NR}.</p>"
    t0, t1 = points[0]["at"], points[-1]["at"]
    pad = max(600.0, (t1 - t0) * 0.04)
    peak = max([g.get("n_capsules") or 0 for g in points] + [g["n_passed"] for g in points] + [1])
    width, height, left, right, top, bottom = 960, 230, 52, 940, 12, 196
    x, y = Scale(t0 - pad, t1 + pad, left, right), Scale(0, peak * 1.05, bottom, top)
    parts = [
        _y_axis(y, nice_ticks(0, peak, 5), left, right, lambda v: f"{v:g}"),
        _x_time_axis(x, t0 - pad, t1 + pad, bottom),
    ]
    totals = [g for g in points if g.get("n_capsules") is not None]
    if totals:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(g['at']):.1f} {y(g['n_capsules']):.1f}" for i, g in enumerate(totals)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    entries = [("var(--muted)", "capsules in grade")]
    notes = []
    tiers = sorted({t for g in points for c in g.get("capsules", ()) for t in c.get("tiers", {})}, key=R.tier_key)
    for i, tier in enumerate(tiers):
        colour = f"var(--t{i % 7})"
        observed = []
        unavailable = 0
        for g in points:
            statuses = [c.get("tiers", {}).get(tier) for c in g.get("capsules", ())]
            measured = [s for s in statuses if s in ("pass", "fail")]
            unavailable = sum(s == "unavailable" for s in statuses)
            if measured:
                observed.append((g, sum(s == "pass" for s in measured), len(measured)))
        if not observed:
            notes.append(f"{esc(tier)}: unavailable / not recorded ({unavailable} unavailable in latest grade)")
            continue
        path = " ".join(f"{'M' if j == 0 else 'L'}{x(g['at']):.1f} {y(n):.1f}" for j, (g, n, _) in enumerate(observed))
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for g, n, total in observed:
            tip = f"{tier}: {n}/{total} recorded pass/fail results\n{g['name']}\n{R.stamp(g['at'])}"
            parts.append(_mark(x(g["at"]), y(n), tip, "measured"))
        entries.append((colour, f"{tier} capsules passed"))
        if unavailable:
            notes.append(f"{esc(tier)}: {unavailable} unavailable in latest grade")
    return (
        legend(entries)
        + _svg(width, height, "".join(parts), "capsules passed per tier per grade")
        + ('<p class="sub">' + "; ".join(notes) + "</p>" if notes else "")
    )


def _heat_cell(name: str, index: int, grade: Mapping[str, Any], tiers: Sequence[str]) -> tuple[str, str]:
    row = next((c for c in grade["capsules"] if c["capsule"] == name), None)
    where = f"{name}\ngrade {index + 1} ({grade['name']})"
    if row is None:
        return "var(--surface-2)", f"{where}: not in this grade"
    if row["status"] == "pass":
        position = tiers.index(row["highest_pass"]) if row["highest_pass"] in tiers else 0
        return _tier_colour(position, len(tiers)), f"{where}: pass, highest tier {row['highest_pass']}"
    reason = f"{row.get('failure_plane') or ''} {row.get('failure_category') or ''}".strip()
    return "var(--fail)", f"{where}: {row['status']}, highest tier passed {row['highest_pass'] or 'none'}\n{reason}"


def heatmap(p1: Mapping[str, Any], limit: int = 40) -> str:
    from merlin.experiments.tracking.pages.markup import NR, _svg, _text, esc, legend

    grades = [g for g in p1.get("grades") or () if g.get("capsules")][-limit:]
    tiers = list(p1.get("tiers") or [])
    if not grades:
        return f"<p>Capsule history: {NR}.</p>"
    names = sorted({c["capsule"] for g in grades for c in g["capsules"]})
    width, left, top, cell_h = 960, 300, 18, 12
    cell_w = max(8, min(28, (width - left - 10) // len(grades)))
    parts = [_text(left + j * cell_w + cell_w / 2, 12, str(j + 1), "middle") for j in range(len(grades))]
    for i, name in enumerate(names):
        parts.append(_text(left - 6, top + i * cell_h + 10, name[:48], "end"))
        for j, grade in enumerate(grades):
            fill, tip = _heat_cell(name, j, grade, tiers)
            parts.append(
                f'<rect data-tip="{esc(tip)}" x="{left + j * cell_w + 1}" y="{top + i * cell_h + 1}" '
                f'width="{cell_w - 2}" height="{cell_h - 2}" rx="2" fill="{fill}"/>'
            )
    entries = [(_tier_colour(i, len(tiers)), f"pass, highest tier {esc(t)}") for i, t in enumerate(tiers)]
    entries += [("var(--fail)", "not passing (status shown on hover)"), ("var(--surface-2)", "not in that grade")]
    note = f'<p class="sub">Columns are grades in time order (the last {len(grades)}).</p>'
    return (
        legend(entries) + _svg(width, top + cell_h * len(names) + 6, "".join(parts), "capsule status per grade") + note
    )


def _tier_cell(status: str | None) -> str:
    from merlin.experiments.tracking.pages.markup import esc

    if status is None:
        return '<td class="cell-other">&ndash;</td>'
    if status == "pass":
        return '<td class="cell-pass">&#10003; pass</td>'
    if status == "fail":
        return '<td class="cell-fail">&#10005; fail</td>'
    return f'<td class="cell-other">{esc(status)}</td>'


def tier_matrix(latest: Mapping[str, Any], tiers: Sequence[str]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, esc, mono, text_or_nr

    if not latest.get("capsules"):
        return f"<p>Capsule &times; tier matrix: {NR}.</p>"
    rows = sorted(latest["capsules"], key=lambda c: (c["status"] == "pass", c["capsule"]))
    head = "".join(f"<th>{esc(t)}</th>" for t in tiers)
    body = "".join(
        f"<tr><td>{mono(c['capsule'])}</td><td>{text_or_nr(c.get('label'))}</td>"
        + "".join(_tier_cell(c["tiers"].get(t)) for t in tiers)
        + f"<td>{text_or_nr(c['status'])}</td><td>{text_or_nr(c.get('failure_plane'))}</td></tr>"
        for c in rows
    )
    header = f"<tr><th>capsule</th><th>set</th>{head}<th>status</th><th>failure plane</th></tr>"
    return f'<div class="scroll"><table><thead>{header}</thead><tbody>{body}</tbody></table></div>'


def _phase1_tiles(p1: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, badge, esc, text_or_nr, tile, when

    live, latest, plateau = p1.get("liveness") or {}, p1.get("latest") or {}, p1.get("plateau")
    if latest.get("n_passed") is not None:
        score = f"{latest['n_passed']} / {latest['n_capsules']}"
        when_graded = f"{esc(latest.get('name', ''))} &middot; {when(latest.get('at'))}"
    else:
        score, when_graded = NR, ""
    tiles = [
        tile("State", badge(live.get("state")), esc(live.get("detail") or "")),
        tile("Latest grade", score, when_graded),
        tile("Grades recorded", str(len(p1.get("grades") or [])), ""),
        tile("Highest tier (latest)", text_or_nr(latest.get("highest_tier")), ""),
        tile(
            "Plateau",
            ("stuck" if plateau.get("stuck") else "progressing") if plateau else NR,
            text_or_nr(plateau.get("sentence")) if plateau else "",
        ),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _phase1_end(p1: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, details, digest, esc, missing, table, text_or_nr, when

    freeze, manifest, loop, commits = (
        p1.get("freeze"),
        p1.get("manifest"),
        p1.get("qa_loop_summary"),
        p1.get("oot_commits"),
    )
    out = "<h3>Freeze, formal grade and OOT history</h3>"
    if freeze:
        error = f"; OOT error {esc(freeze['oot_error'])}" if freeze.get("oot_error") else ""
        out += (
            f"<p>Frozen {when(freeze['at'])}: submission {digest(freeze.get('submission_sha256'))}, "
            f"commit {digest(freeze.get('frozen_commit'))}{error}.</p>"
        )
    else:
        out += missing("Freeze", "freeze.json")
    if manifest:
        out += (
            f"<p>Formal grade: public {esc(manifest['public'])}, hidden {esc(manifest['hidden'])}, complete "
            f"{text_or_nr(manifest['formal_grade_complete'])}; failures "
            f"{text_or_nr(', '.join(manifest['completion_failures']))}; model {text_or_nr(manifest.get('model'))}; "
            f"wall {text_or_nr(manifest.get('wall_time_seconds'))} s.</p>"
        )
    else:
        out += missing("Formal manifest", "run_manifest.yaml")
    if loop:
        out += (
            f"<p>QA loop: {text_or_nr(loop['n_rounds'])} rounds, converged {text_or_nr(loop['converged'])}, "
            f"formal complete {text_or_nr(loop['formal_complete'])}, feedback healthy "
            f"{text_or_nr(loop['feedback_healthy'])}.</p>"
        )
    else:
        out += missing("QA loop summary", "qa_loop_summary.yaml")
    if commits is None:
        return out + missing("OOT commit log", "oot_commits.jsonl")
    rows = [
        [
            when(r["at"]),
            text_or_nr(r["label"]),
            text_or_nr(r["key"]),
            digest(r["commit"]),
            f"{r['n_passed']}/{r['n_capsules']}" if r["n_passed"] is not None else NR,
            text_or_nr(r["error"]),
        ]
        for r in reversed(commits)
    ]
    return out + details(f"OOT commits ({len(rows)})", table(["at", "label", "key", "commit", "passed", "error"], rows))


def phase1_section(p1: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import bars, details, esc, mono, num, table, text_or_nr, when

    latest, tiers = p1.get("latest") or {}, p1.get("tiers") or []
    grades = [
        [esc(g["name"]), when(g["at"]), num(g["n_passed"]), num(g["n_capsules"]), text_or_nr(g["highest_tier"])]
        for g in reversed(p1.get("grades") or [])
    ]
    failing = [
        [
            mono(c["capsule"]),
            text_or_nr(c["status"]),
            text_or_nr(c.get("highest_pass")),
            text_or_nr(c.get("failure_plane")),
            text_or_nr(c.get("failure_category")),
            text_or_nr(c.get("failure_detail")),
        ]
        for c in p1.get("failing") or ()
    ]
    planes = latest.get("first_failure_planes") or {}
    plane_rows = [
        (str(k), int(v), f"{v} capsules first fail at {k}")
        for k, v in sorted(planes.items(), key=lambda kv: -int(kv[1]))
    ]
    return (
        '<section id="phase1"><h2>Phase 1 &middot; capsule grades</h2>'
        + _phase1_tiles(p1)
        + "<h3>Score progression</h3>"
        + progression_chart(p1.get("grades") or [])
        + details("Table view", table(["grade", "at", "passed", "graded", "highest tier"], grades, numeric=(2, 3)))
        + "<h3>Capsules over grades</h3>"
        + heatmap(p1)
        + "<h3>Capsule &times; tier, latest grade</h3>"
        + tier_matrix(latest, tiers)
        + "<h3>Where the latest grade fails first</h3>"
        + bars(plane_rows, "first failure planes")
        + "<h3>Failing capsules and their recorded reasons</h3>"
        + table(["capsule", "status", "highest tier passed", "plane", "category", "detail"], failing)
        + _phase1_end(p1)
        + "</section>"
    )


def orchestration_section(record: Mapping[str, Any] | None) -> str:
    from merlin.experiments.tracking.pages.markup import esc, mono, table, text_or_nr, when

    if not record:
        return ""

    def pid(a: Mapping[str, Any]) -> str:
        present = {None: "", True: " (present now)", False: " (absent now)"}[a["process_present"]]
        return text_or_nr(a["pid"]) + present

    attempts = [
        [
            text_or_nr(a["phase"]),
            text_or_nr(a["adapter"]),
            text_or_nr(a["state"]),
            when(a["started"]),
            when(a["ended"]),
            text_or_nr(a["returncode"]),
            pid(a),
        ]
        for a in record.get("attempts") or ()
    ]
    phases = [
        [esc(n), text_or_nr(p["adapter"]), text_or_nr(p["state"]), mono(p["engine_output"])]
        for n, p in sorted((record.get("phases") or {}).items())
    ]
    return (
        '<section id="orchestration"><h2>Orchestration</h2>'
        f"<p>Experiment {text_or_nr(record.get('experiment'))}, state {text_or_nr(record.get('state'))}, "
        f"frozen {text_or_nr(record.get('frozen_at'))}; plan {esc(record.get('plan_binding'))}.</p>"
        + table(["phase", "adapter", "state", "engine output"], phases)
        + "<h3>Attempts</h3>"
        + table(["phase", "adapter", "state", "started", "ended", "exit", "pid"], attempts)
        + "</section>"
    )


def inventory_section(rows: Sequence[Mapping[str, Any]]) -> str:
    from merlin.experiments.tracking.pages.markup import details, esc, mono, table, text_or_nr

    body = [[esc(r["record"]), esc(r["state"]), text_or_nr(r.get("detail")), mono(r["path"])] for r in rows]
    absent = sum(1 for r in rows if r["state"] != "read")
    return (
        '<section id="records"><h2>Records read</h2>'
        f'<p class="sub">{len(rows)} records consulted, {absent} absent, unreadable or of an unexpected schema. '
        "Nothing was measured, graded or rebuilt to draw this page.</p>"
        + details("Inventory", table(["record", "state", "detail", "path"], body))
        + "</section>"
    )


def page(title: str, heading: str, sub: str, nav: Sequence[tuple[str, str]], body: str, generated: float) -> str:
    from merlin.experiments.tracking.pages.markup import CSS, JS, esc

    links = "".join(f'<a href="#{esc(anchor)}">{esc(label)}</a>' for anchor, label in nav)
    head = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
    )
    footer = (
        f'<footer>Generated {esc(R.stamp(generated))} by <span class="mono">merlin-experiment dashboard</span> '
        "from existing records only.</footer>"
    )
    return (
        f'{head}<header><h1>{heading}</h1><span class="sub">{sub}</span></header><nav>{links}</nav>{body}{footer}'
        f'</main><div id="tip" hidden></div><script>{JS}</script></body></html>\n'
    )


def run_body(summary: Mapping[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    from merlin.experiments.tracking.pages.markup import NR, esc

    sections, nav = [], []
    if summary.get("phase2"):
        sections.append(phase2_section(summary, summary["phase2"]))
        nav += [("phase2", "Phase 2"), ("lineage", "Lineage")]
    if summary.get("phase1"):
        sections.append(phase1_section(summary["phase1"]))
        nav.append(("phase1", "Phase 1"))
    if summary.get("orchestration"):
        sections.append(orchestration_section(summary["orchestration"]))
        nav.append(("orchestration", "Orchestration"))
    for number, engine in sorted((summary.get("engines") or {}).items()):
        inner, _ = run_body({**engine, "orchestration": None, "engines": None})
        anchor = f"engine{number}"
        sections.append(
            f'<section id="{esc(anchor)}"><h2>Phase {esc(number)} engine output: {esc(engine["run_id"])}</h2>'
            + ("".join(inner) or f"<p>No phase records: {NR}.</p>")
            + "</section>"
        )
        nav.append((anchor, f"Phase {number} engine"))
    return sections, nav


def render_run(summary: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, badge, esc, text_or_nr

    live = summary.get("liveness") or {}
    sections, nav = run_body(summary)
    if not sections:
        sections.append(f"<section><p>No phase records were found in this directory: {NR}.</p></section>")
    sections.append(inventory_section(summary.get("inventory") or []))
    nav.append(("records", "Records"))
    phases = ", ".join(summary.get("phases") or []) or "not recorded"
    sub = (
        f"target {text_or_nr(summary.get('target'))} &middot; phase {esc(phases)} &middot; "
        f"{badge(live.get('state'))} {esc(live.get('detail') or '')}"
    )
    title = f"Experiment dashboard {summary['run_id']}"
    return page(title, esc(summary["run_id"]), sub, nav, "".join(sections), summary["generated"])


def _champion_chain(row: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import digest, num, text_or_nr

    lineage, firesim = row.get("lineage") or {}, row.get("firesim") or {}
    control = text_or_nr(firesim.get("control_in_batch"))
    return _chain(
        [
            _node("phase-0 corpus seal", digest(lineage.get("corpus_seal_digest"))),
            _node(
                "phase-1 frozen", f"{text_or_nr(lineage.get('phase1_run'))}<br>{digest(lineage.get('frozen_commit'))}"
            ),
            _node("phase-2 best", f"{text_or_nr(lineage.get('phase2_run'))}<br>{digest(lineage.get('best_commit'))}"),
            _node(
                "champion",
                f"{text_or_nr(row.get('package_id'))}<br>{num(firesim.get('cycles'))} cycles "
                f"&middot; control in batch {control}",
            ),
        ]
    )


def _origin(row: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import digest, text_or_nr

    origin = row.get("origin")
    return digest(origin.get("commit")) if isinstance(origin, Mapping) else text_or_nr(origin)


def index_section(summary: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import NR, details, digest, esc, mono, table, text_or_nr

    index = summary.get("index")
    if not index:
        return (
            f'<section id="lineage"><h2>Champion lineage</h2><p>INDEX.yaml: {NR} at {mono(summary.get("index_path"))} '
            f'(write it with <span class="mono">merlin-experiment index {esc(summary["target"])}</span>).</p></section>'
        )
    champions = "".join(_champion_chain(row) for row in index.get("champions") or ())
    best = [
        [
            text_or_nr(r.get("run")),
            _origin(r),
            digest(r.get("best_commit")),
            digest(r.get("package_digest")),
            str(len(r.get("measured") or [])),
        ]
        for r in index.get("phase2_best") or ()
    ]
    frozen = [
        [
            text_or_nr(r.get("run")),
            digest(r.get("frozen_commit")),
            digest(r.get("package_digest")),
            text_or_nr(r.get("rounds")),
            text_or_nr(r.get("frozen_at")),
        ]
        for r in index.get("phase1_frozen") or ()
    ]
    releases = [
        [
            text_or_nr(r.get("release")),
            text_or_nr(r.get("state")),
            digest(r.get("review_digest")),
            text_or_nr(r.get("source_run")),
            text_or_nr(r.get("sealed_at")),
        ]
        for r in index.get("phase0_releases") or ()
    ]
    problems = [[text_or_nr(p.get("path")), text_or_nr(p.get("problem"))] for p in index.get("problems") or ()]
    return (
        '<section id="lineage"><h2>Champion lineage</h2>'
        + (champions or f"<p>Exported champions: {NR}.</p>")
        + "<h3>Phase-2 bests</h3>"
        + table(["run", "origin", "best commit", "package", "measured tags"], best)
        + "<h3>Phase-1 frozen compilers</h3>"
        + table(["run", "frozen commit", "package", "rounds", "frozen at"], frozen)
        + "<h3>Phase-0 corpus releases</h3>"
        + table(["release", "state", "review digest", "source run", "sealed at"], releases)
        + (details(f"Index problems ({len(problems)})", table(["path", "problem"], problems)) if problems else "")
        + "</section>"
    )


def _run_row(phase: str, r: Mapping[str, Any]) -> list[str]:
    from merlin.experiments.tracking.pages.markup import NR, badge, digest, esc, mono, num, text_or_nr

    state = esc(r["problem"]) if r.get("problem") else badge(r.get("state"))
    if phase == "1":
        progress = (
            f"{r['latest_passed']}/{r['latest_capsules']} over {r['grades']} grades"
            if r.get("latest_passed") is not None
            else NR
        )
        identity, best = digest(r.get("frozen_commit")), ""
    else:
        progress, identity = "", text_or_nr(r.get("method"))
        best = num(r.get("best_cycles")) if phase == "2" else ""
    return [
        mono(r["run_id"]),
        state,
        text_or_nr(r.get("orchestration_state")),
        progress,
        identity,
        best,
        text_or_nr(r.get("detail")),
    ]


def render_target(summary: Mapping[str, Any]) -> str:
    from merlin.experiments.tracking.pages.markup import details, esc, mono, table, text_or_nr

    sections = [index_section(summary)]
    for phase, rows in sorted((summary.get("phase_runs") or {}).items()):
        extra = (summary.get("truncated") or {}).get(phase) or 0
        title = f"Phase {phase} runs ({len(rows)}{f', {extra} older not shown' if extra else ''})"
        headers = ["run", "state", "orchestration", "latest grade", "frozen commit / method", "best cycles", "detail"]
        sections.append(
            f'<section id="phase{esc(phase)}"><h2>{esc(title)}</h2>'
            + table(headers, [_run_row(phase, r) for r in rows])
            + "</section>"
        )
    orchestrations = summary.get("orchestrations") or {}
    found = [
        [
            text_or_nr(r.get("experiment")),
            text_or_nr(r.get("state")),
            mono(r.get("run_dir")),
            esc(", ".join(f"{n}:{p.get('state')}" for n, p in sorted((r.get("phases") or {}).items()))),
        ]
        for r in orchestrations.get("runs") or ()
    ]
    problems = [
        [text_or_nr(p.get("run_dir")), text_or_nr(p.get("error"))] for p in orchestrations.get("problems") or ()
    ]
    sections.append(
        '<section id="orchestrations"><h2>Orchestrations</h2>'
        + table(["experiment", "state", "run", "phases"], found)
        + (details("Discovery problems", table(["run", "error"], problems)) if problems else "")
        + "</section>"
    )
    sections.append(inventory_section(summary.get("inventory") or []))
    nav = [("lineage", "Lineage")]
    nav += [(f"phase{p}", f"Phase {p}") for p in sorted(summary.get("phase_runs") or {})]
    nav += [("orchestrations", "Orchestrations"), ("records", "Records")]
    target = summary["target"]
    sub = "all runs across phases, from the target's records"
    return page(
        f"Target dashboard {target}", f"Target {esc(target)}", sub, nav, "".join(sections), summary["generated"]
    )


def render(summary: Mapping[str, Any]) -> str:
    return render_target(summary) if summary.get("kind") == "target" else render_run(summary)


__all__ = [
    "render",
    "render_run",
    "render_target",
]
