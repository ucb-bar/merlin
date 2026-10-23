"""The console snapshot as plain text: ``--once``, pipes, CI logs and hosts without the frontend.

It prints the same sections the Bubble Tea frontend shows, one block per panel, from the same
snapshot. Colour is optional ANSI; record text was cleaned by the sources and is clipped here.
"""

from __future__ import annotations

import time
from typing import Any

from ..terminal_views import clip_ansi, paint
from .discovery import STATES

GLYPHS = {"LIVE": "◉", "STALLED": "◐", "ACTIVE": "◎", "INTERRUPTED": "✖", "FINISHED": "✔", "ENDED": "◌", "STORED": "▣"}
COLOURS = {
    "LIVE": "1;38;5;82",
    "STALLED": "1;38;5;220",
    "ACTIVE": "1;38;5;75",
    "INTERRUPTED": "1;38;5;203",
    "FINISHED": "1;38;5;45",
    "ENDED": "38;5;244",
    "STORED": "38;5;141",
}


def _n(value: Any, digits: int = 0) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "—"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.2f}G"
    if a >= 1e6:
        return f"{value / 1e6:.2f}M"
    if a >= 1e4:
        return f"{value / 1e3:.1f}k"
    return f"{value:,.{digits}f}"


def _dur(seconds: Any) -> str:
    if not isinstance(seconds, int | float) or seconds < 0:
        return "—"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02}s"
    return f"{seconds:.1f}s"


def _ranges(cpus: Any) -> str:
    if not isinstance(cpus, list) or not cpus:
        return "—"
    from .sources.machine import probe_ranges

    return probe_ranges([c for c in cpus if isinstance(c, int)])


def _section(snapshot: dict, name: str) -> dict:
    value = (snapshot.get("sections") or {}).get(name)
    return value if isinstance(value, dict) else {}


def experiments_text(items: list[dict], *, color: bool = False, width: int = 160) -> str:
    lines = [paint(" EXPERIMENTS", "1", color)]
    now = time.time()
    for state in STATES:
        group = [e for e in items if e.get("state") == state]
        if not group:
            continue
        lines.append(paint(f" {GLYPHS[state]} {state} ({len(group)})", COLOURS[state], color))
        for e in group:
            age = _dur(now - e["last_record_at"]) if e.get("last_record_at") else "—"
            lines.append(
                f"   {e.get('run_id', '?'):<48} {e.get('task_name') or '-':<40} {age:>8}  {e.get('state_basis', '')}"
            )
    return "\n".join(clip_ansi(line, width) for line in lines)


def render(snapshot: dict[str, Any], *, color: bool = False, width: int = 160) -> str:
    e = snapshot.get("experiment") or {}
    state = e.get("state", "?")
    out = [
        paint(f" {GLYPHS.get(state, '?')} {state}  {e.get('run_id')}", COLOURS.get(state, "1"), color)
        + f"  task {e.get('task_name') or '-'}  {'OPERATOR' if snapshot.get('operator_private') else 'PUBLIC'}",
        f" basis: {e.get('state_basis', '')}",
        f" run: {e.get('run_dir')}",
        "",
    ]

    p = _section(snapshot, "progress")
    out.append(
        paint(" PROGRESS", "1", color) + f"  latest grade {_n(p.get('n_completed'))}/{_n(p.get('n_selected'))} checked"
        f"  passed {_n(p.get('n_passed'))}  state {p.get('state') or '—'}"
        + ("  (retained snapshot)" if p.get("snapshot_current") is False else "")
    )
    for c in p.get("cohorts") or []:
        out.append(
            f"   {c.get('name', '?'):<12} all {_n(c.get('all')):>4}  L2 {_n(c.get('L2')):>4}  L3 {_n(c.get('L3')):>4}"
            f"  refusal {_n(c.get('refusal')):>3}  of {_n(c.get('total'))}"
        )
    if p.get("completion"):
        out.append(f"   completed: {p['completion'].get('reason')} after {_dur(p['completion'].get('elapsed_s'))}")

    stages = _section(snapshot, "stages").get("stages") or []
    if stages:
        out.append(paint(" STAGES", "1", color) + "  " + " → ".join(f"{s.get('id')}:{s.get('state')}" for s in stages))

    out.append(paint(" BOTTLENECKS", "1", color))
    for d in (snapshot.get("diagnostics") or [])[:8]:
        times = f" ×{d['occurrences']}" if d.get("occurrences", 1) > 1 else ""
        out.append(f"   [{d.get('severity')}] {d.get('title')}{times} — {d.get('detail')}")
    if not snapshot.get("diagnostics"):
        out.append("   no findings from recorded numbers")

    tl = _section(snapshot, "timeline")
    lead, verdict, window = tl.get("lead") or {}, tl.get("verdict") or {}, (tl.get("window") or {}).get("seconds")
    if lead and isinstance(window, int | float) and window > 0:
        shares = "  ".join(
            f"{k} {100 * (lead.get(f'{k}_s') or 0) / window:.0f}%" for k in ("setup", "think", "tool", "wait", "idle")
        )
        para = tl.get("parallelism") or {}
        out.append(
            paint(" TIMELINE", "1", color) + f"  lead {shares}  over {_dur(window)}"
            f"  agents {_n(para.get('agents_avg'), 1)} avg / {_n(para.get('agents_peak'))} peak"
        )
    recorded = tl.get("recorded") or {}
    if recorded:
        total = (recorded.get("think_s") or 0) + (recorded.get("tool_wait_s") or 0)
        if total:
            out.append(
                f"   recorded split: think {100 * recorded['think_s'] / total:.0f}%  tool+wait"
                f" {100 * recorded['tool_wait_s'] / total:.0f}%"
            )
    if verdict:
        out.append(
            f"   bound: {str(verdict.get('bound')).replace('_', ' ')} {_n(verdict.get('share_pct'))}%"
            f" — {verdict.get('explanation')}"
        )

    agents = _section(snapshot, "agents").get("agents") or []
    out.append(paint(" AGENTS", "1", color) + f"  {len(agents)} recorded")
    for a in agents[:20]:
        tokens = (a.get("tokens") or {}).get("total")
        name = a.get("name") or ("lead" if a.get("role") == "lead" else a.get("id"))
        indent = "   " + "  " * int(a.get("depth") or 0)
        out.append(
            f"{indent}{name} [{a.get('state')}] {a.get('path') or ''}"
            f"  tokens {_n(tokens)}  active {_dur(a.get('active_s'))}"
        )

    t = _section(snapshot, "tokens")
    total = t.get("total") or {}
    out.append(
        paint(" TOKENS", "1", color) + f"  fresh {_n(total.get('fresh_input'))}  cache {_n(total.get('cache_read'))}"
        f"  out {_n(total.get('output'))}  reasoning {_n(total.get('reasoning'))}  total {_n(total.get('total'))}"
    )

    r = _section(snapshot, "resources")
    m = _section(snapshot, "machine")
    out.append(
        paint(" CORES", "1", color)
        + f"  machine busy {_n(m.get('busy_cpus'), 1)}/{_n(m.get('total_cpus'))}"
        + f"  experiment {_ranges(m.get('experiment_cpus'))}  console {_ranges(m.get('console_cpus'))}"
        + f"  run telemetry {r.get('mode') or '—'}"
    )
    for alloc in m.get("allocations") or []:
        if isinstance(alloc, dict):
            procs = f"  {alloc['processes']} processes" if isinstance(alloc.get("processes"), int) else ""
            out.append(f"   {alloc.get('kind', '?'):<12} {alloc.get('name')}  cpus {alloc.get('cpus')}{procs}")

    perf = _section(snapshot, "perf")
    rows = sorted(
        (x for x in perf.get("rows") or [] if isinstance(x.get("efficiency_pct"), int | float)),
        key=lambda x: x["efficiency_pct"],
    )
    out.append(paint(" PERFORMANCE", "1", color) + f"  {len(perf.get('rows') or [])} rows")
    for row in rows[:8]:
        out.append(
            f"   {str(row.get('capsule', '?'))[:40]:<40} model {_n(row.get('spike_model_cycles')):>9}"
            f"  gSIM {_n(row.get('gsim_cycles')):>9}"
            f"  floor {_n(row.get('floor_cycles')):>8}  eff {_n(row.get('efficiency_pct'), 2)}%"
        )

    caps = _section(snapshot, "capsules")
    counts = (caps.get("counts") or {}).get("status") or {}
    out.append(paint(" CAPSULES", "1", color) + "  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    passes = _section(snapshot, "passes")
    pc = passes.get("counts") or {}
    out.append(paint(" PASSES", "1", color) + "  " + "  ".join(f"{k} {_n(v)}" for k, v in sorted(pc.items())))
    if passes.get("pipeline"):
        out.append("   pipeline: " + " → ".join(passes["pipeline"]))

    events = _section(snapshot, "events").get("events") or []
    out.append(paint(" EVENTS", "1", color))
    for ev in events[:10]:
        stamp = time.strftime("%H:%M:%S", time.gmtime(ev["t"])) if isinstance(ev.get("t"), int | float) else "--:--:--"
        out.append(f"   {stamp} {ev.get('kind', ''):<10} {ev.get('agent') or ''} {ev.get('text', '')}")
    errors = [
        f"{k}: {v['error']}"
        for k, v in (snapshot.get("sections") or {}).items()
        if isinstance(v, dict) and v.get("error")
    ]
    if errors:
        out.append(paint(" READER ERRORS", "1;38;5;203", color))
        out += [f"   {x}" for x in errors]
    return "\n".join(clip_ansi(line, width) for line in out)


__all__ = ["experiments_text", "render"]
