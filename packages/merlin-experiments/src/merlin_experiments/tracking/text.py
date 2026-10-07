"""The terminal view of a tracking summary: plain text, ANSI colour optional, no curses.

The same summary the HTML page draws, as a few dense lines a person can leave running in a tmux pane.
A value the records do not hold is printed as "not recorded".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import records as R

NR = "not recorded"
_COLOUR = {
    R.LIVE: "32",
    R.FINISHED: "32",
    R.STALLED: "1;31",
    R.STOPPED: "33",
    R.ENDED: "33",
    R.RELAUNCHED: "36",
    R.UNKNOWN: "2",
}
CLEAR = "\x1b[H\x1b[2J"


def _paint(text: str, code: str | None, colour: bool) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if colour and code else text


def _n(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else NR


def _h(value: float | None) -> str:
    if value is None:
        return NR
    return f"{value:.1f} h" if value < 48 else f"{value / 24:.1f} d"


def _when(value: float | None) -> str:
    return R.stamp(value) or NR


def _state(liveness: Mapping[str, Any] | None, colour: bool) -> str:
    state = str((liveness or {}).get("state") or R.UNKNOWN)
    return f"{_paint(state, _COLOUR.get(state), colour)}  {(liveness or {}).get('detail') or ''}".rstrip()


def _reading(c: Mapping[str, Any]) -> str:
    if c.get("cycles"):
        return _n(c["cycles"])
    return _n(c["whole_window"]) + " wrong" if c.get("whole_window") else ""


def _phase2(p2: Mapping[str, Any], now: float, colour: bool) -> list[str]:
    counts, in_run = p2.get("counts") or {}, p2.get("counts_in_run")
    best, bar = p2.get("best") or {}, p2.get("bar") or {}
    store, board, run = p2.get("store") or {}, p2.get("board"), p2.get("run") or {}
    ratio = f" ({best['cycles'] / bar['cycles']:.3f}x bar)" if best.get("cycles") and bar.get("cycles") else ""
    roles = ", ".join(run["prohibited_roles"]) if run.get("prohibited_roles") else NR
    absent = "" if store.get("loaded") or not store.get("root") else " (absent)"
    this_run = f" (this run {in_run.get(R.MEASURED, 0)})" if in_run else ""
    last = _when((p2.get("liveness") or {}).get("last_measured"))
    lines = [
        "PHASE 2  " + _state(p2.get("liveness"), colour),
        f"  method {run.get('method') or NR}  prohibited roles {roles}",
        f"  store {store.get('root') or NR}{absent}",
        f"  candidates {len(p2.get('candidates') or [])}  measured {counts.get(R.MEASURED, 0)}{this_run}"
        f"  pending {counts.get(R.OPEN, 0)}",
        f"  best {_n(best.get('cycles'))}{ratio} {str(best.get('key') or '')[:12]}  bar {_n(bar.get('cycles'))}"
        f"  last measured {last}",
        "  failures  " + "  ".join(f"{name} {counts.get(name, 0)}" for name in R.FAILURE_CLASSES),
    ]
    plateau = p2.get("plateau")
    if plateau:
        rule, improved = plateau.get("rule") or {}, plateau.get("last_improvement") or {}
        anchor = R.epoch(improved.get("epoch")) or R.epoch(improved.get("at"))
        lines.append(
            f"  plateau  rule {rule.get('hours', NR)} h / {rule.get('min_sessions', NR)} sessions;"
            f" last improvement {_when(anchor)} ({_n(improved.get('cycles'))});"
            f" now {_h((now - anchor) / 3600 if anchor else None)}"
        )
    else:
        lines.append(f"  plateau  {NR}")
    if board:
        outage = board.get("open_outage")
        state = f"OPEN since {_when(outage['opened'])}" if outage else "none open"
        lines.append(
            f"  board  queue {board.get('queue', 0)} (oldest {_when(board.get('oldest_waiting'))})"
            f"  running {board.get('running', 0)}  outage {state}; {len(board.get('closed_outages') or [])} closed"
            f"  solo streak {_n(board.get('solo_streak'))}"
        )
    else:
        lines.append(f"  board  {NR}")
    recent = [c for c in p2.get("candidates") or () if c.get("finished")][-6:]
    if recent:
        lines.append("  recent:")
        for c in reversed(recent):
            note = (c.get("reason") or c.get("label") or "")[:60]
            lines.append(
                f"    {_when(c['finished'])}  {c['class']:<22} {_reading(c):>16}  {str(c['key'])[:12]}  {note}"
            )
    return lines


def _phase1(p1: Mapping[str, Any], colour: bool) -> list[str]:
    latest, plateau, freeze = p1.get("latest") or {}, p1.get("plateau"), p1.get("freeze")
    tiers = p1.get("tiers") or []
    lines = ["PHASE 1  " + _state(p1.get("liveness"), colour)]
    if latest:
        reached = {t: sum(1 for c in latest.get("capsules") or () if c["tiers"].get(t) == "pass") for t in tiers}
        lines.append(
            f"  latest grade {latest.get('name')} at {_when(latest.get('at'))}: "
            f"{_n(latest.get('n_passed'))}/{_n(latest.get('n_capsules'))} passed;"
            f" highest tier {latest.get('highest_tier') or NR}"
        )
        if reached:
            lines.append("  tier passes  " + "  ".join(f"{t} {n}" for t, n in reached.items()))
        planes = sorted((latest.get("first_failure_planes") or {}).items(), key=lambda kv: -int(kv[1]))
        if planes:
            lines.append("  first failure planes  " + "  ".join(f"{k} {v}" for k, v in planes))
    else:
        lines.append(f"  grades  {NR}")
    lines.append(f"  grades recorded {len(p1.get('grades') or [])}")
    lines.append(f"  plateau  {(plateau or {}).get('sentence') or NR}")
    if freeze:
        lines.append(f"  freeze  frozen {_when(freeze['at'])} commit {str(freeze.get('frozen_commit') or NR)[:12]}")
    else:
        lines.append(f"  freeze  {NR}")
    for c in (p1.get("failing") or [])[:5]:
        detail = (c.get("failure_detail") or "")[:50]
        lines.append(f"    {c['capsule'][:48]:<48} {c['status']!s:<10} {c.get('failure_plane') or '':<18} {detail}")
    return lines


def render(summary: Mapping[str, Any], *, colour: bool = False) -> str:
    now = float(summary.get("generated") or 0.0)
    phases = ", ".join(summary.get("phases") or []) or NR
    lines = [
        f"{summary.get('run_id')}  target {summary.get('target') or NR}  phase {phases}  ({R.stamp(now)})",
        "STATE  " + _state(summary.get("liveness"), colour),
    ]
    orchestration = summary.get("orchestration")
    if orchestration:
        lines.append(
            f"ORCHESTRATION  {orchestration.get('experiment')}  state {orchestration.get('state')}"
            f"  plan {orchestration.get('plan_binding')}"
        )
    if summary.get("phase2"):
        lines += _phase2(summary["phase2"], now, colour)
    if summary.get("phase1"):
        lines += _phase1(summary["phase1"], colour)
    for number, engine in sorted((summary.get("engines") or {}).items()):
        lines.append(f"-- phase {number} engine output {engine.get('run_id')}")
        lines += render(engine, colour=colour).splitlines()[1:]
    if not summary.get("phase1") and not summary.get("phase2") and not summary.get("engines"):
        lines.append(f"no phase records: {NR}")
    absent = sum(1 for row in summary.get("inventory") or () if row["state"] != "read")
    lines.append(f"records: {len(summary.get('inventory') or [])} consulted, {absent} absent/unreadable")
    return "\n".join(lines) + "\n"


__all__ = ["CLEAR", "render"]
