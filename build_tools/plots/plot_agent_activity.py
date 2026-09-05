#!/usr/bin/env python3
"""Activity share + token rate over wall time, for ANY agentic run.

One figure per run dir, driver-agnostic and target-agnostic: it reads the transcript's per-event
arrival stamps through :mod:`merlin.agent_trace`, so the x-axis is MEASURED rather than laid out
proportionally across a round. The older per-flavour figure states in its own docstring that
"per-message wall stamps don't exist" and interpolates within each round; they exist now, and this
draws them.

It REFUSES rather than approximating. A run whose transcript carries no arrival stamps, or whose
stamps cannot describe one session, gets a printed reason and no PNG -- because a plausible activity
chart drawn on stamps that mean something else is worse than no chart.

    plot_agent_activity.py <run_dir> [--out fig.png] [--bins 160]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))
from merlin.agent_trace import ACTIVITIES, Timeline, timeline  # noqa: E402
from merlin.plotting.merlin_plotstyle import (BG, GOLD, INK, MAUVE, NAVY,  # noqa: E402
                                              SAGE, SLATE, style_ax, use_merlin_style)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

#: One colour per activity, from the house palette, ordered as the stack reads bottom-to-top.
BAND = {"thinking": (BLUE_ := "#5C5C7A"), "reading": SLATE, "writing": GOLD,
        "bash": SAGE, "tool_wait": MAUVE}
#: Below this many usage reports a rate curve is a dot, not a trend.
_MIN_RATE_SAMPLES = 5

LABEL = {"thinking": "thinking", "reading": "reading", "writing": "writing code",
         "bash": "bash / shell", "tool_wait": "tool wait"}


def _rounds(run_dir: Path) -> list[Path]:
    """Every round transcript, in order; falls back to the flat one a single-round run writes."""
    rs = sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    return rs or [p for p in [run_dir / "transcript.jsonl"] if p.exists()]


def _milestones(run_dir: Path) -> list[tuple[float, str]]:
    """(wall_offset_s, label) for each graded improvement -- the same source the older figure uses."""
    log = run_dir / "selfcheck_log.jsonl"
    out: list[tuple[float, str]] = []
    if not log.exists():
        return out
    best = -1
    for line in log.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        off, n, tot = row.get("wall_offset_s"), row.get("n_passed"), row.get("n_capsules")
        if off is None or n is None or n <= best:
            continue
        best = n
        out.append((float(off), f"{n}/{tot}" if tot else str(n)))
    return out


def render(run_dir: Path, out: Path, bins: int) -> int:
    parts = [timeline(p) for p in _rounds(run_dir)]
    # A measured timeline with NO spans is still nothing to draw: an empty stacked area renders as a
    # blank chart that looks like a finished figure. Require actual spans before claiming a run is
    # plottable.
    usable = [t for t in parts if t.measured and t.spans]
    if not usable:
        empty = [t for t in parts if t.measured and not t.spans]
        if empty:
            why = ("every round parsed but produced no completed tool call to place on a timeline "
                   "(no paired tool_use/tool_result with arrival stamps)")
        else:
            why = parts[0].reason if parts else f"no transcript under {run_dir}"
        print(f"REFUSED: {run_dir.name} cannot be charted on a measured time axis.\n  {why}",
              file=sys.stderr)
        return 2

    # Rounds are consecutive sessions: lay them end to end on one axis and mark the seams.
    merged, seams, offset = Timeline(basis="wall_clock"), [], 0.0
    for t in usable:
        for sp in t.spans:
            sp.start_s += offset
            sp.end_s += offset
            merged.spans.append(sp)
        merged.tokens += [(x + offset, i, o) for x, i, o in t.tokens]
        offset += t.wall_s
        seams.append(offset)
    merged.wall_s = offset

    use_merlin_style()
    fig, ax = plt.subplots(figsize=(15, 4.2))
    fig.patch.set_facecolor(BG)
    centres, share = merged.share(bins=bins)
    xs = [c / 60.0 for c in centres]
    ax.stackplot(xs, *[share[a] for a in ACTIVITIES],
                 colors=[BAND[a] for a in ACTIVITIES], alpha=0.55, linewidth=0)
    ax.set_xlim(0, merged.wall_s / 60.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("activity share")
    ax.set_xlabel("Time (min)")
    style_ax(ax, grid="both")

    # WHAT THE FIGURE CANNOT SHOW, IT SAYS. An empty right axis reads as "no tokens were used";
    # the truth is that some drivers report usage once per TURN, which cannot make a rate curve.
    notes: list[str] = []
    rate = ax.twinx()
    # A rate needs several samples to be a curve rather than a dot. Codex reports usage once per
    # TURN, so a two-round run yields two points -- which would draw an invisible line and read as
    # "no tokens used". Say so instead.
    if len(merged.tokens) < _MIN_RATE_SAMPLES:
        notes.append(f"token rate unavailable: only {len(merged.tokens)} usage sample(s) — this "
                     f"driver reports usage once per turn, not per message")
    if len(merged.tokens) >= _MIN_RATE_SAMPLES:
        tm = [x / 60.0 for x, _, _ in merged.tokens]
        for idx, (colour, lab) in ((1, (SLATE, "rate input")), (2, (NAVY, "rate output"))):
            series = [row[idx] for row in merged.tokens]
            deltas = [max(b - a, 0) / max(t2 - t1, 1e-6)
                      for (a, b), (t1, t2) in zip(zip(series, series[1:]), zip(tm, tm[1:]))]
            rate.plot(tm[1:], [max(d, 1e-1) for d in deltas], color=colour, lw=2.0, label=lab)
        rate.set_yscale("log")
    rate.set_ylabel("token rate (tok/min, log)")

    for s in seams[:-1]:
        ax.axvline(s / 60.0, color=INK, lw=0.9, ls=(0, (2, 3)), alpha=0.5)
    ms = _milestones(run_dir)
    if not ms:
        notes.append("no test-pass milestones: this run wrote no selfcheck_log.jsonl")
    for off, lab in ms:
        if off <= merged.wall_s:
            ax.axvline(off / 60.0, color=GOLD, lw=2.0, ls=(0, (4, 3)))
            ax.text(off / 60.0, 1.01, lab, color=GOLD, fontsize=9, ha="center")

    tot = merged.totals()
    ax.set_title(f"{run_dir.name}   {merged.wall_s/60:.0f} min measured   "
                 f"thinking {100*tot['thinking']/max(merged.wall_s,1):.0f}%   "
                 f"tool wait {100*tot['tool_wait']/max(merged.wall_s,1):.0f}%", loc="left")
    dropped = [t for t in parts if not t.measured]
    if dropped:
        notes.append(f"{len(dropped)} of {len(parts)} round(s) omitted — {dropped[0].basis}: "
                     f"{dropped[0].reason.split('.')[0][:90]}")
    if notes:
        ax.text(0.995, 0.02, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=INK, alpha=0.75,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=INK, alpha=0.55, lw=0.6))
    handles = [Patch(facecolor=BAND[a], alpha=0.55, label=LABEL[a]) for a in ACTIVITIES]
    handles += [Line2D([0], [0], color=SLATE, lw=2, label="rate input"),
                Line2D([0], [0], color=NAVY, lw=2, label="rate output"),
                Line2D([0], [0], color=GOLD, lw=2, ls=(0, (4, 3)), label="test-pass milestone")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=True, fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=BG)
    print(f"wrote {out}  ({len(merged.spans)} spans over {merged.wall_s/60:.1f} min, "
          f"{len(usable)}/{len(parts)} round(s) measured)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--bins", type=int, default=160)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run / "fig_agent_activity.png"
    return render(run, out, a.bins)


if __name__ == "__main__":
    raise SystemExit(main())
