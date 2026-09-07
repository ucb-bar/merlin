#!/usr/bin/env python3
"""One phase-1 run, in full — the anatomy of a capsule-bench experiment.

    anatomy_figure.py --anatomy anatomy_<run>.json [--out DIR]

Three stacked views on one shared clock, because the question this answers is how they line up:

1. what the agent had passing, and which capsule turned green when;
2. where its wall clock went, split into DEVELOPMENT and FEEDBACK, with every call marked on top;
3. what it consumed -- the three token rates, which move for different reasons, against spend.

The view ENDS where the score stopped moving, because on this run what came after was not the agent
failing to improve -- it was the agent working on capsules that could not move. Two of the four
unresolved ones PASS their RTL tier and fail on a lane contract or a routing rule; a third is marked
incomplete because a required lane is never measured on this path. Those are the instrument's ceiling
and no amount of further work reaches them, so the tail is summarised in a line rather than drawn as
four hours of flat.

``--full-span`` draws the whole run instead, for when the plateau itself is the subject.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

from merlin.agentreport.anatomy import CATEGORIES, PLANE_PLAIN            # noqa: E402
from merlin.common.paths import artifacts_dir                            # noqa: E402
from merlin.plotting.merlin_plotstyle import (BLUE, GOLD, INK, MAUVE, NAVY,   # noqa: E402
                                              SAGE, SLATE, suptitle, title,
                                              use_merlin_style)
from merlin.plotting.merlin_plotstyle import style_ax as _house_style_ax  # noqa: E402

PAGE_BG = "#FFFFFF"
#: Development shades run cool, feedback shades run warm: the split is the figure's whole argument,
#: so it is carried by hue and not only by the legend.
CAT_COLOUR = {"author": NAVY, "inspect": SLATE, "merlin_tool": BLUE, "build": "#6E7F8D",
              "shell": "#AAB4BF", "selfcheck": MAUVE, "oracle": GOLD}


def style_ax(ax, *, grid="y"):
    _house_style_ax(ax, grid=grid)
    ax.set_facecolor(PAGE_BG)


def _smooth(y, win=9):
    """Hanning-smoothed series. The band is an occupancy SHARE, and at one-minute bins a single long
    call makes a square wall that reads as structure it does not have; a short window keeps the
    shape and loses the aliasing."""
    if win < 3 or len(y) < win:
        return y
    w = np.hanning(win)
    w = w / w.sum()
    pad = win // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(padded, w, mode="same")[pad:pad + len(y)]


def _bins(calls, wall_s, n=160):
    """Occupied seconds per category per time bin, from measured span overlap."""
    width = max(wall_s / n, 1e-9)
    acc = {k: np.zeros(n) for k in CATEGORIES}
    for c in calls:
        if c["duration_s"] <= 0:
            continue
        lo, hi = c["t_s"], c["t_s"] + c["duration_s"]
        first, last = max(int(lo // width), 0), min(int(hi // width), n - 1)
        for b in range(first, last + 1):
            b_lo, b_hi = b * width, (b + 1) * width
            acc[c["category"]][b] += max(min(hi, b_hi) - max(lo, b_lo), 0.0)
    centres = np.array([(b + 0.5) * width for b in range(n)])
    return centres, acc, width


def render(a: dict, out: Path, *, until_best: bool = True) -> Path:
    wall_min = a["wall_s"] / 60.0
    verdicts_all = a["verdicts"]
    best_at = None
    if verdicts_all:
        best = max(v["n_passed"] for v in verdicts_all)
        best_at = min(v["t_s"] for v in verdicts_all if v["n_passed"] == best) / 60.0
    span = (best_at * 1.08 if (until_best and best_at) else max(wall_min, 1.0) * 1.02)

    # Clip every series to the span rather than relying on the axis limit: an annotation placed past
    # the limit is not clipped by default, and a tight bounding box then grows to include it.
    calls_all = a["calls"]
    calls = [c for c in calls_all if c["t_s"] <= span * 60]
    a = dict(a)
    a["verdicts"] = [v for v in verdicts_all if v["t_s"] <= span * 60]
    a["token_curve"] = [p for p in (a.get("token_curve") or []) if p["t_s"] <= span * 60]
    a["cost_curve"] = [p for p in (a.get("cost_curve") or []) if p["t_s"] <= span * 60]
    use_merlin_style()
    plt.rcParams.update({"axes.facecolor": PAGE_BG, "figure.facecolor": PAGE_BG,
                         "savefig.facecolor": PAGE_BG})

    fig = plt.figure(figsize=(17.0, 10.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 2.0], hspace=0.30,
                          left=0.100, right=0.742, top=0.812, bottom=0.170)
    ax_band = fig.add_subplot(gs[0])
    ax_sim = fig.add_subplot(gs[1], sharex=ax_band)
    ax_caps = ax_band          # progress is drawn ON the main panel, not beside it
    ax_tok = ax_band

    # ---------------------------------------------------------------- 1. capsules
    verdicts = a["verdicts"]
    v_max = max((v["t_s"] for v in verdicts), default=0) / 60.0
    order = sorted({c for v in verdicts for c in v["per_capsule"]}) if verdicts else []

    # ---------------------------------------------------------------- 2. activity band
    centres, acc, width = _bins(calls, a["wall_s"])
    xs = centres / 60.0
    dev = [k for k in CATEGORIES if not CATEGORIES[k][1]]
    fb = [k for k in CATEGORIES if CATEGORIES[k][1]]
    stack = [_smooth(np.minimum(acc[k] / width, 1.0)) for k in dev + fb]
    whole = Counter()
    whole_s = Counter()
    for c in calls_all:
        whole[c["category"]] += 1
        whole_s[c["category"]] += c["duration_s"]
    def _lab(k):
        sec = whole_s[k]
        amount = f"{sec / 3600:.1f} h" if sec >= 3600 else (f"{sec:.0f} s" if sec >= 1 else "<1 s")
        name = CATEGORIES[k][0].split(":")[0].split("(")[0].strip()
        return f"{name} · {whole[k]}× · {amount}"
    ax_band.stackplot(xs, *stack, colors=[CAT_COLOUR[k] for k in dev + fb],
                      labels=[_lab(k) for k in dev + fb], alpha=0.95, lw=0, zorder=2)
    occupied = np.clip(sum(stack), 0, 1)
    # The remainder is the agent thinking. It is most of the panel, so it has to recede: a light
    # wash with a faint hatch reads as ground, while a dark hatch competes with the data on top.
    ax_band.fill_between(xs, occupied, 1.0, color="#F4F1EC", lw=0, zorder=1)
    ax_band.fill_between(xs, occupied, 1.0, color="none", edgecolor="#C9C2B8", lw=0.0,
                         hatch="///", alpha=0.55, zorder=1)
    ax_band.set_ylabel("share of each minute")
    style_ax(ax_band, grid="")
    dev_s = sum(c["duration_s"] for c in calls_all if not CATEGORIES[c["category"]][1])
    fb_s = sum(c["duration_s"] for c in calls_all if CATEGORIES[c["category"]][1])
    title(ax_band, f"What the agent did, and when the score moved   ·   waiting for feedback "
                   f"{fb_s / 3600:.1f} h vs doing the work {dev_s / 3600:.1f} h   ·   hatched = "
                   f"thinking   ·   dots = every tool call, sized by duration", fs=12.5, pad=34)
    ax_band.legend(loc="upper left", bbox_to_anchor=(1.135, 1.04), fontsize=8.0, frameon=True,
                   title="what the agent did (whole run)", title_fontsize=8.4,
                   borderaxespad=0.0, labelspacing=0.42)

    # Every call, marked in a strip INSIDE the same panel: the band is their aggregate, so stacking
    # them in their own panel spent a whole row restating one curve. Inside, not above, so nothing
    # escapes the axes when the view is truncated.
    lanes = [k for k in CATEGORIES if any(c["category"] == k for c in calls)]
    strip_lo, strip_hi = 1.06, 1.44
    step = (strip_hi - strip_lo) / max(len(lanes) - 1, 1)
    for i, k in enumerate(lanes):
        pts = [c for c in calls if c["category"] == k]
        y = strip_lo + i * step
        ax_band.scatter([c["t_s"] / 60.0 for c in pts], np.full(len(pts), y),
                        s=[4 + min(c["duration_s"], 400) * 0.40 for c in pts],
                        color=CAT_COLOUR[k], edgecolor=INK, lw=0.3, alpha=0.85, zorder=5)
        ax_band.annotate(CATEGORIES[k][0].split(":")[0].split("(")[0].strip(),
                         xy=(0, y), xycoords=("axes fraction", "data"),
                         xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                         fontsize=7.2, color=CAT_COLOUR[k], annotation_clip=False)
    ax_band.set_ylim(0, strip_hi + step * 0.6)
    ax_band.set_yticks([0, 0.5, 1.0])
    ax_band.set_yticklabels(["0", "50%", "100%"])
    ax_band.axhline(1.02, color=INK, lw=0.6, alpha=0.30)

    # Progress rides the same panel on its own axis, so the reader does not have to carry a second
    # picture: the question is what the agent was doing WHEN the score moved.
    if verdicts:
        total = max(v["n_capsules"] for v in verdicts)
        ax_prog = ax_band.twinx()
        xs = [v["t_s"] / 60.0 for v in verdicts]
        ys = [v["n_passed"] for v in verdicts]
        import matplotlib.patheffects as _pe
        ax_prog.step(xs + [span], ys + [ys[-1]], where="post", color=INK, lw=2.4, zorder=7,
                     path_effects=[_pe.withStroke(linewidth=4.6, foreground="white")])
        prev = None
        for v in verdicts:
            if v["n_passed"] == prev:
                continue
            prev = v["n_passed"]
            x = v["t_s"] / 60.0
            ax_prog.plot([x], [v["n_passed"]], "o", color=INK, ms=6, zorder=8,
                         markeredgecolor="white", markeredgewidth=1.2)
            usd = next((c["usd"] for c in reversed(a.get("cost_curve") or [])
                        if c["t_s"] <= v["t_s"]), None)
            chip = f"{v['n_passed']}/{total}" + (f"  ${usd:,.0f}" if usd else "")
            near = x > 0.86 * span
            ax_prog.annotate(chip, (x, v["n_passed"]), textcoords="offset points",
                             xytext=(-9 if near else 7, 10), ha="right" if near else "left",
                             va="bottom",
                             fontsize=9.5, color=INK, fontweight="bold", zorder=8,
                             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=INK, lw=0.7,
                                       alpha=0.92))
        ax_prog.set_ylim(0, total * 1.62)
        ax_prog.set_yticks([])
        ax_prog.set_ylabel("")
        ax_prog.spines["top"].set_visible(False)
        for v in verdicts:
            ax_band.axvline(v["t_s"] / 60.0, color=INK, lw=0.7, ls=(0, (2, 4)), alpha=0.28, zorder=1)

    # ---------------------------------------------------------------- 4. token rate
    import matplotlib.patheffects as _pe
    ax_rate = ax_band.twinx()
    ax_rate.spines["right"].set_position(("axes", 1.062))
    ax_rate.spines["top"].set_visible(False)
    curve = a.get("token_curve") or []
    if len(curve) >= 3:
        for key, colour, label in (("input", SLATE, "new input (fresh prompt)"),
                                   ("output", NAVY, "output (what the agent writes)"),
                                   ("cache_read", GOLD, "re-read context (cached)")):
            ts, vs = [], []
            for i in range(1, len(curve)):
                lo = max(0, i - 2)
                dt = (curve[i]["t_s"] - curve[lo]["t_s"]) / 60.0
                if dt <= 0:
                    continue
                ts.append(curve[i]["t_s"] / 60.0)
                vs.append(max(curve[i][key] - curve[lo][key], 0) / dt)
            if ts:
                ax_rate.plot(ts, [max(v, 1e-1) for v in vs], color=colour, lw=1.7, label=label,
                             marker="o", ms=2.6, alpha=1.0, zorder=6,
                             path_effects=[_pe.withStroke(linewidth=3.4, foreground="white")])
        ax_rate.set_yscale("log")
        ax_rate.legend(loc="upper left", bbox_to_anchor=(1.135, 0.36), fontsize=8.0,
                       frameon=True, title="token rate", title_fontsize=8.4,
                       borderaxespad=0.0, labelspacing=0.42)
    ax_rate.set_ylabel("tokens / min (log)", labelpad=2)


    cost = a.get("cost_curve") or []

    ax_band.set_xlim(0, span)
    # The moment the score stopped moving, marked on every panel — it is the line the other two
    # panels have to be read against.
    if best_at is not None and best_at <= span:
        for ax in (ax_band, ax_sim):
            ax.axvline(best_at, color=GOLD, lw=2.0, ls=(0, (5, 3)), zorder=6)

    tail_calls = [c for c in calls_all if best_at is not None and c["t_s"] > best_at * 60]
    tail_s = sum(c["duration_s"] for c in tail_calls)
    total_s = sum(c["duration_s"] for c in calls_all) or 1.0
    # ---------------------------------------------------------------- simulator panel
    # Grading is a BATCH: one grade launches every capsule across the worker pool at once, so the
    # simulations arrive in bursts by construction. Six hundred individual bars render that as
    # slivers; the honest and legible form is how many were in flight at each moment.
    sim = [e for e in (a.get("sim_events") or []) if e["end_s"] / 60.0 <= span]
    TIER_COLOUR = {"L3": MAUVE, "L2": SLATE, "L4": BLUE}
    NAME = {"L2": "fast functional (spike)", "L3": "cycle-accurate RTL (GSIM)", "L4": "L4"}

    def _inflight(events, grid):
        """Simulations in flight at each point on ``grid`` (seconds)."""
        out = np.zeros(len(grid))
        for e in events:
            out += (grid >= e["start_s"]) & (grid < e["end_s"])
        return out

    if sim:
        grid = np.linspace(0, span * 60, 1400)
        tiers = [t for t in ("L2", "L3", "L4") if any(e["tier"] == t for e in sim)]
        series = [_inflight([e for e in sim if e["tier"] == t], grid) for t in tiers]
        ax_sim.stackplot(grid / 60.0, *series, colors=[TIER_COLOUR[t] for t in tiers],
                         labels=[NAME[t] for t in tiers], alpha=0.9, lw=0)
        peak = int(max(sum(series)) if len(series) else 0)
        ax_sim.set_ylim(0, max(peak * 1.35, 2))
        ax_sim.set_ylabel("simulations\nin flight")
        secs = Counter()
        for e in sim:
            secs[e["tier"]] += e["sim_active_s"]
        n_by = Counter(e["tier"] for e in sim)

        def _amount(sec):
            return f"{sec / 60:.0f} min" if sec >= 60 else f"{sec:.0f} s"
        costs = a.get("grade_costs") or []
        typical = [g for g in costs if g["sim_active_s"] and
                   g["build_s"] / g["sim_active_s"] > 1]
        head_bits = ["  ·  ".join(f"{n_by[t]} × {NAME[t]}, {_amount(secs[t])}" for t in tiers)]
        title(ax_sim, "Hardware simulation — " + head_bits[0], fs=11.5, pad=26)
        if typical:
            g = max(typical, key=lambda g: g["build_s"])
            # Sits between the title and the axes, where nothing else competes for the row.
            ax_sim.text(0.0, 1.015,
                        f"…but a grade is mostly REBUILDING: {g['build_s']:.0f} s compiling "
                        f"{g['n_capsules']} tests to run {g['sim_active_s']:.0f} s of simulation — "
                        f"{g['build_s'] / g['sim_active_s']:.0f}× more build than simulate",
                        transform=ax_sim.transAxes, ha="left", va="bottom", fontsize=9.6,
                        color=MAUVE, fontweight="bold")
        ax_sim.legend(loc="upper right", fontsize=8.2, labelspacing=0.35, framealpha=0.95,
                      borderaxespad=0.4)

        # Zoom the busiest grade: at run scale a ten-minute burst is a smudge, and the overlap the
        # panel exists to show lives inside it.
        best_grade, best_span = None, 0.0
        for g in {e["grade"] for e in sim}:
            sel = [e for e in sim if e["grade"] == g]
            work = sum(e["end_s"] - e["start_s"] for e in sel)
            if work > best_span:
                best_grade, best_span = g, work
        if best_grade:
            sel = [e for e in sim if e["grade"] == best_grade]
            lo = min(e["start_s"] for e in sel)
            hi = max(e["end_s"] for e in sel)
            # Zoom the window that actually carries the work, not the grade's full extent: a long
            # tail of stragglers flattens the burst the inset exists to show.
            dense = _inflight(sel, np.linspace(lo, hi, 1200))
            xs_d = np.linspace(lo, hi, 1200)
            busy = xs_d[dense >= max(dense.max() * 0.12, 1)]
            if len(busy):
                lo, hi = float(busy[0]), float(busy[-1])
            pad = (hi - lo) * 0.05
            ins = ax_sim.inset_axes([0.30, 0.14, 0.46, 0.74])
            zgrid = np.linspace(lo - pad, hi + pad, 800)
            zser = [_inflight([e for e in sel if e["tier"] == t], zgrid) for t in tiers]
            ins.stackplot(zgrid / 60.0, *zser, colors=[TIER_COLOUR[t] for t in tiers],
                          alpha=0.9, lw=0)
            zpeak = int(max(sum(zser)))
            ins.set_ylim(0, max(zpeak * 1.2, 2))
            ins.set_xlim((lo - pad) / 60.0, (hi + pad) / 60.0)
            ins.tick_params(labelsize=8.0, length=0)
            ins.set_facecolor("white")
            for sp in ("top", "right"):
                ins.spines[sp].set_visible(False)
            work = sum(e["end_s"] - e["start_s"] for e in sel)
            inside = [e for e in sel if e["start_s"] < hi and e["end_s"] > lo]
            ins.set_title(f"busiest grade, zoomed — {len(inside)} evaluations in "
                          f"{(hi - lo) / 60:.1f} min · "
                          f"{sum(e['end_s'] - e['start_s'] for e in inside) / max(hi - lo, 1):.1f}× "
                          f"overlapped · peak {zpeak} at once", fontsize=9.0, color=INK, pad=4)
            ins.set_xlabel("min", fontsize=7.6, labelpad=1)
            ax_sim.indicate_inset_zoom(ins, edgecolor=INK, alpha=0.45, lw=0.9)
    else:
        ax_sim.text(0.5, 0.5, "no per-capsule simulator timing recorded for this run",
                    transform=ax_sim.transAxes, ha="center", va="center", fontsize=9, color=MAUVE)
    style_ax(ax_sim, grid="x")
    ax_sim.set_xlabel("Time (min)")

    counts = Counter(c["category"] for c in calls_all)
    head = (f"{a['target']} · {a['arm']} · {a['run_id']}   —   {a['model']}   ·   "
            f"whole run {wall_min / 60:.1f} h, {len(calls_all)} tool calls, "
            f"{counts['selfcheck']} self-checks, {len(verdicts_all)} grades   ·   "
            f"shown: the first {span:.0f} min, to where the score stopped moving")
    fig.text(0.100, 0.945, head, fontsize=10.2, color=INK)
    fig.text(0.100, 0.921,
             "An agent is given a chip it has never seen and must write a compiler for it. Each test "
             "is one operation whose answer it never sees; it checks itself with a redacted "
             "self-check and can queue hardware simulations.",
             fontsize=9.2, color=INK, alpha=0.85, va="top")
    suptitle(fig, "Anatomy of a phase-1 run: an agent writing a compiler for unseen hardware",
             y=0.985, fs=17)

    blocked = a.get("blocked") or []
    if tail_calls:
        deep = [b for b in blocked if b.get("deepest_tier_passed")]
        # Lead with WHY the score stopped, not with how long the agent kept going. "kept working
        # without improving" reads as a failure of persistence; these capsules could not move.
        if blocked:
            planes = "; ".join(sorted({PLANE_PLAIN.get(b["plane"], b["plane"] or b["category"]
                                                        or "unknown") for b in blocked}))
            summary = (f"The last {len(blocked)} tests could not move: they turn on {planes}"
                       + (f" — {len(deep)} already pass on simulated hardware" if deep else "")
                       + f". The run spent {(wall_min - (best_at or 0)) / 60:.1f} h more on them.")
        else:
            summary = (f"the run continued {(wall_min - (best_at or 0)) / 60:.1f} h and "
                       f"{len(tail_calls)} more calls past this line without moving the score")
        blocked_line = summary
    blocked_line = locals().get("blocked_line", "")
    note = (a.get("notes") or [""])[0]
    if blocked_line:
        fig.text(0.100, 0.088, "\u25b6  " + "\n     ".join(textwrap.wrap(blocked_line, width=168)),
                 ha="left", va="top", fontsize=8.8, color=GOLD, fontweight="bold")
    fig.text(0.012, 0.012,
             "Progress and simulation come from the grader, everything else from the agent's "
             "transcript; all anchored on the run's start. Grading is batched across a worker "
             "pool, so simulations arrive in bursts, and a verdict lands some minutes after the "
             "grading behind it. " + (note + " " if note else "")
             + f"Edits finish inside one arrival stamp, so authoring reads as {whole['author']} "
               f"events rather than a duration.",
             fontsize=8.2, color=INK, alpha=0.8, wrap=True)

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig12_phase1_anatomy.{ext}", dpi=190, bbox_inches="tight",
                    facecolor=PAGE_BG)
    plt.close(fig)
    return out / "fig12_phase1_anatomy.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anatomy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=artifacts_dir() / "agentic-report" / "figures")
    ap.add_argument("--full-span", action="store_true",
                    help="draw the whole run rather than stopping where the score did")
    a = ap.parse_args(argv)
    path = render(json.loads(a.anatomy.read_text()), a.out, until_best=not a.full_span)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
