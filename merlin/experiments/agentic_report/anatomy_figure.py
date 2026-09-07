#!/usr/bin/env python3
"""One phase-1 run, on one wide clock — the anatomy of a capsule-bench experiment.

    anatomy_figure.py --anatomy anatomy_<run>.json [--out DIR]

A single panel: what the agent was DOING each minute (a stacked share that fills the panel, thinking
included as a band of its own), what it CONSUMED (token rate, right axis, log), when the score MOVED
(gold dashes with a chip carrying score, spend and tokens at that moment), and — on a rail in the
margin above the panel — the hardware simulations each grade launched, which arrive in bursts
because a grade is a batch.

Only FOUR bands, deliberately. Seven categories at this height were seven indistinguishable
slivers; the split that carries the argument is work / shell / waiting / thinking, and the per-call
counts ride in the legend labels.

The view ENDS where the score stopped moving: what came after was not the agent failing to improve,
it was the agent working on capsules that could not move. ``--full-span`` draws the whole run.

What the figure cannot show, it SAYS: a driver that reports usage once per run cannot make a rate
curve, and an empty log axis reads as "no tokens were used".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.patheffects as pe                                      # noqa: E402
import numpy as np                                                       # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

from merlin.agentreport.anatomy import CATEGORIES                        # noqa: E402
from merlin.common.paths import artifacts_dir                            # noqa: E402
from merlin.plotting.merlin_plotstyle import (BLUE, GOLD, INK, MAUVE, NAVY,   # noqa: E402
                                              SAGE, SERIF, SLATE, use_merlin_style)
from merlin.plotting.merlin_plotstyle import style_ax as _house_style_ax  # noqa: E402

PAGE_BG = "#FFFFFF"

#: The stack, bottom to top: (label, categories, colour). Four bands, four clearly different hues,
#: so adjacent bands separate at alpha 0.55. GOLD (milestones), SAGE and NAVY (the two rate lines)
#: are reserved and never used for a band.
BANDS: list[tuple[str, tuple[str, ...], str]] = [
    ("writing & reading code", ("author", "inspect", "merlin_tool"), BLUE),
    ("compiling / shell", ("build", "shell"), "#B7A99A"),
    ("waiting for feedback", ("selfcheck", "oracle"), MAUVE),
]
#: The remainder of each minute: the agent thinking. A band like any other — as a hatch wash it read
#: as "no data" and swamped the panel.
THINK_LABEL, THINK_COLOUR = "thinking", SLATE

#: The simulation rail, in the margin above the panel.
TIER_COLOUR = {"L2": "#4E7A8C", "L3": "#B2705A", "L4": BLUE}

#: Margin geometry, in AXES-FRACTION units above the panel (the y axis itself stops at 1.00, so the
#: chips and the rail never eat into the data area).
CHIP_LO, CHIP_HI = 1.02, 1.13
RAIL_LO, RAIL_HI = 1.18, 1.32


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


def _at(curve, t_s, key):
    """The last value of ``key`` reported at or before ``t_s`` — cumulative curves, sampled."""
    return next((p[key] for p in reversed(curve or []) if p["t_s"] <= t_s), None)


def _tokens_at(curve, t_s):
    p = next((p for p in reversed(curve or []) if p["t_s"] <= t_s), None)
    if not p:
        return None
    return p["input"] + p["output"] + p.get("cache_read", 0) + p.get("cache_creation", 0)


def _inflight(events, grid):
    """Simulations in flight at each point on ``grid`` (seconds)."""
    out = np.zeros(len(grid))
    for e in events:
        out += (grid >= e["start_s"]) & (grid < e["end_s"])
    return out


def _cost(a):
    """How this run's dollars must be SPOKEN, from ``cost_kind`` — never from the presence of a
    number. A subscription seat is not billed per token, so its dollar figure is what the same
    traffic would have cost metered; printing it as spend is the confusion ``cost_kind`` exists to
    prevent. Returns (total_label, per_moment_prefix): the word "notional" is carried once, by the
    summary chip, and the tilde marks every other figure as an estimate.

    ``cost_usd`` wins whenever it is present: if a record ever disagrees with itself, the metered
    number is the one that was actually billed."""
    kind = (a.get("cost_kind") or "").strip().lower()
    metered, notional = a.get("cost_usd"), a.get("notional_usd")
    if metered is not None:
        return f"${metered:,.0f}", "$"
    if kind == "subscription_notional" and notional is not None:
        return f"~${notional:,.0f} notional", "~$"
    if notional is not None or kind == "subscription_notional":
        return (f"~${notional:,.0f} notional" if notional is not None
                else "cost not reported"), "~$"
    return "", "$"


def _dur(sec):
    return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min"


def _chip_picks(improved, span, min_gap=0.075):
    """Which improvements get a CHIP (all of them get a dash). A run that improves sixteen times
    cannot carry sixteen boxes on one row; keep the first, the best, and whatever else fits."""
    gap = span * min_gap
    picks: list[dict] = []
    for v in improved:
        if picks and v["t_s"] / 60.0 - picks[-1]["t_s"] / 60.0 < gap:
            continue
        picks.append(v)
    if improved and picks and picks[-1] is not improved[-1]:
        last = improved[-1]
        while picks and last["t_s"] / 60.0 - picks[-1]["t_s"] / 60.0 < gap:
            picks.pop()
        picks.append(last)
    return picks


def render(a: dict, out: Path, *, until_best: bool = True) -> Path:
    wall_min = a["wall_s"] / 60.0
    verdicts_all = a["verdicts"]
    best = max((v["n_passed"] for v in verdicts_all), default=0)
    best_at = (min(v["t_s"] for v in verdicts_all if v["n_passed"] == best) / 60.0
               if verdicts_all else None)
    span = min(best_at * 1.06 if (until_best and best_at) else 1e18, max(wall_min, 1.0) * 1.02)

    # Clip every series to the span rather than relying on the axis limit: an annotation placed past
    # the limit is not clipped by default, and a tight bounding box then grows to include it.
    calls_all = a["calls"]
    calls = [c for c in calls_all if c["t_s"] <= span * 60]
    verdicts = [v for v in verdicts_all if v["t_s"] <= span * 60]
    token_curve = [p for p in (a.get("token_curve") or []) if p["t_s"] <= span * 60]
    cost_curve = [p for p in (a.get("cost_curve") or []) if p["t_s"] <= span * 60]
    sim = [e for e in (a.get("sim_events") or []) if e["start_s"] / 60.0 <= span]
    cost_label, cost_prefix = _cost(a)

    use_merlin_style()
    plt.rcParams.update({"axes.facecolor": PAGE_BG, "figure.facecolor": PAGE_BG,
                         "savefig.facecolor": PAGE_BG})

    fig, ax = plt.subplots(figsize=(15.0, 5.0))
    fig.subplots_adjust(left=0.052, right=0.938, top=0.700, bottom=0.175)
    marg = ax.get_xaxis_transform()      # x in data units, y in axes fractions

    # ------------------------------------------------------------------ activity band
    centres, acc, width = _bins(calls, a["wall_s"])
    xs = centres / 60.0
    keep = xs <= span
    xs = xs[keep]
    stack = [_smooth(np.minimum(sum(acc[k] for k in ks) / width, 1.0))[keep]
             for _, ks, _ in BANDS]
    # Calls OVERLAP -- a background simulation runs while the agent reads -- so the occupancies can
    # sum past the minute they are drawn in. A band climbing past 100% would be a lie about the
    # clock; hold the total at one minute and keep the mix.
    tot = np.maximum(sum(stack), 1e-9)
    stack = [s * np.minimum(1.0, 1.0 / tot) for s in stack]
    stack.append(np.clip(1.0 - sum(stack), 0.0, 1.0))     # thinking: the remainder, as a band
    colours = [c for _, _, c in BANDS] + [THINK_COLOUR]
    ax.stackplot(xs, *stack, colors=colours, alpha=0.55, lw=0.7, edgecolor="white", zorder=2)
    ax.set_xlim(0, span)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("activity share")
    ax.set_xlabel("Time (min)")
    style_ax(ax, grid="")

    # ------------------------------------------------------------------ token rate (right axis)
    notes: list[str] = []
    rate = ax.twinx()
    rate.spines["top"].set_visible(False)
    has_rate = len(token_curve) >= 3
    if has_rate:
        for key, colour, label in (("input", SAGE, "rate input"), ("output", NAVY, "rate output")):
            ts, vs = [], []
            for i in range(1, len(token_curve)):
                lo = max(0, i - 2)
                dt = (token_curve[i]["t_s"] - token_curve[lo]["t_s"]) / 60.0
                if dt <= 0:
                    continue
                ts.append(token_curve[i]["t_s"] / 60.0)
                vs.append(max(token_curve[i][key] - token_curve[lo][key], 0) / dt)
            if ts:
                rate.plot(ts, [max(v, 1e-1) for v in vs], color=colour, lw=2.0, label=label,
                          marker="o", ms=3.4, zorder=6, clip_on=True,
                          path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
        rate.set_yscale("log")
        rate.set_ylabel(f"token rate (tok/min, log)   ·   {len(token_curve)} usage samples",
                        labelpad=3)
        # Keep the curves in the lower part of the panel: they share it with nothing above, but a
        # line hugging the top edge reads as clipped.
        lo_r, hi_r = rate.get_ylim()
        rate.set_ylim(lo_r, lo_r * (hi_r / max(lo_r, 1e-6)) ** (1.0 / 0.68))
    else:
        # An empty log axis reads as "no tokens were used". The truth is that this driver reports
        # usage once per turn (here: once for the whole run), which cannot make a rate curve.
        rate.set_yticks([])
        rate.spines["right"].set_visible(False)
        n_rep = len(a.get("token_curve") or [])
        notes.append("no token rate: this driver reports usage once per run, not per message"
                     + (f" ({n_rep} sample{'s' if n_rep != 1 else ''} in this record)"
                        if n_rep else ""))

    # ------------------------------------------------------------------ simulation rail (margin)
    sim_line = ""
    if sim:
        grid = np.linspace(0, span * 60, 1400)
        tiers = [t for t in ("L2", "L3", "L4") if any(e["tier"] == t for e in sim)]
        engines = {t: sorted({e["engine"] for e in sim if e["tier"] == t}) for t in tiers}
        series = [_inflight([e for e in sim if e["tier"] == t], grid) for t in tiers]
        gx = grid / 60.0
        ax.fill_between([0, span], RAIL_LO, RAIL_HI, transform=marg, clip_on=False,
                        color="#F2EEE8", lw=0, zorder=2)
        # ONE LANE PER TIER, each scaled to its OWN peak. Stacked on a shared scale, a tier that
        # runs one job at a time next to one that runs eighteen is two pixels tall -- a legend
        # colour that never appears in the figure. The per-lane peak rides in the legend label.
        peaks = {t: max(float(np.max(s)), 1.0) for t, s in zip(tiers, series)}
        lane_h = (RAIL_HI - RAIL_LO) / len(tiers)
        for i, (t, s) in enumerate(zip(tiers, series)):
            lo = RAIL_LO + (len(tiers) - 1 - i) * lane_h
            ax.fill_between(gx, np.full(len(grid), lo), lo + (s / peaks[t]) * lane_h * 0.92,
                            transform=marg, clip_on=False, color=TIER_COLOUR[t], alpha=0.9, lw=0,
                            zorder=3)
            # The tier tag rides INSIDE its lane: the left margin is one column wide and the
            # rail's own name already has it.
            ax.annotate(t, xy=(0, lo + lane_h * 0.45), xycoords=marg, xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=6.8,
                        color=TIER_COLOUR[t], annotation_clip=False, zorder=4)
        ax.annotate("hardware\nsimulation", xy=(0, (RAIL_LO + RAIL_HI) / 2), xycoords=marg,
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=7.6, color=INK, alpha=0.9, linespacing=1.25, annotation_clip=False)
        ax.plot([0, span], [RAIL_LO] * 2, transform=marg, clip_on=False, color=INK, lw=0.5,
                alpha=0.35, zorder=4)
        n_by = Counter(e["tier"] for e in sim)
        grades = sorted({(min(e["start_s"] for e in sim if e["grade"] == g), g)
                         for g in {e["grade"] for e in sim}})
        sim_line = f"{len(sim)} sims in {len(grades)} grade batches"
        # Each grade is a BATCH: one dotted seam per burst, labelled where the labels fit.
        last_lab = -1e9
        for i, (t0, g) in enumerate(grades):
            ax.axvline(t0 / 60.0, color=INK, lw=0.8, ls=(0, (1, 4)), alpha=0.40, zorder=3)
            if t0 / 60.0 - last_lab < span * 0.085 or t0 / 60.0 > span * 0.94:
                continue
            last_lab = t0 / 60.0
            n = sum(1 for e in sim if e["grade"] == g)
            # Drawn on the RATE axis: a twin is painted over its host, so a label on ``ax`` gets
            # struck through by a rate line that happens to run low.
            rate.annotate(f"grade {i + 1} · {n} sims", xy=(t0 / 60.0, 0.015),
                          xycoords=("data", "axes fraction"), xytext=(4, 0),
                          textcoords="offset points", ha="left", va="bottom", fontsize=7.4,
                          color=INK, alpha=0.9, zorder=9,
                          bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#D9D2C8", lw=0.6))

    # ------------------------------------------------------------------ score milestones
    improved, seen = [], 0
    for v in verdicts:
        if v["n_passed"] > seen:
            seen = v["n_passed"]
            improved.append(v)
    picks = _chip_picks(improved, span)
    for v in improved:
        x = v["t_s"] / 60.0
        ax.axvline(x, color=GOLD, lw=2.0, ls=(0, (4, 3)), zorder=5)
        if v not in picks:
            continue
        usd = _at(cost_curve, v["t_s"], "usd")
        tok = _tokens_at(token_curve, v["t_s"])
        chip = f"{v['n_passed']}/{v['n_capsules']}"
        if usd:
            chip += f" · {cost_prefix}{usd:,.0f}"
        if tok:
            chip += f" · {tok / 1e6:.0f}M"
        near_r, near_l = x > 0.92 * span, x < 0.05 * span
        ha = "right" if near_r else ("left" if near_l else "center")
        ax.plot([x, x], [1.0, (CHIP_LO + CHIP_HI) / 2], transform=marg, clip_on=False, color=GOLD,
                lw=1.4, alpha=0.8, zorder=5)
        ax.annotate(chip, xy=(x, (CHIP_LO + CHIP_HI) / 2), xycoords=marg,
                    xytext=(-4 if near_r else (4 if near_l else 0), 0),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.6, color=INK,
                    fontweight="bold", zorder=8, annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.26", fc="#FDF7EF", ec=GOLD, lw=0.9,
                              alpha=0.96))

    # ------------------------------------------------------------------ header
    fig.text(0.052, 0.958, f"{a['target']} · {a['arm']} — an agent writing a compiler for "
                           f"unseen hardware", ha="left", va="center", fontsize=14.0,
             color=INK, fontfamily=SERIF)

    counts = Counter(c["category"] for c in calls_all)
    usd_end = _at(cost_curve, span * 60, "usd")
    if not cost_label:
        cost_label = f"${usd_end:,.0f}" if usd_end else ""
    tok_end = _tokens_at(token_curve, span * 60)
    bits = [a.get("model", ""),
            f"{span:.0f} of {wall_min:.0f} min shown" if span < wall_min else f"{wall_min:.0f} min",
            cost_label or "cost not reported",
            f"{tok_end / 1e6:.0f}M tok" if tok_end else "",
            f"{len(calls_all)} tool calls", f"{counts['selfcheck']} self-checks",
            f"best {best}/{max((v['n_capsules'] for v in verdicts), default=0)}"]
    fig.text(0.938, 0.958, "  ·  ".join(b for b in bits if b), ha="right", va="center",
             fontsize=8.8, color=INK,
             bbox=dict(boxstyle="round,pad=0.34", fc="#FDF7EF", ec="#D9D2C8", lw=0.7, alpha=0.95))

    fb_s = sum(c["duration_s"] for c in calls if CATEGORIES[c["category"]][1])
    dev_s = sum(c["duration_s"] for c in calls if not CATEGORIES[c["category"]][1])
    blocked = a.get("blocked") or []
    facts = [f"{_dur(fb_s)} waiting for feedback   vs   {_dur(dev_s)} doing the work"]
    if sim_line:
        facts.append(sim_line)
    if best_at is not None and wall_min - best_at > 1:
        facts.append(f"then {(wall_min - best_at) / 60:.1f} h more"
                     + (f", on {len(blocked)} tests that could not move" if blocked
                        else " without moving"))
    # The run's headline bottleneck, when there is one: a self-check that is mostly a rebuild.
    costs = a.get("grade_costs") or []
    heavy = max((g for g in costs if g["sim_active_s"] > 0 and g["build_s"] > g["sim_active_s"]),
                key=lambda g: g["build_s"] / g["sim_active_s"], default=None)
    if heavy:
        notes.insert(0, f"a self-check ≈ {heavy['adapter_wall_s'] / 60:.0f} min: "
                        f"{heavy['build_s']:.0f} s of build for "
                        f"{heavy['sim_active_s']:.0f} s of simulation — "
                        f"{heavy['build_s'] / heavy['sim_active_s']:.0f}× more build than simulate")
    # One row holds both: measure before drawing and drop the least essential fact rather than let
    # two texts overprint each other (they did, and the row became unreadable).
    row_in = (0.938 - 0.052) * fig.get_size_inches()[0]
    note_in = max((len(n) for n in notes), default=0) * 0.061
    while facts and (sum(len(f) + 5 for f in facts) * 0.061 + note_in + 0.4) > row_in:
        facts.pop(1 if len(facts) > 2 else len(facts) - 1)
    if facts:
        fig.text(0.052, 0.900, "   ·   ".join(facts), ha="left", va="center", fontsize=8.8,
                 color=INK, alpha=0.9)
    if notes:
        fig.text(0.938, 0.900, "\n".join(notes), ha="right", va="center", fontsize=8.8,
                 color=TIER_COLOUR["L3"], fontweight="bold", linespacing=1.45)

    # ------------------------------------------------------------------ legend (one row, below)
    whole = Counter(c["category"] for c in calls_all)
    handles = [Patch(facecolor=c, alpha=0.55, edgecolor="white",
                     label=f"{lab} · {sum(whole[k] for k in ks)}×") for lab, ks, c in BANDS]
    handles.append(Patch(facecolor=THINK_COLOUR, alpha=0.55, edgecolor="white", label=THINK_LABEL))
    if has_rate:
        handles += [Line2D([0], [0], color=SAGE, lw=2, marker="o", ms=3.4, label="rate input"),
                    Line2D([0], [0], color=NAVY, lw=2, marker="o", ms=3.4, label="rate output")]
    handles.append(Line2D([0], [0], color=GOLD, lw=2, ls=(0, (4, 3)), label="test-pass milestone"))
    if sim:
        handles += [Patch(facecolor=TIER_COLOUR[t], alpha=0.9,
                          label=f"sim {t} · {'/'.join(engines[t])} · {n_by[t]}× · "
                                f"peak {int(peaks[t])}") for t in tiers]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True, fontsize=7.8,
               columnspacing=1.0, handlelength=1.5, handletextpad=0.5, borderpad=0.5,
               bbox_to_anchor=(0.5, 0.006))

    out.mkdir(parents=True, exist_ok=True)
    stem = f"fig12_phase1_anatomy_{a['run_id']}"
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, facecolor=PAGE_BG)
    plt.close(fig)
    return out / f"{stem}.png"


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
