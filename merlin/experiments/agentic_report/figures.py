#!/usr/bin/env python3
"""Render the agentic-run report's figures from ``run_facts.json`` and nothing else.

Every number drawn here is read from the facts file, so no caption can drift from the run that
produced it. Where a run could not supply a number, the figure DRAWS THE GAP -- a hatched band and a
generated caption naming the reason -- because omitting the run makes the chart claim a smaller,
tidier study than the one we ran.

    figures.py [--facts PATH] [--out DIR] [--only NAME]...
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import matplotlib.pyplot as plt                                              # noqa: E402
import numpy as np                                                           # noqa: E402
from matplotlib.lines import Line2D                                          # noqa: E402
from matplotlib.patches import Patch                                         # noqa: E402

from merlin.plotting.merlin_plotstyle import (BLUE, GOLD, INK, MAUVE, NAVY,       # noqa: E402
                                              SAGE, SERIF, SLATE, suptitle, title,
                                              use_merlin_style, vbars)
from merlin.plotting.merlin_plotstyle import style_ax as _house_style_ax             # noqa: E402

#: The house palette carries the series identity; the page does not. These figures are read on a
#: white page and in a white slide deck, so the canvas is white while every ink, bar and accent stays
#: the house colour. `use_merlin_style()` sets the cream canvas repo-wide, so it is overridden here
#: rather than changed there -- other figures in this repo still want the cream.
PAGE_BG = "#FFFFFF"
#: Where a fill needs to read as "absent" against white rather than against cream.
EMPTY_FILL = "#EDE7E0"

ARM_ORDER = ["arm1", "arm2", "arm3", "arm4", "eqsat", "UNKNOWN"]
ARM_COLOR = {"arm1": MAUVE, "arm2": NAVY, "arm3": SLATE, "arm4": SAGE,
             "eqsat": BLUE, "UNKNOWN": "#bdb2a4"}
ARM_LABEL = {"arm1": "arm 1 · raw C++", "arm2": "arm 2 · C++ & infra",
             "arm3": "arm 3 · Merlin Python", "arm4": "arm 4 · Merlin & CIRCT",
             "eqsat": "e-graph seam", "UNKNOWN": "unclassified"}
#: One hatch, one meaning: this cell is not a measurement.
GAP_HATCH = "//"

def style_ax(ax, *, grid="y"):
    """The house axes treatment, on a white canvas.

    Identical to the house helper -- ink spines, dotted value grid, no top/right -- except the
    facecolor, which the house sets to cream for every figure in the repo. Only the page changes;
    every series colour stays the house one."""
    _house_style_ax(ax, grid=grid)
    ax.set_facecolor(PAGE_BG)


#: The substrate every arm gets regardless of bundle. Excluded when asking whether an ARM's
#: OWN tools were reached for, since the substrate would mask the answer.
SUBSTRATE_NAMES = frozenset({"agent_selfcheck.py", "simjob.py"})

_REGISTRY: dict[str, callable] = {}


def figure(name):
    def wrap(fn):
        _REGISTRY[name] = fn
        return fn
    return wrap


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=PAGE_BG)
    plt.close(fig)
    print(f"  wrote {name}.png / .svg")


def _caption(ax, text: str, y: float = -0.16) -> None:
    """Provenance line under ONE axes. Use `_figcaption` when several panels share a figure."""
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, color=INK, alpha=0.78, wrap=True)


def _figcaption(fig, text: str, y: float = 0.008) -> None:
    """One provenance line for the whole figure. Panels writing their own overprint each other."""
    fig.text(0.012, y, text, ha="left", va="bottom", fontsize=8.0, color=INK, alpha=0.78, wrap=True)


def _gap_note(counts: Counter) -> str:
    if not counts:
        return ""
    return "; ".join(f"{n} {reason}" for reason, n in counts.most_common())


def _cost(f: dict) -> tuple[float | None, str]:
    if f.get("cost_usd") is not None:
        return f["cost_usd"], "metered"
    if f.get("notional_usd") is not None:
        return f["notional_usd"], "notional"
    return None, "unpriced"


# --------------------------------------------------------------------------- figures

@figure("fig01_arm_ladder")
def arm_ladder(facts, out):
    """The ladder, drawn only where a ladder actually exists.

    A bar chart of the best run per arm invites a comparison the data does not support: on one target
    the arms were graded against 11, 27 and 57 capsules, and putting those fractions side by side
    says the arms differ when what differs is the corpus. So this draws only LADDERS -- a set of runs
    sharing one tag, and therefore one corpus, one model and one day -- which is the only
    like-for-like contrast the study contains. The corpus size is in each panel's title, and a rung
    with no run is drawn as an explicit gap rather than omitted."""
    ladders = defaultdict(list)
    for f in facts:
        if f.get("ladder"):
            ladders[f["ladder"]].append(f)
    if not ladders:
        return
    keys = sorted(ladders, key=lambda k: -max((f["capsules"] or 0) for f in ladders[k]))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 5.8), squeeze=False)
    for ax, key in zip(axes[0], keys):
        members = {f["arm"]: f for f in ladders[key]}
        corpora = sorted({f["capsules"] for f in ladders[key] if f["capsules"]})
        arms = ARM_ORDER[:4]
        xs = np.arange(len(arms))
        total = corpora[0] if len(corpora) == 1 else max(corpora)
        ticks = []
        for x, a in zip(xs, arms):
            f = members.get(a)
            if f is None:
                # A rung nobody ran is a GAP, not a score. Drawn as a low hatched stub so it can
                # never be read off the value axis.
                ax.bar(x, total * 0.06, 0.62, color="none", edgecolor=INK, lw=1.0,
                       hatch=GAP_HATCH, zorder=3)
                ax.text(x, total * 0.10, "not run", ha="center", fontsize=8.0, color=MAUVE)
                ticks.append(f"arm {a[-1]}\n—")
                continue
            ax.bar(x, f["passed"], 0.62, color=ARM_COLOR[a], edgecolor=INK, lw=1.2, zorder=3)
            ax.text(x, f["passed"] + total * 0.035, str(f["passed"]), ha="center",
                    fontsize=11, color=INK, fontweight="bold")
            cost, kind = _cost(f)
            money = "unpriced" if cost is None else (
                f"${cost:.0f}" if kind == "metered" else f"~${cost:.0f}")
            hours = (f.get("active_wall_s") or 0) / 3600.0
            clock = f"{hours:.1f} h" if hours else "no clock"
            ticks.append(f"arm {a[-1]}\n{money}\n{clock}")
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=8.4, linespacing=1.45)
        ax.set_ylim(0, total * 1.20)
        ax.set_ylabel("capsules passed" if key == keys[0] else "")
        style_ax(ax)
        target, _, tag = key.split("/")
        quality = ladders[key][0].get("ladder_quality", "full")
        badge = {"full": "", "patch": "  ·  PATCH LADDER",
                 "null": "  ·  NULL CELL"}.get(quality, "")
        title(ax, f"{target} · {tag}{badge}\n{total} capsules", fs=10.5, pad=8)
        if quality != "full":
            # Dim a ladder that is not a from-scratch contrast, so it cannot be read as one at a
            # glance. The note beneath says which kind it is and why.
            ax.set_facecolor("#F2EDE7")
            note = ladders[key][0].get("ladder_note") or ""
            # Wrapped to the panel, not clipped at a character count: an unwrapped note runs into
            # the neighbouring panel's note and the two become one unreadable line.
            wrapped = "\n".join(textwrap.wrap(note, width=44)[:4])
            ax.text(0.0, -0.34, wrapped, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.0, color=MAUVE)
    _figcaption(fig, "Only tag-matched ladders are shown: every rung in a panel was graded against "
                     "the same corpus with the same model, so the bars are comparable. Cost (~ = "
                     "notional) and active wall are under each rung. Fractions from DIFFERENT panels "
                     "are not comparable — the corpora differ. A shaded panel is not a from-scratch "
                     "contrast: PATCH means its rungs adjusted an existing compiler, NULL means every "
                     "rung scored zero.", y=0.02)
    suptitle(fig, "The ladder, where a like-for-like ladder exists", y=1.02)
    fig.subplots_adjust(bottom=0.32, top=0.83, wspace=0.30)
    _save(fig, out, "fig01_arm_ladder")


@figure("fig01b_best_per_cell")
def best_per_cell(facts, out):
    """Best run per (target, arm) as a labelled scatter, with corpus size on the point.

    Deliberately NOT a bar chart: the corpora differ between cells, so height would imply a
    comparison that is not available. The reader is shown the fraction and its denominator."""
    sel = [f for f in facts if f.get("selected") and f["phase"] == "phase1" and f.get("capsules")]
    targets = sorted({f["target"] for f in sel})
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for f in sel:
        y = targets.index(f["target"])
        x = ARM_ORDER.index(f["arm"])
        frac = f["passed"] / f["capsules"]
        ax.scatter(x, y, s=140 + 900 * frac, color=ARM_COLOR[f["arm"]], edgecolor=INK, lw=1.0,
                   alpha=0.9, zorder=3)
        ax.text(x, y - 0.30, f"{f['passed']}/{f['capsules']}", ha="center", fontsize=9.2,
                color=INK, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER[:4]], fontsize=9)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=10)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.7, len(targets) - 0.3)
    style_ax(ax, grid="both")
    _figcaption(fig, "Marker area is the pass fraction; the label carries its denominator. Cells are "
                     "NOT comparable across rows or columns — each was graded against whatever corpus "
                     "that target had when the run happened, and the corpus grew from 11 to 96 over "
                     "the study. An empty cell had no graded run at all.")
    suptitle(fig, "Best run in each target × arm cell")
    fig.subplots_adjust(bottom=0.26)
    _save(fig, out, "fig01b_best_per_cell")


@figure("fig02_capsules_over_time")
def capsules_over_time(facts, out):
    """Capsules passing over wall time, one lane per selected run. The reference figure."""
    rows = [f for f in facts if f.get("selected") and f.get("pass_milestones")]
    rows.sort(key=lambda f: (f["target"], f["arm"]))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 0.92 * len(rows) + 1.0),
                             squeeze=False, sharex=True)
    for ax, f in zip(axes[:, 0], rows):
        ms = f["pass_milestones"]
        xs = [m["t_s"] / 60.0 for m in ms]
        ys = [m["n_passed"] for m in ms]
        total = ms[-1]["n_capsules"] or 1
        end = (f.get("pass_wall_s") or xs[-1] * 60) / 60.0
        xs, ys = xs + [end], ys + [ys[-1]]
        colour = ARM_COLOR[f["arm"]]
        ax.step(xs, ys, where="post", color=colour, lw=2.0)
        ax.fill_between(xs, 0, ys, step="post", color=colour, alpha=0.18)
        ax.set_ylim(0, total * 1.28)
        ax.set_yticks([0, total])
        ax.text(xs[-1], ys[-1], f"  {ys[-1]}/{total}", va="center", fontsize=9,
                color=colour, fontweight="bold")
        ax.text(0.004, 0.98, f"{f['target']} · {ARM_LABEL[f['arm']]} · {f['run_id'][:34]}",
                transform=ax.transAxes, va="top", fontsize=8.4, color=INK, alpha=0.85)
        style_ax(ax)
        if f.get("availability", {}).get("passes", {}).get("kind") == "derived":
            ax.text(0.997, 0.96, "mtime clock", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.0, color=MAUVE, alpha=0.85)
    axes[-1, 0].set_xlabel("Time (min)")
    skipped = Counter()
    for f in facts:
        if f.get("selected") and not f.get("pass_milestones"):
            skipped[f.get("availability", {}).get("passes", {}).get("reason", "no reason recorded")[:70]] += 1
    n_mtime = sum(1 for f in rows
                  if f.get("availability", {}).get("passes", {}).get("kind") == "derived")
    _figcaption(fig, f"{len(rows)} selected run(s) drawn. {len(rows) - n_mtime} carry a self-check "
                     f"log and so a clock the run itself wrote; the {n_mtime} marked 'mtime clock' "
                     f"were graded continuously and kept no such log, so their x-axis is the verdict "
                     f"files' modification time — real enough to order events, not a stamp the run "
                     f"recorded."
                     + (f" {sum(skipped.values())} selected run(s) kept no progress record at all: "
                        f"{_gap_note(skipped)}." if skipped else ""))
    suptitle(fig, "Capsules passing over time", y=0.995)
    fig.subplots_adjust(top=0.955, bottom=0.075, hspace=0.55)
    _save(fig, out, "fig02_capsules_over_time")


@figure("fig03_spend_over_time")
def spend_over_time(facts, out):
    """Cost against active wall, metered and notional in SEPARATE panels.

    They are different quantities -- money spent versus what a seat run would have cost metered --
    and one axis carrying both invites a total that means nothing."""
    sel = [f for f in facts if f.get("selected")]
    panels = [("metered", "cost (USD, billed)"), ("notional", "cost (USD, notional — a seat is\nnot billed per token)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (kind, ylab) in zip(axes, panels):
        drawn = 0
        for f in sel:
            value, k = _cost(f)
            hours = (f.get("active_wall_s") or 0) / 3600.0
            if k != kind or value is None or hours <= 0:
                continue
            drawn += 1
            colour = ARM_COLOR[f["arm"]]
            ax.plot([0, hours], [0, value], color=colour, lw=1.6, alpha=0.9,
                    marker="o", markevery=[1], ms=6)
            ax.text(hours, value, f"  {f['target'][:3]}·{f['arm'][-1]}", fontsize=7.6,
                    color=colour, va="center")
        ax.set_xlabel("active wall time (h)")
        ax.set_ylabel(ylab)
        style_ax(ax, grid="both")
        title(ax, f"{kind} spend", fs=13)
        _caption(ax, f"{drawn} selected run(s) priced this way", y=-0.22)
    unp = sum(1 for f in sel if _cost(f)[1] == "unpriced")
    fig.text(0.5, -0.04, f"{unp} selected run(s) carry no dollar figure at all and appear in neither "
                         f"panel — an unpriced model is not a free one.",
             ha="center", fontsize=8.2, color=INK, alpha=0.8)
    suptitle(fig, "What the runs cost")
    _save(fig, out, "fig03_spend_over_time")


@figure("fig04_token_accounting")
def token_accounting(facts, out):
    """Where the tokens went. Cache reads dominate by ~30x, so the axis is log."""
    sel = [f for f in facts if f.get("selected") and (f.get("total_tokens") or 0) > 0]
    sel.sort(key=lambda f: -(f["total_tokens"] or 0))
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(sel) + 2.2))
    ys = np.arange(len(sel))
    series = [("cache_read_tokens", SLATE, "cache read"),
              ("cache_creation_tokens", GOLD, "cache write"),
              ("input_tokens", NAVY, "fresh input"),
              ("output_tokens", MAUVE, "output")]
    left = np.zeros(len(sel))
    for key, colour, label in series:
        vals = np.array([max(f.get(key) or 0, 0) for f in sel], dtype=float)
        ax.barh(ys, vals, left=left, color=colour, edgecolor=INK, lw=0.6, label=label, height=0.66)
        left += vals
    for y, f in zip(ys, sel):
        split = f.get("availability", {}).get("token_split", {}).get("kind")
        if split != "measured":
            ax.text(left[y] * 1.02, y, "  reads and writes not separable", va="center",
                    fontsize=7.2, color=MAUVE)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{f['target'][:8]} {f['arm']} {f['run_id'][:26]}" for f in sel], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("tokens")
    style_ax(ax, grid="x")
    ax.legend(loc="lower right", fontsize=9)
    n_split = sum(1 for f in sel if f.get("availability", {}).get("token_split", {}).get("kind") == "measured")
    _caption(ax, f"{n_split} of {len(sel)} run(s) recorded cache reads and writes separately; the rest "
                 f"recorded only their sum, which cannot be undone after the fact — reads and writes "
                 f"are billed about an order of magnitude apart.", y=-0.10)
    suptitle(fig, "Token accounting: the run is mostly re-reading its own context")
    _save(fig, out, "fig04_token_accounting")


@figure("fig05_where_the_wall_went")
def where_the_wall_went(facts, out):
    """Tool occupancy against the clock, per run, with the unmeasured remainder hatched."""
    sel = [f for f in facts if f.get("selected") and (f.get("active_wall_s") or 0) > 0]
    sel.sort(key=lambda f: (f["target"], f["arm"]))
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(sel) + 2.4))
    ys = np.arange(len(sel))
    for y, f in zip(ys, sel):
        wall_h = f["active_wall_s"] / 3600.0
        tool_h = min((f.get("tool_seconds") or 0) / 3600.0, wall_h)
        rl_h = (f.get("rate_limit_wait_s") or 0) / 3600.0
        ax.barh(y, tool_h, color=SAGE, edgecolor=INK, lw=0.7, height=0.62, label="_")
        ax.barh(y, max(wall_h - tool_h, 0), left=tool_h, color="none", edgecolor=INK, lw=0.7,
                height=0.62, hatch=GAP_HATCH, label="_")
        if rl_h > 0:
            ax.barh(y, rl_h, left=wall_h, color=MAUVE, edgecolor=INK, lw=0.7, height=0.62, alpha=0.55)
        if f.get("span_source") == "":
            ax.text(0.02, y, " no tool spans recoverable", va="center", fontsize=7.4, color=MAUVE)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{f['target'][:8]} {f['arm']} {f['run_id'][:26]}" for f in sel], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("hours")
    style_ax(ax, grid="x")
    ax.legend(handles=[Patch(facecolor=SAGE, edgecolor=INK, label="occupied by a tool call"),
                       Patch(facecolor="none", edgecolor=INK, hatch=GAP_HATCH,
                             label="clock not attributed to any span"),
                       Patch(facecolor=MAUVE, edgecolor=INK, alpha=0.55, label="rate-limit wait")],
              loc="lower right", fontsize=9)
    _caption(ax, "The hatched remainder is the agent thinking, plus any tool call whose duration the "
                 "stream could not preserve — it is unattributed time, not idle time.", y=-0.10)
    suptitle(fig, "Where the wall clock went")
    _save(fig, out, "fig05_where_the_wall_went")


@figure("fig06_concurrency")
def concurrency_fig(facts, out):
    """Overlap share per run, split by arm. The parallelism question, answered per run."""
    usable = [f for f in facts
              if f.get("availability", {}).get("concurrency", {}).get("kind") in ("measured", "derived")
              and (f.get("span_wall_s") or 0) > 60]
    if not usable:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    arms = [a for a in ARM_ORDER if any(f["arm"] == a for f in usable)]
    for i, arm in enumerate(arms):
        vals = [f["overlap_share"] for f in usable if f["arm"] == arm]
        jitter = (np.random.default_rng(7).random(len(vals)) - 0.5) * 0.26
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=26, color=ARM_COLOR[arm],
                   edgecolor=INK, lw=0.5, alpha=0.85, zorder=3)
        if vals:
            ax.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color=INK, lw=2.2, zorder=4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a].split("·")[0].strip() for a in arms], fontsize=9)
    ax.set_ylabel("share of wall with ≥2 tool calls live")
    style_ax(ax)
    title(ax, "Tool concurrency by arm", fs=13)
    phases = defaultdict(list)
    for f in usable:
        phases[f["phase"]].append(f)
    labels = sorted(phases)
    xs = np.arange(len(labels))
    shares = [sum(1 for f in phases[p] if f["overlap_share"] > 0) / len(phases[p]) for p in labels]
    vbars(ax2, xs, shares, [NAVY if p == "phase1" else SAGE for p in labels], width=0.5)
    for x, p in zip(xs, labels):
        peaks = sorted(f["max_concurrent"] for f in phases[p])
        med_peak = int(np.median(peaks))
        n_over = sum(1 for f in phases[p] if f["overlap_share"] > 0)
        ax2.text(x, shares[x] + 0.035,
                 f"{n_over}/{len(phases[p])} runs overlap\nmedian peak {med_peak}, max {peaks[-1]}",
                 ha="center", fontsize=9, color=INK)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("share of runs with any tool overlap")
    style_ax(ax2)
    title(ax2, "How often it happens at all", fs=13)
    # The outlier sentence is DERIVED. It was hardcoded once and went stale the moment the flush
    # guard tightened, which is the exact drift this kit exists to prevent.
    peakiest = max(usable, key=lambda f: (f["max_concurrent"], -f["overlap_share"]))
    _figcaption(fig, f"{len(usable)} run(s) whose overlap survived the flush check (left bar = "
                     f"median). A run whose overlap moved when flush-suspect spans were dropped is "
                     f"excluded: an overlap that depends on those was measuring the reader, not the "
                     f"agent. The right panel counts RUNS rather than reporting a peak, because peak "
                     f"concurrency is carried by single outliers — the deepest here reaches "
                     f"{peakiest['max_concurrent']} while occupying "
                     f"{peakiest['overlap_share']:.1%} of its wall "
                     f"({peakiest['target']} {peakiest['arm']}, {peakiest['run_id'][:30]}).")
    suptitle(fig, "Does the agent do things in parallel?", y=1.0)
    fig.subplots_adjust(bottom=0.30, top=0.86)
    _save(fig, out, "fig06_concurrency")


@figure("fig07_tier_cost")
def tier_cost(facts, out):
    """What one capsule costs to grade at each tier, passing capsules only."""
    rows = []
    for f in facts:
        for key, s in (f.get("tier_cost") or {}).items():
            tier, status = key.split("/", 1)
            if status != "pass" or s.get("median_active_s") is None:
                continue
            rows.append((f["phase"], tier, s["median_active_s"], s["n"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    groups = sorted({(p, t) for p, t, _, _ in rows})
    xs = np.arange(len(groups))
    for i, (phase, tier) in enumerate(groups):
        vals = [v for p, t, v, _ in rows if (p, t) == (phase, tier)]
        jitter = (np.random.default_rng(3).random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=30,
                   color=NAVY if phase == "phase1" else SAGE, edgecolor=INK, lw=0.5, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [np.median(vals)] * 2, color=INK, lw=2.2, zorder=4)
        # Beside the group, not through it: a label at median*k lands inside the point cloud.
        ax.text(i + 0.36, np.median(vals), f"{np.median(vals):.3g}s\nn={len(vals)}",
                ha="left", va="center", fontsize=8.4, color=INK, fontweight="bold", zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(groups) - 0.15)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t}\n{p}" for p, t in groups], fontsize=9)
    ax.set_ylabel("median seconds to grade one capsule (log)")
    style_ax(ax, grid="y")
    ax.legend(handles=[Patch(facecolor=NAVY, edgecolor=INK, label="functional lane"),
                       Patch(facecolor=SAGE, edgecolor=INK, label="performance lane")],
              loc="upper left", fontsize=9)
    _caption(ax, "Passing capsules only: a failing capsule aborts in hundredths of a second, so "
                 "pooling the two yields a median about the pass rate rather than the cost. "
                 "Build + simulation; queueing excluded. Carried certificates record no duration "
                 "and are not counted.", y=-0.16)
    suptitle(fig, "The certifying tier is the whole cost of a grade")
    _save(fig, out, "fig07_tier_cost")


@figure("fig08_phase_tools")
def phase_tools(facts, out):
    """The tool surface each phase exposes. Read from the runs, never assumed."""
    p2 = [f for f in facts if f["phase"] == "phase2" and f.get("broker_actions")]
    if not p2:
        return
    actions = sorted({a for f in p2 for a in f["broker_actions"]})
    families = defaultdict(list)
    for a in actions:
        families[a.split("-", 1)[0]].append(a)
    totals = Counter()
    seconds = Counter()
    for f in p2:
        for action, d in (f.get("broker_totals") or {}).items():
            totals[action] += d["calls"]
            seconds[action] += d["seconds"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    fams = sorted(families, key=lambda k: -len(families[k]))
    vbars(ax, np.arange(len(fams)), [len(families[k]) for k in fams],
          [SAGE if k != "tuning" else GOLD for k in fams], width=0.55)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels(fams, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("actions in the family")
    style_ax(ax)
    title(ax, f"The performance lane's {len(actions)} brokered actions", fs=13)


    # The free analysis action is the whole point of the caption, and it is invisible in a
    # most-common-by-seconds list precisely BECAUSE it costs nothing. Pin it in.
    top = [a for a, _ in seconds.most_common(7)]
    free = [a for a in actions if a.startswith("analyze")]
    for a in free:
        if a not in top and a in seconds:
            top = top[:6] + [a]
    top = top[::-1]
    ax2.barh(np.arange(len(top)), [seconds[a] / 60 for a in top], color=NAVY,
             edgecolor=INK, lw=0.9, height=0.6)
    for i, a in enumerate(top):
        ax2.text(seconds[a] / 60 * 1.02, i, f" {totals[a]} calls", va="center", fontsize=8.4, color=INK)
    ax2.set_yticks(np.arange(len(top)))
    ax2.set_yticklabels(top, fontsize=8.6)
    ax2.set_xlabel("minutes across all trials")
    style_ax(ax2, grid="x")
    title(ax2, "Where the brokered time goes", fs=13)
    _figcaption(fig, "The action set is derived per run from the candidate's own manifest plus "
                     "descriptor-declared probes, so it is read from STAGE_CONTEXT rather than "
                     "hardcoded. The measurement dominates the brokered time; `analyze-command-"
                     "buffers` reads only the candidate's own emitted buffers and so costs no oracle "
                     "time at all, which is the point of adding it.")
    suptitle(fig, "Phase 1 hands the agent a simulator; phase 2 hands it a closed action set",
             y=1.01)
    fig.subplots_adjust(bottom=0.26, top=0.84, wspace=0.35)
    _save(fig, out, "fig08_phase_tools")


@figure("fig09_coverage_matrix")
def coverage_matrix(facts, out):
    """Which runs can supply which fields. The denominator, made visible."""
    fields = ["arm", "tokens", "cost", "token_split", "passes", "spans", "concurrency"]
    sel = [f for f in facts if f.get("selected")]
    sel.sort(key=lambda f: (f["target"], f["arm"]))
    if not sel:
        return
    kind_val = {"measured": 2, "derived": 1, "unavailable": 0}
    grid = np.zeros((len(sel), len(fields)))
    for i, f in enumerate(sel):
        for j, name in enumerate(fields):
            grid[i, j] = kind_val.get(f.get("availability", {}).get(name, {}).get("kind"), 0)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(sel) + 2.4))
    ax.imshow(grid, cmap=ListedColormap([EMPTY_FILL, GOLD, SAGE]), aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{f['target'][:8]} {f['arm']} {f['run_id'][:26]}" for f in sel], fontsize=7.4)
    ax.set_xticks(np.arange(-0.5, len(fields), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sel), 1), minor=True)
    ax.grid(which="minor", color=PAGE_BG, lw=1.6)
    ax.tick_params(which="minor", length=0)
    ax.legend(handles=[Patch(facecolor=SAGE, label="measured"),
                       Patch(facecolor=GOLD, label="derived"),
                       Patch(facecolor=EMPTY_FILL, label="unavailable")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    # Grouped by FIELD, not by reason text: every reason names the run it came from, so the strings
    # are all distinct and a "commonest causes" tally over them is a list of one-offs.
    missing = Counter()
    derived_n = Counter()
    for f in sel:
        for name in fields:
            kind = f.get("availability", {}).get(name, {}).get("kind")
            if kind == "unavailable":
                missing[name] += 1
            elif kind == "derived":
                derived_n[name] += 1
    parts = [f"{n} run(s) cannot supply `{k}`" for k, n in missing.most_common()]
    dparts = [f"`{k}` is reconstructed for {n}" for k, n in derived_n.most_common(3)]
    _figcaption(fig, "Gaps are drawn, not dropped — an absent row would read as 'no such run', which "
                     "is a different claim. " + ("; ".join(parts) + ". " if parts else "")
                     + ("Where a value is derived rather than read: " + "; ".join(dparts) + "."
                        if dparts else ""))
    suptitle(fig, "What each selected run can actually tell us", y=1.0)
    fig.subplots_adjust(bottom=0.16, top=0.93)
    _save(fig, out, "fig09_coverage_matrix")


@figure("fig10_rate_panels")
def rate_panels(facts, out):
    """Activity share over wall time with the token-rate overlay — the reference view.

    One panel per run: the stacked band is what the agent was occupied by, the lines are how fast
    tokens moved. Only runs whose reconstructed token curve AGREES with the total the harness
    recorded independently are drawn, because a rate curve is the easiest thing here to draw
    plausibly and wrongly."""
    rows = [f for f in facts if f.get("selected") and f.get("token_curve")
            and (f.get("span_wall_s") or 0) > 60]
    rows.sort(key=lambda f: (f["target"], f["arm"]))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(11.5, 1.55 * len(rows) + 1.2), squeeze=False)
    for ax, f in zip(axes[:, 0], rows):
        curve = f["token_curve"]
        xs = [c["t_s"] / 60.0 for c in curve]
        span_end = max((f.get("span_wall_s") or 0) / 60.0, xs[-1] if xs else 0)

        # Activity share: the fraction of each bin occupied by a tool call. What is NOT occupied is
        # the agent thinking (or a duration the stream could not preserve), drawn as the remainder.
        band = f.get("activity_bins") or []
        if band:
            bx = [b["t_s"] / 60.0 for b in band]
            by = [b["occupied"] for b in band]
            ax.fill_between(bx, 0, by, color=SAGE, alpha=0.45, lw=0, step="mid")
            ax.fill_between(bx, by, 1.0, color="none", edgecolor=INK, lw=0.0,
                            hatch=GAP_HATCH, alpha=0.30, step="mid")
        else:
            ax.text(0.5, 0.5, "no tool spans to place on this axis", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=MAUVE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_ylabel("share", fontsize=8)

        rate = ax.twinx()
        drew = False
        for key, colour, label in (("output", NAVY, "output"), ("cache_read", SLATE, "cached input")):
            ts, vs = [], []
            for i in range(1, len(curve)):
                lo = max(0, i - 5)
                dt = (curve[i]["t_s"] - curve[lo]["t_s"]) / 60.0
                if dt <= 0:
                    continue
                ts.append(curve[i]["t_s"] / 60.0)
                vs.append(max(curve[i][key] - curve[lo][key], 0) / dt)
            if len(ts) >= 2:
                rate.plot(ts, [max(v, 1e-1) for v in vs], color=colour, lw=1.7, label=label)
                drew = True
        if drew:
            rate.set_yscale("log")
        rate.set_ylabel("tok/min", fontsize=8)
        rate.tick_params(labelsize=7)

        for m in f.get("pass_milestones", []):
            t = m["t_s"] / 60.0
            if t <= span_end:
                ax.axvline(t, color=GOLD, lw=1.4, ls=(0, (4, 3)), alpha=0.9)
        cost, kind = _cost(f)
        money = "unpriced" if cost is None else (f"${cost:.0f}" if kind == "metered" else f"~${cost:.0f}")
        ax.text(0.004, 1.06, f"{f['target']} · {ARM_LABEL[f['arm']]} · {f['run_id'][:30]}  —  "
                             f"{span_end:.0f} min · {money} · "
                             f"{(f.get('total_tokens') or 0) / 1e6:.0f}M tok · "
                             f"final {f.get('passed')}/{f.get('capsules')}",
                transform=ax.transAxes, va="bottom", fontsize=8.2, color=INK)
        ax.set_xlim(0, span_end)
        style_ax(ax, grid="x")
    axes[-1, 0].set_xlabel("Time (min)")
    fig.legend(handles=[Patch(facecolor=SAGE, alpha=0.30, label="occupied by a tool call"),
                        Patch(facecolor="none", edgecolor=INK, hatch=GAP_HATCH,
                              label="thinking / duration not preserved"),
                        Line2D([0], [0], color=NAVY, lw=1.7, label="output tok/min"),
                        Line2D([0], [0], color=SLATE, lw=1.7, label="cached input tok/min"),
                        Line2D([0], [0], color=GOLD, lw=1.4, ls=(0, (4, 3)), label="capsule-pass milestone")],
               loc="lower center", ncol=5, fontsize=8.5, frameon=True)
    skipped = sum(1 for f in facts if f.get("selected") and not f.get("token_curve"))
    _figcaption(fig, f"{len(rows)} of the selected runs carry a token curve that agrees with the "
                     f"total the harness recorded independently; {skipped} do not and are omitted "
                     f"rather than drawn from an unverified reconstruction. The green band is the "
                     f"share of each time slice covered by a tool call, from measured span overlap; "
                     f"the hatched remainder is the agent thinking plus any call whose duration the "
                     f"stream could not preserve — the two are not separable here.", y=0.045)
    suptitle(fig, "Activity and token rate over the run", y=0.995)
    fig.subplots_adjust(top=0.95, bottom=0.10, hspace=0.75)
    _save(fig, out, "fig10_rate_panels")


@figure("fig11_granted_vs_used")
def granted_vs_used(facts, out):
    """Which granted tools the functional agent actually reached for.

    An arm is DEFINED by what it may read, but granting a tool and the tool mattering are different
    claims, and only the second explains a score. Brokered tools expose a filename a command line can
    name, so a zero there means the shim was never invoked. Path grants expose no such name and their
    use is invisible to this reader — they are shown as granted-but-unmeasurable rather than as zero,
    because drawing them at zero would assert something the data cannot support."""
    rows = [f for f in facts if f.get("selected") and f["phase"] != "phase2"
            and f.get("granted_tools") and f.get("n_spans")]
    rows.sort(key=lambda f: (f["arm"], f["target"]))
    if not rows:
        return
    names: list[str] = []
    for f in rows:
        for t in f["granted_tools"]:
            if t["name"] not in names:
                names.append(t["name"])
    invocable = {t["name"]: t["invocable"] for f in rows for t in f["granted_tools"]}
    order = [n for n in names if invocable.get(n)] + [n for n in names if not invocable.get(n)]

    fig, ax = plt.subplots(figsize=(1.05 * len(order) + 5.0, 0.34 * len(rows) + 2.2))
    for y, f in enumerate(rows):
        have = {t["name"]: t for t in f["granted_tools"]}
        for x, name in enumerate(order):
            t = have.get(name)
            if t is None:
                continue                       # not granted to this arm: leave the cell empty
            if not t["invocable"]:
                ax.add_patch(plt.Rectangle((x - 0.42, y - 0.38), 0.84, 0.76, facecolor="none",
                                           edgecolor=INK, lw=0.7, hatch=GAP_HATCH, alpha=0.45))
                continue
            n = t["invocations"]
            if n == 0:
                ax.add_patch(plt.Rectangle((x - 0.42, y - 0.38), 0.84, 0.76,
                                           facecolor=EMPTY_FILL, edgecolor=INK, lw=0.7))
                ax.text(x, y, "0", ha="center", va="center", fontsize=8, color=MAUVE)
            else:
                ax.add_patch(plt.Rectangle((x - 0.42, y - 0.38), 0.84, 0.76,
                                           facecolor=SAGE, edgecolor=INK, lw=0.7,
                                           alpha=min(0.35 + 0.09 * n, 1.0)))
                ax.text(x, y, str(n), ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8.4, rotation=28, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{f['target'][:8]} {f['arm']} {f['run_id'][:24]}" for f in rows], fontsize=7.6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    style_ax(ax, grid="")
    ax.legend(handles=[Patch(facecolor=SAGE, edgecolor=INK, label="invoked (count shown)"),
                       Patch(facecolor=EMPTY_FILL, edgecolor=INK, label="granted, never invoked"),
                       Patch(facecolor="none", edgecolor=INK, hatch=GAP_HATCH,
                             label="path grant — use is invisible"),
                       Patch(facecolor=PAGE_BG, edgecolor=INK, lw=0.5, label="blank = not granted to this arm")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.4)
    # Which rungs left every brokered tool untouched is DERIVED, not asserted: it is the sentence a
    # reader will quote, so it has to move when the data does.
    unused = []
    for f in rows:
        brokered = [t for t in f["granted_tools"]
                    if t["invocable"] and t["name"] not in SUBSTRATE_NAMES]
        if brokered and all(t["invocations"] == 0 for t in brokered):
            unused.append(f"{f['target']} {f['arm']}")
    tail = (" Rungs that invoked none of the brokered tools their own arm provides: "
            + "; ".join(unused) + ".") if unused else ""
    _figcaption(fig, "Counts come from matching staged tool filenames in each run's own command "
                     "text, so they are a LOWER BOUND: a tool imported inside a script the agent "
                     "wrote never appears on a command line. A zero on a brokered tool still means "
                     "its shim was not invoked." + tail, y=0.02)
    suptitle(fig, "Granted is not used", y=1.0)
    fig.subplots_adjust(bottom=0.30, top=0.90, right=0.78)
    _save(fig, out, "fig11_granted_vs_used")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from merlin.common.paths import artifacts_dir
    ap.add_argument("--facts", type=Path, default=artifacts_dir() / "agentic-report" / "run_facts.json")
    ap.add_argument("--out", type=Path, default=artifacts_dir() / "agentic-report" / "figures")
    ap.add_argument("--only", action="append", default=[])
    a = ap.parse_args(argv)
    facts = json.loads(a.facts.read_text())
    use_merlin_style()
    # The house style paints a cream canvas repo-wide. Repaint to white for this kit only.
    plt.rcParams.update({"axes.facecolor": PAGE_BG, "figure.facecolor": PAGE_BG,
                         "savefig.facecolor": PAGE_BG})
    names = a.only or list(_REGISTRY)
    for name in names:
        fn = _REGISTRY.get(name)
        if fn is None:
            print(f"  [skip] unknown figure {name!r}; known: {sorted(_REGISTRY)}", file=sys.stderr)
            continue
        fn(facts, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
