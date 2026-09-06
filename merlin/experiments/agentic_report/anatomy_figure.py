#!/usr/bin/env python3
"""One phase-1 run, in full — the anatomy of a capsule-bench experiment.

    anatomy_figure.py --anatomy anatomy_<run>.json [--out DIR]

Five stacked views on one shared clock, because the question this answers is how they line up:

1. what the agent had passing, and which capsules turned green when;
2. where its wall clock went, split into DEVELOPMENT and FEEDBACK;
3. every tool call as an event, so bursts and stalls are visible rather than averaged;
4. input against output token rate, which move for different reasons;
5. what it had spent.
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
import numpy as np                                                       # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

from merlin.agentreport.anatomy import CATEGORIES                        # noqa: E402
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


def render(a: dict, out: Path) -> Path:
    wall_min = a["wall_s"] / 60.0
    calls = a["calls"]
    use_merlin_style()
    plt.rcParams.update({"axes.facecolor": PAGE_BG, "figure.facecolor": PAGE_BG,
                         "savefig.facecolor": PAGE_BG})

    fig = plt.figure(figsize=(15.5, 15.0))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.1, 1.5, 1.2, 1.2, 1.0], hspace=0.34,
                          left=0.085, right=0.90, top=0.935, bottom=0.085)
    ax_caps, ax_band, ax_rug, ax_tok, ax_cost = (fig.add_subplot(gs[i]) for i in range(5))

    # ---------------------------------------------------------------- 1. capsules
    verdicts = a["verdicts"]
    v_max = max((v["t_s"] for v in verdicts), default=0) / 60.0
    if verdicts:
        total = max(v["n_capsules"] for v in verdicts)
        names: list[str] = []
        for v in verdicts:                       # order capsules by when they first passed
            for cap, st in sorted(v["per_capsule"].items()):
                if st == "pass" and cap not in names:
                    names.append(cap)
        never = sorted({c for v in verdicts for c in v["per_capsule"]} - set(names))
        order = names + never
        grid = np.zeros((len(order), len(verdicts)))
        for j, v in enumerate(verdicts):
            for i, cap in enumerate(order):
                st = v["per_capsule"].get(cap, "")
                grid[i, j] = {"pass": 3.0, "fail": 2.0, "incomplete": 1.0}.get(st, 0.0)
        from matplotlib.colors import ListedColormap
        # The last verdict has no successor, so its column is drawn one median grade-interval wide
        # rather than stretched to the end of the run: a block that wide would imply the grader kept
        # confirming that state, which it did not.
        gaps = [verdicts[i + 1]["t_s"] - verdicts[i]["t_s"] for i in range(len(verdicts) - 1)]
        tail = (sorted(gaps)[len(gaps) // 2] if gaps else 60.0) / 60.0
        edges = [v["t_s"] / 60.0 for v in verdicts] + [verdicts[-1]["t_s"] / 60.0 + tail]
        ax_caps.pcolormesh(edges, np.arange(len(order) + 1), grid, shading="flat",
                           cmap=ListedColormap(["#EFEAE4", "#D8C7BE", MAUVE, SAGE]),
                           vmin=0, vmax=3, alpha=0.85)
        ax_caps.set_ylabel(f"each of the {len(order)} capsules,\nordered by when it first passed")
        ax_caps.set_ylim(0, len(order))
        ax_caps.set_yticks([])
        # The count is read off the block boundary itself, so no second axis is needed and the two
        # cannot disagree. Annotate only where the count CHANGES.
        prev = None
        for v in verdicts:
            if v["n_passed"] == prev:
                continue
            prev = v["n_passed"]
            ax_caps.plot([v["t_s"] / 60.0], [v["n_passed"]], "o", color=INK, ms=5, zorder=5)
            ax_caps.text(v["t_s"] / 60.0 + tail * 0.12, v["n_passed"] + len(order) * 0.02,
                         f"{v['n_passed']}/{total}", fontsize=10, color=INK, fontweight="bold",
                         zorder=5)
        step_x = [v["t_s"] / 60.0 for v in verdicts] + [edges[-1]]
        step_y = [v["n_passed"] for v in verdicts] + [verdicts[-1]["n_passed"]]
        ax_caps.step(step_x, step_y, where="post", color=INK, lw=1.6, zorder=4)
        style_ax(ax_caps, grid="")
        n_fail = sum(1 for st in verdicts[-1]["per_capsule"].values() if st == "fail")
        n_inc = sum(1 for st in verdicts[-1]["per_capsule"].values() if st == "incomplete")
        title(ax_caps, f"What was passing, and which capsules turned green when — "
                       f"{verdicts[-1]['n_passed']} passing, {n_fail} failing, "
                       f"{n_inc} incomplete at the last grade", fs=12.5)
        ax_caps.legend(handles=[Patch(facecolor=SAGE, alpha=0.85, label="passing"),
                                Patch(facecolor=MAUVE, alpha=0.85, label="failing"),
                                Patch(facecolor="#D8C7BE", label="incomplete"),
                                Patch(facecolor="#EFEAE4", label="not in this grade")],
                       loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=8.4)

    # ---------------------------------------------------------------- 2. activity band
    centres, acc, width = _bins(calls, a["wall_s"])
    xs = centres / 60.0
    dev = [k for k in CATEGORIES if not CATEGORIES[k][1]]
    fb = [k for k in CATEGORIES if CATEGORIES[k][1]]
    stack = [np.minimum(acc[k] / width, 1.0) for k in dev + fb]
    ax_band.stackplot(xs, *stack, colors=[CAT_COLOUR[k] for k in dev + fb],
                      labels=[CATEGORIES[k][0] for k in dev + fb], alpha=0.92, lw=0)
    occupied = np.clip(sum(stack), 0, 1)
    ax_band.fill_between(xs, occupied, 1.0, color="none", edgecolor=INK, lw=0.0,
                         hatch="//", alpha=0.28)
    ax_band.set_ylim(0, 1)
    ax_band.set_ylabel("share of each minute")
    style_ax(ax_band, grid="")
    dev_s = sum(c["duration_s"] for c in calls if not CATEGORIES[c["category"]][1])
    fb_s = sum(c["duration_s"] for c in calls if CATEGORIES[c["category"]][1])
    title(ax_band, f"Where the clock went — feedback {fb_s / 3600:.1f} h vs development "
                   f"{dev_s / 3600:.1f} h; the hatched remainder is the agent thinking", fs=12.5)
    ax_band.legend(loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=8.4, frameon=True)

    # ---------------------------------------------------------------- 3. every call
    order_rug = list(CATEGORIES)
    for i, k in enumerate(order_rug):
        pts = [c for c in calls if c["category"] == k]
        if not pts:
            continue
        ax_rug.scatter([c["t_s"] / 60.0 for c in pts], np.full(len(pts), i),
                       s=[6 + min(c["duration_s"], 400) * 0.6 for c in pts],
                       color=CAT_COLOUR[k], edgecolor=INK, lw=0.35, alpha=0.75, zorder=3)
    ax_rug.set_yticks(range(len(order_rug)))
    ax_rug.set_yticklabels([f"{CATEGORIES[k][0]}  ({sum(1 for c in calls if c['category'] == k)})"
                            for k in order_rug], fontsize=8.4)
    ax_rug.set_ylim(-0.7, len(order_rug) - 0.3)
    style_ax(ax_rug, grid="x")
    title(ax_rug, "Every tool call, sized by how long it took", fs=12.5)

    # ---------------------------------------------------------------- 4. token rate
    curve = a.get("token_curve") or []
    if len(curve) >= 3:
        for key, colour, label in (("input", SLATE, "fresh input"), ("output", NAVY, "output"),
                                   ("cache_read", GOLD, "cached input")):
            ts, vs = [], []
            for i in range(1, len(curve)):
                lo = max(0, i - 2)
                dt = (curve[i]["t_s"] - curve[lo]["t_s"]) / 60.0
                if dt <= 0:
                    continue
                ts.append(curve[i]["t_s"] / 60.0)
                vs.append(max(curve[i][key] - curve[lo][key], 0) / dt)
            if ts:
                ax_tok.plot(ts, [max(v, 1e-1) for v in vs], color=colour, lw=1.9, label=label,
                            marker="o", ms=3.2)
        ax_tok.set_yscale("log")
    ax_tok.set_ylabel("tokens / min (log)")
    style_ax(ax_tok, grid="both")
    ax_tok.legend(loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=8.4)
    title(ax_tok, "Token rate — cached input, fresh input and output move for different reasons",
          fs=12.5)

    # ---------------------------------------------------------------- 5. spend
    cost = a.get("cost_curve") or []
    if len(cost) >= 2:
        ax_cost.fill_between([p["t_s"] / 60.0 for p in cost], 0, [p["usd"] for p in cost],
                             color=SAGE, alpha=0.35, lw=0)
        ax_cost.plot([p["t_s"] / 60.0 for p in cost], [p["usd"] for p in cost], color=SAGE, lw=2.0)
        ax_cost.text(cost[-1]["t_s"] / 60.0, cost[-1]["usd"], f"  ${cost[-1]['usd']:,.0f}",
                     fontsize=9.5, color=SAGE, fontweight="bold", va="center")
        ax_cost.set_ylabel("cumulative USD")
    else:
        ax_cost.text(0.5, 0.5, "no priced curve for this run", transform=ax_cost.transAxes,
                     ha="center", va="center", fontsize=9, color=MAUVE)
    style_ax(ax_cost, grid="both")
    title(ax_cost, "Cumulative spend, priced per token bucket", fs=12.5)
    ax_cost.set_xlabel("Time (min)")

    span = max(wall_min, v_max) * 1.02
    for ax in (ax_band, ax_rug, ax_tok, ax_cost):
        ax.set_xlim(0, span)
    if verdicts:
        ax_caps.set_xlim(0, span)
        for ax in (ax_band, ax_rug, ax_tok, ax_cost):
            for v in verdicts:
                ax.axvline(v["t_s"] / 60.0, color=INK, lw=0.7, ls=(0, (2, 4)), alpha=0.35, zorder=1)

    counts = Counter(c["category"] for c in calls)
    head = (f"{a['target']} · {a['arm']} · {a['run_id']}   —   {a['model']}   ·   "
            f"{wall_min / 60:.1f} h   ·   {len(calls)} tool calls   ·   "
            f"{counts['selfcheck']} self-checks   ·   {len(verdicts)} grades")
    fig.text(0.085, 0.955, head, fontsize=10.5, color=INK)
    suptitle(fig, "Anatomy of a phase-1 run", y=0.985)

    note = (a.get("notes") or [""])[0]
    fig.text(0.012, 0.012,
             "Panel 1's clock is the grader's; panels 2-5 are the transcript's. Both start when the "
             "run does. Feedback is the self-check and the oracle — the agent cannot answer those "
             "itself and waits; development is everything it does under its own power. "
             + (note + " " if note else "")
             + "Call durations come from paired arrival stamps; an authoring edit completes inside "
               "one stamp and so registers as an event rather than a duration, which is why "
               "authoring is 46 calls and about a second.",
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
    a = ap.parse_args(argv)
    path = render(json.loads(a.anatomy.read_text()), a.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
