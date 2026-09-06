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

    fig = plt.figure(figsize=(15.5, 10.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.9, 2.1, 1.3], hspace=0.46,
                          left=0.085, right=0.885, top=0.856, bottom=0.125)
    ax_caps, ax_band, ax_tok = (fig.add_subplot(gs[i]) for i in range(3))

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
            near_end = v["t_s"] / 60.0 > 0.82 * (edges[-1] or 1.0)
            ax_caps.text(v["t_s"] / 60.0 + (-tail * 0.12 if near_end else tail * 0.12),
                         v["n_passed"] + len(order) * 0.02, f"{v['n_passed']}/{total}",
                         fontsize=10, color=INK, fontweight="bold", zorder=5,
                         ha="right" if near_end else "left")
        step_x = [v["t_s"] / 60.0 for v in verdicts] + [edges[-1]]
        step_y = [v["n_passed"] for v in verdicts] + [verdicts[-1]["n_passed"]]
        ax_caps.step(step_x, step_y, where="post", color=INK, lw=1.6, zorder=4)
        style_ax(ax_caps, grid="")
        n_fail = sum(1 for st in verdicts[-1]["per_capsule"].values() if st == "fail")
        n_inc = sum(1 for st in verdicts[-1]["per_capsule"].values() if st == "incomplete")
        title(ax_caps, f"Tests passing over time — each row is one test, ordered by when it first "
                       f"passed. Ended at {verdicts[-1]['n_passed']} of {total}", fs=12.5, pad=16)
        ax_caps.legend(handles=[Patch(facecolor=SAGE, alpha=0.85, label="passing"),
                                Patch(facecolor=MAUVE, alpha=0.85, label="failing"),
                                Patch(facecolor="#D8C7BE", label="could not be judged"),
                                Patch(facecolor="#EFEAE4", label="not graded yet")],
                       loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=8.4)

    # ---------------------------------------------------------------- 2. activity band
    centres, acc, width = _bins(calls, a["wall_s"])
    xs = centres / 60.0
    dev = [k for k in CATEGORIES if not CATEGORIES[k][1]]
    fb = [k for k in CATEGORIES if CATEGORIES[k][1]]
    stack = [np.minimum(acc[k] / width, 1.0) for k in dev + fb]
    whole = Counter()
    whole_s = Counter()
    for c in calls_all:
        whole[c["category"]] += 1
        whole_s[c["category"]] += c["duration_s"]
    def _lab(k):
        sec = whole_s[k]
        amount = f"{sec / 3600:.1f} h" if sec >= 3600 else (f"{sec:.0f} s" if sec >= 1 else "<1 s")
        return f"{CATEGORIES[k][0]} — {whole[k]} calls, {amount}"
    ax_band.stackplot(xs, *stack, colors=[CAT_COLOUR[k] for k in dev + fb],
                      labels=[_lab(k) for k in dev + fb], alpha=0.92, lw=0)
    occupied = np.clip(sum(stack), 0, 1)
    ax_band.fill_between(xs, occupied, 1.0, color="none", edgecolor=INK, lw=0.0,
                         hatch="//", alpha=0.28)
    ax_band.set_ylabel("share of each minute")
    style_ax(ax_band, grid="")
    dev_s = sum(c["duration_s"] for c in calls_all if not CATEGORIES[c["category"]][1])
    fb_s = sum(c["duration_s"] for c in calls_all if CATEGORIES[c["category"]][1])
    title(ax_band, f"Where the time went — waiting for feedback {fb_s / 3600:.1f} h against doing "
                   f"the work {dev_s / 3600:.1f} h. Hatched = the agent thinking. Each dot above is "
                   f"one tool call, sized by how long it took", fs=12.5, pad=28)
    ax_band.legend(loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=8.0, frameon=True,
                   title="what the agent did (whole run)", title_fontsize=8.4)

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
                        color=CAT_COLOUR[k], edgecolor=INK, lw=0.3, alpha=0.8, zorder=5)
    ax_band.set_ylim(0, strip_hi + step * 0.6)
    ax_band.set_yticks([0, 0.5, 1.0])
    ax_band.set_yticklabels(["0", "50%", "100%"])
    ax_band.axhline(1.02, color=INK, lw=0.6, alpha=0.30)

    # ---------------------------------------------------------------- 4. token rate
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
                ax_tok.plot(ts, [max(v, 1e-1) for v in vs], color=colour, lw=1.9, label=label,
                            marker="o", ms=3.2)
        ax_tok.set_yscale("log")
    ax_tok.set_ylabel("tokens per minute\n(log scale)")
    style_ax(ax_tok, grid="both")
    ax_tok.legend(loc="upper left", bbox_to_anchor=(1.085, 1.02), fontsize=8.4)
    title(ax_tok, "How fast tokens moved — re-read context, new input and generated output, with "
                  "money spent on the right", fs=12.5)

    # Spend rides the same panel on its own axis: it is the integral of these rates against price,
    # so putting it anywhere else asks the reader to hold two pictures at once.
    cost = a.get("cost_curve") or []
    if len(cost) >= 2:
        ax_cost = ax_tok.twinx()
        ax_cost.fill_between([p["t_s"] / 60.0 for p in cost], 0, [p["usd"] for p in cost],
                             color=SAGE, alpha=0.20, lw=0, zorder=1)
        ax_cost.plot([p["t_s"] / 60.0 for p in cost], [p["usd"] for p in cost], color=SAGE,
                     lw=2.0, zorder=2)
        ax_cost.text(0.015, 0.93, f"${cost[-1]['usd']:,.0f} spent by this point",
                     transform=ax_cost.transAxes, ha="left", va="top",
                     fontsize=9.5, color=SAGE, fontweight="bold")
        ax_cost.set_ylabel("cumulative USD", color=SAGE, labelpad=1)
        ax_cost.tick_params(labelcolor=SAGE)
        ax_cost.set_ylim(0, max(p["usd"] for p in cost) * 1.25)
        ax_cost.spines["top"].set_visible(False)
    ax_tok.set_xlabel("Time (min)")

    for ax in (ax_caps, ax_band, ax_tok):
        ax.set_xlim(0, span)
        for v in verdicts:
            ax.axvline(v["t_s"] / 60.0, color=INK, lw=0.7, ls=(0, (2, 4)), alpha=0.30, zorder=1)
    # The moment the score stopped moving, marked on every panel — it is the line the other two
    # panels have to be read against.
    if best_at is not None and best_at <= span:
        for ax in (ax_caps, ax_band, ax_tok):
            ax.axvline(best_at, color=GOLD, lw=2.0, ls=(0, (5, 3)), zorder=6)
        ax_caps.text(best_at, len(order) * 1.01 if verdicts else 1.0,
                     f" best score reached at {best_at:.0f} min", color=GOLD, fontsize=9.5,
                     fontweight="bold", va="bottom")

    tail_calls = [c for c in calls_all if best_at is not None and c["t_s"] > best_at * 60]
    tail_s = sum(c["duration_s"] for c in tail_calls)
    total_s = sum(c["duration_s"] for c in calls_all) or 1.0
    counts = Counter(c["category"] for c in calls_all)
    head = (f"{a['target']} · {a['arm']} · {a['run_id']}   —   {a['model']}   ·   "
            f"whole run {wall_min / 60:.1f} h, {len(calls_all)} tool calls, "
            f"{counts['selfcheck']} self-checks, {len(verdicts_all)} grades   ·   "
            f"shown: the first {span:.0f} min, to where the score stopped moving")
    fig.text(0.085, 0.958, head, fontsize=10.2, color=INK)
    fig.text(0.085, 0.936, "\n".join(textwrap.wrap(
             "Phase 1: an AI agent is given a chip it has never seen — its RTL and its instruction "
             "set — and has to write a working compiler for it. A test (\u201ccapsule\u201d) is one "
             "operation with its shapes and a correct answer the agent never sees. It checks its own "
             "work with a redacted self-check, and can queue simulations of the real hardware.",
             width=196)), fontsize=9.2, color=INK, alpha=0.85, va="top", linespacing=1.5)
    suptitle(fig, "Anatomy of a phase-1 run: an agent writing a compiler for unseen hardware",
             y=0.992)

    blocked = a.get("blocked") or []
    if tail_calls:
        deep = [b for b in blocked if b.get("deepest_tier_passed")]
        # Lead with WHY the score stopped, not with how long the agent kept going. "kept working
        # without improving" reads as a failure of persistence; these capsules could not move.
        if blocked:
            planes = "; ".join(sorted({PLANE_PLAIN.get(b["plane"], b["plane"] or b["category"]
                                                        or "unknown") for b in blocked}))
            summary = (f"the remaining {len(blocked)} test(s) could not be made to pass — they turn "
                       f"on {planes}"
                       + (f", and {len(deep)} of them ALREADY pass on simulated hardware"
                          if deep else "")
                       + f". The run spent a further {(wall_min - (best_at or 0)) / 60:.1f} h and "
                         f"{len(tail_calls)} calls on them")
        else:
            summary = (f"the run continued {(wall_min - (best_at or 0)) / 60:.1f} h and "
                       f"{len(tail_calls)} more calls past this line without moving the score")
        ax_band.text(0.004, -0.10, "\n".join(textwrap.wrap(summary, width=150)),
                     transform=ax_band.transAxes, ha="left", va="top",
                     fontsize=8.8, color=GOLD, fontweight="bold")
    note = (a.get("notes") or [""])[0]
    fig.text(0.012, 0.012,
             "The top panel is timed by the grader, the lower two by the agent's own transcript; "
             "both start when the run does. \u201cFeedback\u201d is the self-check and the hardware "
             "simulator — the agent cannot answer those itself and must wait; \u201cdoing the "
             "work\u201d is everything it does under its own power. " + (note + " " if note else "")
             + f"Call durations come from paired arrival stamps, so an authoring edit that "
               f"completes inside one stamp registers as an event rather than a duration — which "
               f"is why authoring is {whole['author']} calls and under a second in total.",
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
