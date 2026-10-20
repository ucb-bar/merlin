#!/usr/bin/env python3
"""Three editorial Phase 1 plates from the public normalized exploration evidence.

Run with matplotlib, numpy and Pillow installed (fontTools for ``--font-dir``)::

  python build_tools/plots/plot_phase1_editorial.py --data DATA_JSON --out OUTPUT_DIR

The input is the existing ``merlin.phase1.paper_exploration.v1`` projection. No
experiment, simulator, provider or private capture is opened. Figures preserve
observed wall intervals and snapshot evidence; no CPU residency, finer progress,
complete billing or causal tool effect is inferred. The physical figure width is
7.2 inches, so text is designed at the intended two-column paper width.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import shlex
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors, font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STYLE_PATH = REPO / "packages/merlin-analysis/src/merlin/plotting/merlin_plotstyle.py"
if not STYLE_PATH.is_file():
    STYLE_PATH = Path(__file__).with_name("merlin_plotstyle.py")
spec = importlib.util.spec_from_file_location("merlin_plotstyle", STYLE_PATH)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.use_merlin_style()

SERIF, SANS, MONO = "DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono"
for family in (SERIF, SANS, MONO):
    font_manager.findfont(family, fallback_to_default=False)
plt.rcParams.update(
    {
        "font.family": SANS,
        "font.size": 7.7,
        "text.color": style.INK,
        "axes.labelcolor": style.INK,
        "axes.edgecolor": style.GOLD,
        "xtick.color": style.INK,
        "ytick.color": style.INK,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
NAVY, BLUE, SLATE = style.NAVY, style.BLUE, style.SLATE
GOLD, SAGE, MAUVE, INK, CREAM = style.GOLD, style.SAGE, style.MAUVE, style.INK, style.BG
PALE = "#F1EEE9"
STAGES = {"Compiler package": GOLD, "Build / link": SLATE, "L2 Spike": NAVY, "L3 gSIM": MAUVE}
STATES = {"Execution + trace": NAVY, "Numeric only": GOLD, "Refusal met": SAGE, "Failed / incomplete": MAUVE}
EXAMPLE = "SY_conv_operand_capacity_boundary"


def select_fonts(font_dir, out):
    """Register a local licensed font bundle; otherwise use Matplotlib's bundled fonts."""
    global SERIF, SANS
    if font_dir is None:
        return None
    import fontTools
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    assets = out / "fonts"
    shutil.copytree(font_dir, assets)
    registered = []
    for path in sorted(assets.glob("*.ttf")):
        font = TTFont(path)
        if "fvar" in font:
            defaults = {axis.axisTag: axis.defaultValue for axis in font["fvar"].axes}
            for weight in (400, 700):
                limits = {**defaults, "wght": weight} if "wght" in defaults else defaults
                fixed = instantiateVariableFont(font, limits, inplace=False, updateFontNames=True)
                destination = assets / f"{path.stem}-static-{weight}.ttf"
                fixed.save(destination)
                registered.append(destination)
        else:
            registered.append(path)
    for path in registered:
        font_manager.fontManager.addfont(path)
    SERIF, SANS = style.SERIF, style.SANS
    for family in (SERIF, SANS):
        font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams["font.family"] = SANS
    return {
        "fonttools_version": fontTools.__version__,
        "files": {str(path.relative_to(out)): digest(path) for path in assets.rglob("*") if path.is_file()},
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(row):
    if row["status"] == "pass" and row["numeric_status"] == "pass" and row["tiers"].get("L3") == "pass":
        return "Execution + trace" if row["trace_requirement_status"] == "pass" else "Numeric only"
    return "Refusal met" if row["status"] == "pass" else "Failed / incomplete"


def occupancy(intervals, edges, *, union=False):
    merged = []
    for a, b in sorted(intervals):
        if a is None or b is None or b <= a:
            continue
        if union and merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    values = np.zeros(len(edges) - 1)
    for a, b in merged:
        values += np.maximum(0, np.minimum(edges[1:], b) - np.maximum(edges[:-1], a))
    return values / np.diff(edges)


class Plate:
    """A virtual-inch editorial grid rendered at 7.2 physical inches."""

    def __init__(self, title, subtitle, *, height=10.0):
        self.w, self.h = 14.4, height
        self.fig = plt.figure(figsize=(7.2, height / 2))
        self.text(0.38, 0.26, "MERLIN  /  PHASE 1  /  EL4", size=7.0, color=NAVY, weight="bold")
        self.text(0.38, 0.72, title, size=16.4, serif=True, weight="bold")
        self.text(0.38, 1.02, subtitle, size=7.6, color=INK)
        self.line(0.38, 1.14, 14.0, 1.14, color=GOLD, lw=0.6)

    def text(
        self,
        x,
        y,
        value,
        *,
        size=7.7,
        color=INK,
        weight="normal",
        serif=False,
        mono=False,
        ha="left",
        va="baseline",
        **kwargs,
    ):
        if "\n" in value and va == "baseline":
            va = "top"
        if serif and SERIF == style.SERIF:
            weight = "normal"  # DM Serif Display is an intentional single-weight display face.
        return self.fig.text(
            x / self.w,
            1 - y / self.h,
            value,
            fontsize=size,
            fontfamily=MONO if mono else SERIF if serif else SANS,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            linespacing=1.2,
            **kwargs,
        )

    def rect(self, x, y, w, h, *, color=CREAM, edge="none", lw=0.5, alpha=1):
        patch = Rectangle(
            (x / self.w, 1 - (y + h) / self.h),
            w / self.w,
            h / self.h,
            transform=self.fig.transFigure,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            zorder=-10,
        )
        self.fig.patches.append(patch)

    def line(self, x1, y1, x2, y2, *, color=GOLD, lw=0.6, ls="-", **kwargs):
        self.fig.lines.append(
            Line2D(
                [x1 / self.w, x2 / self.w],
                [1 - y1 / self.h, 1 - y2 / self.h],
                transform=self.fig.transFigure,
                color=color,
                lw=lw,
                ls=ls,
                **kwargs,
            )
        )

    def arrow(self, x1, y1, x2, y2, *, color=NAVY, lw=0.9, style_="-|>"):
        self.fig.patches.append(
            FancyArrowPatch(
                (x1 / self.w, 1 - y1 / self.h),
                (x2 / self.w, 1 - y2 / self.h),
                transform=self.fig.transFigure,
                arrowstyle=style_,
                mutation_scale=8,
                color=color,
                linewidth=lw,
            )
        )

    def heading(self, x, y, number, title, *, color=NAVY, size=10.4):
        self.text(
            x + 0.13,
            y,
            str(number),
            size=8,
            color="white",
            ha="center",
            va="center",
            weight="bold",
            bbox={"boxstyle": "circle,pad=.20", "fc": color, "ec": "none"},
        )
        self.text(x + 0.43, y + 0.075, title, size=size, serif=True, weight="bold", color=BLUE)

    def axes(self, x, y, w, h, *, clock=False, duration=240):
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h])
        ax.tick_params(labelsize=6.9, length=2, width=0.5, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        if clock:
            ax.set_xlim(0, duration)
            ax.set_xticks(np.arange(0, duration + 1, 30))
            ax.grid(axis="x", color=GOLD, alpha=0.23, linewidth=0.45)
            ax.set_axisbelow(True)
        return ax

    def footer(self, value):
        self.line(0.38, self.h - 0.53, 14.0, self.h - 0.53, lw=0.5)
        self.text(0.38, self.h - 0.27, value, size=6.5, color=MAUVE)

    def save(self, out, name):
        for ext in ("png", "pdf", "svg"):
            self.fig.savefig(
                out / f"{name}.{ext}", dpi=320, metadata={"Creator": "Merlin Phase 1 public evidence plates"}
            )
        plt.close(self.fig)


def workflow(p, *, y=1.40):
    cards = [
        (0.38, 3.0, "Selected inputs", "RTL facts + target contract\n153 public capsules", NAVY),
        (3.82, 3.0, "Parallel authoring", "One lead + six helpers\nAn initially empty compiler", BLUE),
        (7.28, 3.0, "Independent grading", "L0 semantics → L1 audit\nL2 Spike → L3 gSIM", MAUVE),
        (10.74, 3.26, "Retained evidence", "Hashed artifacts + verdicts\nFull-model gates stay separate", SAGE),
    ]
    for i, (x, w, title, body, color) in enumerate(cards):
        p.rect(x, y, w, 0.92, color=CREAM)
        p.line(x, y, x, y + 0.92, color=color, lw=2)
        p.text(x + 0.15, y + 0.26, title, size=8.6, weight="bold", color=color)
        p.text(x + 0.15, y + 0.57, body, size=7.4, va="center")
        if i < len(cards) - 1:
            p.arrow(x + w + 0.04, y + 0.45, cards[i + 1][0] - 0.07, y + 0.45, color=color)


def run_story(d, out):
    p = Plate(
        "The making of an accelerator compiler",
        "A four-hour campaign · 153 public capsules · seven author actors",
        height=10.4,
    )
    workflow(p)
    p.heading(0.38, 2.68, 1, "Author work and capsule evidence share one clock")
    ax = p.axes(0.92, 3.05, 9.3, 2.30, clock=True)
    T = d["duration"]
    a, b = d["lead_span"]
    edges = np.arange(0, T + 2, 2)
    centers = (edges[1:] + edges[:-1]) / 2
    paired = [c for c in d["tools"] if c["completed"] and c["b"] is not None]
    observed = occupancy([(a, b)], edges, union=True)
    share = occupancy([(c["a"], c["b"]) for c in paired], edges, union=True)
    share = np.divide(share, observed, out=np.full_like(share, np.nan), where=observed > 0)
    share = np.clip(share, 0, 1)
    ax.fill_between(centers, 0, share, step="mid", color=SLATE, alpha=0.88)
    ax.fill_between(centers, share, 1, where=np.isfinite(share), step="mid", color=BLUE, alpha=0.16)
    ax.axvspan(b, T, color=CREAM, zorder=-2)
    ax.axvline(b, color=GOLD, lw=0.8, ls=(0, (3, 3)))
    ax.set_ylim(0, 1.11)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_ylabel("Lead wall share", fontsize=7.3, labelpad=3)
    ax.text(a + 2, 1.055, "Authoring", color=BLUE, fontsize=7.1)
    ax.text(b + 3, 1.055, "Post-author validation", color=MAUVE, fontsize=7.1)
    ax.text(48, 0.48, "Whole-cohort progress\nis unknown here", ha="center", fontsize=7.6, color=BLUE)
    gs = d["grades"]
    ys = [g["summary"]["execution_passes_with_trace_evidence"] / 153 for g in gs]
    ax.step([g["minute"] for g in gs] + [T], ys + [ys[-1]], where="post", color=NAVY, lw=1.8)
    ax.scatter([g["minute"] for g in gs], ys, c=NAVY, s=18, zorder=5)
    for g, value in zip(gs, ys):
        ax.annotate(
            str(round(value * 153)),
            (g["minute"], value),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.6,
            weight="bold",
            color=NAVY,
        )
        ax.axvline(g["minute"], color=NAVY, lw=0.5, ls=":", alpha=0.45)
    right = ax.secondary_yaxis("right", functions=(lambda x: x * 153, lambda x: x / 153))
    right.set_yticks([0, 86, 140, 153])
    right.tick_params(labelsize=6.6, length=2, pad=2)
    p.text(10.13, 2.93, "capsules", size=6.6, color=NAVY, ha="right")
    ax.set_xticklabels([])
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fc=SLATE, label="Paired tool interval"),
            Rectangle((0, 0), 1, 1, fc=colors.to_rgba(BLUE, 0.16), label="Client / model / network gap"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=6.6,
        handlelength=1.2,
    )

    p.line(10.65, 2.65, 10.65, 7.25, color=GOLD, lw=0.55)
    p.text(11.0, 2.94, "At the last snapshot", size=8.4, color=BLUE, weight="bold")
    p.text(11.0, 3.65, "140", size=31, color=NAVY, weight="bold", serif=True)
    p.text(12.55, 3.65, "/ 153", size=13, color=GOLD, serif=True)
    p.text(11.0, 3.91, "capsules with execution\n+ required trace evidence", size=8.0)
    p.text(11.0, 4.75, "141 result passes", size=9, weight="bold")
    p.text(11.0, 5.05, "includes one expected refusal", size=7.5, color=SAGE)
    p.rect(10.95, 5.35, 3.05, 1.15, color=CREAM)
    p.text(11.1, 5.65, "12 unresolved cases", size=8.3, color=MAUVE, weight="bold")
    p.text(11.1, 5.88, "Six public model/program\nchecks remained unqualified.", size=7.3)
    p.text(11.0, 6.85, "Formal Phase 1 close:\nnot achieved in this run", size=8, color=MAUVE, weight="bold")

    p.heading(0.38, 5.77, 2, "The token stream is observed response traffic")
    tx = p.axes(0.92, 6.08, 9.3, 1.08, clock=True)
    token = d["tokens"]
    hist = {
        key: np.histogram([r["minute"] for r in token], bins=edges, weights=[r[key] for r in token])[0] / 2
        for key in ("fresh", "cached", "output", "reasoning")
    }
    tx.stackplot(centers, hist["cached"] / 1e6, hist["fresh"] / 1e6, colors=[SLATE, NAVY], step="mid", alpha=0.9)
    tx.set_yticks([0, 1, 2])
    tx.set_ylim(0, 2.4)
    tx.set_ylabel("M input/min", fontsize=7.1, labelpad=4)
    outax = tx.twinx()
    outax.plot(centers, hist["output"] / 1e3, color=MAUVE, lw=1)
    outax.set_ylim(0, 15)
    outax.set_yticks([0, 7, 14])
    outax.tick_params(labelsize=6.5, length=2, pad=2)
    outax.set_ylabel("k output/min", fontsize=7.0, labelpad=4, color=MAUVE)
    tx.set_xlabel("Minutes since campaign start", fontsize=7.3, labelpad=0)
    tx.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fc=SLATE, label="Cached input"),
            Rectangle((0, 0), 1, 1, fc=NAVY, label="Fresh input"),
            Line2D([], [], color=MAUVE, label="All output"),
        ],
        fontsize=6.5,
        frameon=False,
        ncol=3,
        loc="upper right",
        handlelength=1.1,
    )

    p.line(0.38, 7.76, 14.0, 7.76, color=GOLD, lw=0.5)
    p.heading(0.38, 8.10, 3, "Feedback can overlap")
    q = p.axes(0.92, 8.43, 6.4, 0.87)
    for i, idx in enumerate((9, 10, 11)):
        r = d["requests"][idx]
        q.plot([r["a"], r["start"]], [i, i], color=GOLD, lw=4, solid_capstyle="butt")
        q.plot([r["start"], r["b"]], [i, i], color=NAVY, lw=4, solid_capstyle="butt")
        q.scatter([r["b"]], [i], s=10, c=MAUVE, zorder=4)
        q.text(r["b"] + 0.5, i, f"{r['passed']}/{r['rows']}", va="center", fontsize=7, color=NAVY)
    q.set_xlim(48, 78)
    q.set_ylim(2.7, -0.7)
    q.set_xticks([50, 60, 70])
    q.set_xlabel("Campaign minute", fontsize=6.7, labelpad=1)
    q.set_yticks(range(3), ["R10", "R11", "R12"])
    q.spines[["left", "right", "top"]].set_visible(False)
    p.text(7.8, 8.16, "19 self-check requests", size=9.2, color=NAVY, weight="bold")
    p.text(
        7.8,
        8.45,
        "Subset answers are not the 153-case score.\nA short check can return while a long check runs.",
        size=7.4,
    )
    p.text(7.8, 9.30, "70.16M observed tokens", size=9.2, color=BLUE, weight="bold")
    p.text(7.8, 9.64, "67.23M cached + 2.61M fresh + 0.320M output", size=7.1)
    p.footer(
        "Retained historical evidence only. Client gaps are not measured thinking; "
        "reasoning is part of output; final usage is incomplete."
    )
    p.save(out, "01_compiler_formation")


def family_counts(d):
    groups = collections.defaultdict(list)
    for row in d["grades"][-1]["capsules"]:
        groups[row["capsule"].split("_")[0]].append(row["capsule"])
    return dict(sorted(groups.items()))


def execution(d, out):
    p = Plate(
        "Where repeated work accumulates",
        "Public capsule identities connect build jobs, simulators and verdicts",
        height=11.3,
    )
    events = d["invocations"]
    total = sum(e["b"] - e["a"] for e in events)
    gsim = [e for e in events if e["category"] == "L3 gSIM"]
    gsim_time = sum(e["b"] - e["a"] for e in gsim)
    p.heading(0.38, 1.53, 1, "Public cohort")
    p.text(0.81, 1.96, "153 capsules", size=11, color=NAVY, weight="bold")
    groups = family_counts(d)
    for i, (family, names) in enumerate(groups.items()):
        x = 4.1 + i * 3.25
        p.text(x, 1.66, family, size=12, mono=True, color=BLUE, weight="bold")
        p.text(x + 0.8, 1.66, f"{len(names)} capsules", size=8.2)
        counts = collections.Counter(state(r) for r in d["grades"][-1]["capsules"] if r["capsule"] in names)
        left = x
        for category in STATES:
            width = 2.8 * counts[category] / len(names)
            p.rect(left, 1.87, width, 0.20, color=STATES[category])
            left += width
        if counts["Refusal met"] or counts["Failed / incomplete"]:
            caption = (
                f"{counts['Execution + trace']} trace · {counts['Refusal met']} refusal · "
                f"{counts['Failed / incomplete']} open"
            )
        else:
            caption = f"{counts['Execution + trace']} execution + trace"
        p.text(x, 2.33, caption, size=7.0, color=NAVY)
    p.text(
        0.81,
        2.65,
        "Final within-family fractions; prefixes are retained names. A verdict is not a fresh simulation.",
        size=7.1,
        color=GOLD,
    )
    p.line(0.38, 2.83, 14.0, 2.83, color=GOLD, lw=0.5)
    p.heading(0.38, 3.15, 2, "When each family launches work")
    heat = p.axes(1.82, 3.83, 8.35, 2.15, clock=True)
    edges = np.arange(0, 245, 5)
    histograms = {
        (family, stage): np.histogram(
            [e["a"] for e in events if e["capsule"] in names and e["category"] == stage], bins=edges
        )[0]
        for family, names in groups.items()
        for stage in STAGES
    }
    max_count = max(int(max(row)) for row in histograms.values())
    p.text(0.81, 3.52, "Starts / 5 min · common log-opacity scale", size=7.1)
    for i, n in enumerate((1, 10, 100, max_count)):
        x = 6.95 + i * 0.75
        p.rect(x, 3.35, 0.23, 0.18, color=NAVY, alpha=0.18 + 0.82 * math.log1p(n) / math.log1p(max_count))
        p.text(x + 0.29, 3.52, str(n), size=6.6)
    labels = []
    for fi, (family, names) in enumerate(groups.items()):
        for si, (stage, color) in enumerate(STAGES.items()):
            index = fi * 4 + si
            counts = histograms[family, stage]
            for j, n in enumerate(counts):
                if n:
                    alpha = 0.18 + 0.82 * math.log1p(n) / math.log1p(max_count)
                    heat.add_patch(
                        Rectangle((edges[j], index - 0.37), 5, 0.74, facecolor=color, alpha=alpha, edgecolor="none")
                    )
            labels.append(
                (family if si == 0 else "")
                + "   "
                + {"Compiler package": "package", "Build / link": "build", "L2 Spike": "Spike", "L3 gSIM": "gSIM"}[
                    stage
                ]
            )
        if fi:
            heat.axhline(fi * 4 - 0.5, color=GOLD, lw=0.6)
    heat.set_ylim(11.6, -0.7)
    heat.set_yticks(range(12), labels)
    heat.tick_params(axis="y", length=0, labelsize=7.1)
    heat.spines["left"].set_visible(False)
    heat.set_xlabel("Minutes since campaign start", fontsize=7.4, labelpad=3)
    p.text(10.68, 3.65, "Work is not core time", size=9.3, serif=True, color=BLUE, weight="bold")
    p.text(10.68, 4.16, "11,506", size=22, serif=True, color=NAVY, weight="bold")
    p.text(10.68, 4.48, "closed subprocess intervals", size=7.5)
    p.text(10.68, 5.08, f"{100 * gsim_time / total:.1f}%", size=20, serif=True, color=MAUVE, weight="bold")
    p.text(10.68, 5.28, "of retained process-wall time\nis in 447 gSIM executions", size=7.5)
    p.text(10.68, 5.98, "7,995 build jobs · 658 Spike runs", size=7.0, color=SLATE)
    p.line(0.38, 6.65, 14.0, 6.65, color=GOLD, lw=0.5)
    p.heading(0.38, 7.00, 3, "Zoom: one capacity-boundary capsule")
    p.text(0.81, 7.34, EXAMPLE, size=7.5, mono=True, color=BLUE)
    sub = sorted([e for e in gsim if e["capsule"] == EXAMPLE], key=lambda e: e["a"])
    zoom = p.axes(1.15, 7.66, 8.95, 1.48)
    for i, e in enumerate(sub):
        duration = (e["b"] - e["a"]) * 60
        zoom.plot([e["a"], e["b"]], [i, i], color=MAUVE, lw=7, solid_capstyle="butt")
        zoom.text((e["a"] + e["b"]) / 2, i, f"{duration:.0f}s", fontsize=7.1, color="white", ha="center", va="center")
        zoom.scatter([e["a"]], [i], s=9, color=MAUVE)
    zoom.set_xlim(48, 144)
    zoom.set_ylim(4.7, -0.7)
    zoom.set_yticks(range(5), [f"Attempt {i + 1}" for i in range(5)])
    zoom.set_xticks([50, 70, 90, 110, 130, 140])
    zoom.grid(axis="x", color=GOLD, alpha=0.25, lw=0.5)
    zoom.spines["left"].set_visible(False)
    zoom.set_xlabel("Campaign minute · actual gSIM process intervals", fontsize=7.2)
    zoom.tick_params(axis="y", length=0)
    p.text(10.68, 7.87, f"{sum(e['b'] - e['a'] for e in sub):.1f} min", size=20, serif=True, color=MAUVE, weight="bold")
    p.text(10.68, 8.22, "summed over five executions", size=7.5)
    p.text(
        10.68,
        8.64,
        f"{100 * sum(e['b'] - e['a'] for e in sub) / gsim_time:.1f}% of all gSIM wall time",
        size=7.9,
        weight="bold",
    )
    p.text(10.68, 8.88, "Same public capsule identity;\nseparate retained attempts.", size=7.4)
    spike = [e for e in events if e["capsule"] == EXAMPLE and e["category"] == "L2 Spike"]
    p.text(
        10.68,
        9.54,
        f"Spike: {len(spike)} runs, {sum(e['b'] - e['a'] for e in spike) * 60:.2f}s total",
        size=7.1,
        color=NAVY,
    )
    p.rect(0.8, 9.76, 13.2, 0.70, color=CREAM)
    p.text(1.0, 10.02, "Source interface", size=7.8, color=BLUE, weight="bold")
    p.arrow(3.0, 9.97, 3.55, 9.97)
    p.text(3.70, 10.02, "Compiler + link", size=7.8, color=GOLD, weight="bold")
    p.arrow(6.0, 9.97, 6.55, 9.97)
    p.text(6.70, 10.02, "Spike / gSIM", size=7.8, color=MAUVE, weight="bold")
    p.arrow(9.0, 9.97, 9.55, 9.97)
    p.text(9.70, 10.02, "Outputs + trace evidence", size=7.8, color=NAVY, weight="bold")
    p.text(
        1.0,
        10.33,
        "Conceptual evidence path; the heatmap counts starts, not CPU residency, memory service or array utilization.",
        size=6.9,
    )
    p.footer(
        "Process-wall intervals can overlap or nest. Coverage is the retained public jobs only; "
        "their sum is neither campaign time nor CPU time."
    )
    p.save(out, "02_execution_and_capsules")


def ribbons(ax, grades):
    states = list(STATES)
    lookups = [{r["capsule"]: state(r) for r in g["capsules"]} for g in grades]
    counts = [collections.Counter(x.values()) for x in lookups]
    gap = 3
    bottoms = []
    for count in counts:
        offset = 0
        positions = {}
        for s in states:
            positions[s] = offset
            offset += count[s] + gap
        bottoms.append(positions)
    for i in range(2):
        transitions = collections.Counter((lookups[i][n], lookups[i + 1][n]) for n in lookups[i])
        starts = bottoms[i].copy()
        ends = bottoms[i + 1].copy()
        for s in states:
            for t in states:
                n = transitions[s, t]
                if not n:
                    continue
                a, b = starts[s], ends[t]
                starts[s] += n
                ends[t] += n
                x0, x1 = i + 0.08, i + 1 - 0.08
                verts = [
                    (x0, a),
                    (x0 + 0.35, a),
                    (x1 - 0.35, b),
                    (x1, b),
                    (x1, b + n),
                    (x1 - 0.35, b + n),
                    (x0 + 0.35, a + n),
                    (x0, a + n),
                    (x0, a),
                ]
                codes = [MPath.MOVETO, *[MPath.CURVE4] * 3, MPath.LINETO, *[MPath.CURVE4] * 3, MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), fc=STATES[s], alpha=0.36, lw=0))
    for i, count in enumerate(counts):
        for s in states:
            n = count[s]
            y = bottoms[i][s]
            ax.add_patch(Rectangle((i - 0.07, y), 0.14, n, color=STATES[s], lw=0))
            if n > 4:
                ax.text(
                    i,
                    y + n / 2,
                    str(n),
                    ha="center",
                    va="center",
                    color="white" if s != "Numeric only" else INK,
                    fontsize=6.2 if n < 20 else 8,
                    weight="bold",
                )
    ax.set_xlim(-0.16, 2.2)
    ax.set_ylim(170, -2)
    ax.set_xticks([0, 1, 2], [f"{g['minute']:.1f} min" for g in grades])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=6, labelsize=7.3)
    for s in ax.spines.values():
        s.set_visible(False)
    return lookups, counts


def evidence(d, out):
    p = Plate(
        "From a passing number to an evidence chain",
        "The same public cohort at three verdict snapshots · observed use, not a causal tool effect",
        height=11.2,
    )
    p.heading(0.38, 1.56, 1, "Keep three questions separate")
    boxes = [
        (0.8, 3.7, NAVY, "Execution", "Did the compiled output agree?"),
        (5.1, 3.7, MAUVE, "Required trace", "Was the device protocol witnessed?"),
        (9.4, 4.6, SAGE, "Whole-model closure", "Did every complete chain execute?"),
    ]
    for x, w, c, title, body in boxes:
        p.line(x, 1.83, x, 2.49, color=c, lw=2)
        p.text(x + 0.15, 2.05, title, size=9.4, weight="bold", color=c)
        p.text(x + 0.15, 2.37, body, size=7.3)
    p.heading(0.38, 2.88, 2, "Evidence states change; capsule identities persist")
    ax = p.axes(0.80, 3.22, 7.45, 2.22)
    lookups, counts = ribbons(ax, d["grades"])
    p.text(8.82, 3.43, "54 more capsules with trace", size=10.0, serif=True, color=NAVY, weight="bold")
    p.text(8.82, 3.85, "42 numerical-only + 12 unresolved", size=8.1, color=MAUVE, weight="bold")
    p.text(8.82, 4.15, "join 86 already evidenced cases → 140", size=7.8)
    p.text(8.82, 4.38, "+ 1 expected refusal stays separate", size=7.1, color=SAGE)
    p.rect(8.72, 4.42, 5.28, 1.15, color=CREAM)
    p.text(8.9, 4.67, "numeric_status / trace_requirement_status", size=6.4, color=BLUE, mono=True)
    p.text(8.9, 4.91, EXAMPLE, size=6.3, mono=True, color=BLUE)
    example = [next(r for r in g["capsules"] if r["capsule"] == EXAMPLE) for g in d["grades"][:2]]
    assert example[0]["numeric_status"] == example[1]["numeric_status"] == "pass"
    assert (
        example[0]["trace_requirement_status"] == "unavailable_fields"
        and example[1]["trace_requirement_status"] == "pass"
    )
    p.text(8.9, 5.19, "129.5m   pass / unavailable_fields", size=6.6, mono=True, color=GOLD)
    p.text(8.9, 5.45, "171.1m   pass / pass", size=6.6, mono=True, color=NAVY)
    for (label, color), x in zip(STATES.items(), (0.8, 4.5, 7.25, 10.3)):
        p.rect(x, 5.96, 0.24, 0.16, color=color)
        p.text(x + 0.34, 6.12, label, size=7.0)
    p.line(0.38, 6.37, 14.0, 6.37, color=GOLD, lw=0.5)
    p.heading(0.38, 6.72, 3, "What the author actually reached for")
    p.text(0.81, 7.06, "Structural witnesses in 500 completed shell requests; categories can overlap.", size=7.2)
    tool = p.axes(2.65, 7.37, 5.55, 1.18)
    used = [
        ("direct_facts_file_inspection", "Facts file"),
        ("cca_bijection", "CCA bijection"),
        ("cca_escalation_ladder", "Lever catalog"),
        ("isa_lint", "ISA lint"),
    ]
    for i, (category, label) in enumerate(used):
        rows = [m for m in d["abstractions"] if m["category"] == category]
        ok = sum(m["outer_command_succeeded"] for m in rows)
        tool.barh(i, ok, color=NAVY, height=0.55)
        tool.barh(i, len(rows) - ok, left=ok, color=MAUVE, height=0.55)
        tool.text(len(rows) + 0.2, i, str(len(rows)), va="center", fontsize=8, weight="bold")
    tool.set_yticks(range(4), [x[1] for x in used])
    tool.tick_params(axis="y", length=0, labelsize=7.3)
    tool.set_ylim(3.65, -0.65)
    tool.set_xlim(0, 12.5)
    tool.set_xticks([0, 4, 8, 12])
    tool.set_xlabel("Recognized requests", fontsize=7.1, labelpad=2)
    tool.spines["left"].set_visible(False)
    p.text(8.82, 7.51, "21 distinct requests / 22 markers", size=8.1, weight="bold", color=BLUE)
    p.text(8.82, 7.91, "ISA lint: 5 successful outer shells", size=7.6)
    p.text(8.82, 8.22, "+ 2 failed shells (not lint passes)", size=7.6, color=MAUVE)
    p.text(8.82, 8.62, "No recognized witness:", size=7.7, color=GOLD, weight="bold")
    p.text(8.82, 8.95, "facts / levers / scaffold APIs; ISA asm / disasm / debug", size=6.8)
    p.heading(0.38, 9.44, 4, "A tool effect needs matched fresh runs")
    p.text(0.81, 9.78, "Same capsules, gates, resources and budget; independent authors and repeated seeds.", size=7.2)
    arms = [
        ("EL1", "Raw baseline"),
        ("EL2", "C++ infrastructure"),
        ("EL3", "Merlin tools"),
        ("EL4", "+ RTL facts / checks"),
    ]
    for i, (label, body) in enumerate(arms):
        x = 0.8 + i * 3.36
        p.rect(x, 10.01, 3.15, 0.40, color=CREAM if i < 3 else NAVY)
        p.text(x + 0.10, 10.27, label, size=8.0, weight="bold", color=BLUE if i < 3 else "white")
        p.text(x + 0.72, 10.27, body, size=7.2, color=INK if i < 3 else "white")
    p.footer(
        "Capsule states do not close full models. Request witnesses are not dynamic calls; "
        "zero is not proof of non-use. No matched tool ablation was run."
    )
    p.save(out, "03_evidence_and_abstractions")


def contact(out):
    names = ["01_compiler_formation", "02_execution_and_capsules", "03_evidence_and_abstractions"]
    canvas = Image.new("RGB", (1500, int(1500 * 10.8 / 14.4) * 3 + 120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), "PHASE 1 / EDITORIAL REDESIGN", fill=NAVY, font_size=30)
    y = 80
    for name in names:
        im = Image.open(out / f"{name}.png").convert("RGB")
        im.thumbnail((1470, 1125))
        canvas.paste(im, ((1500 - im.width) // 2, y))
        y += im.height + 20
    canvas.crop((0, 0, 1500, y)).save(out / "contact_sheet.png")


def evidence_summary(d):
    """Small public audit projection for the numeric callouts in the plates."""
    events = d["invocations"]
    time_by_stage = {stage: sum(e["b"] - e["a"] for e in events if e["category"] == stage) for stage in STAGES}
    last = d["grades"][-1]
    lookups = [{r["capsule"]: state(r) for r in g["capsules"]} for g in d["grades"]]
    assert all(set(rows) == set(lookups[0]) for rows in lookups)
    assert len(lookups[0]) == 153 and len(d["agents"]) == 7
    assert sum(len(a["commands"]) for a in d["agents"]) == 500
    transitions = collections.Counter((lookups[0][n], lookups[1][n]) for n in lookups[0])
    assert transitions["Numeric only", "Execution + trace"] == 42
    assert transitions["Failed / incomplete", "Execution + trace"] == 12
    assert collections.Counter(lookups[2].values()) == {
        "Execution + trace": 140,
        "Refusal met": 1,
        "Failed / incomplete": 12,
    }
    selected = [e for e in events if e["capsule"] == EXAMPLE]
    own_gsim = [e for e in selected if e["category"] == "L3 gSIM"]
    return {
        "public_capsules": len(last["capsules"]),
        "observed_actor_commands": 500,
        "grade_snapshots": [
            {"minute": g["minute"], "evidence_states": dict(collections.Counter(v.values()))}
            for g, v in zip(d["grades"], lookups)
        ],
        "stage_invocations": dict(collections.Counter(e["category"] for e in events)),
        "stage_process_wall_minutes": time_by_stage,
        "gsim_share_of_recorded_process_wall": time_by_stage["L3 gSIM"] / sum(time_by_stage.values()),
        "capacity_example": {
            "capsule": EXAMPLE,
            "gsim_runs": len(own_gsim),
            "gsim_process_wall_minutes": sum(e["b"] - e["a"] for e in own_gsim),
            "share_of_gsim_process_wall": sum(e["b"] - e["a"] for e in own_gsim) / time_by_stage["L3 gSIM"],
            "independent_intervals": [
                {
                    "category": e["category"],
                    "start_min": e["a"],
                    "end_min": e["b"],
                    "normalized_record_sha256": hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest(),
                }
                for e in selected
                if e["category"] in {"L2 Spike", "L3 gSIM"}
            ],
        },
        "token_totals": d["token_totals"],
        "scope": (
            "Recorded wall intervals, client-observed usage and discrete evidence states; "
            "no CPU, target-cycle or causal claim."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--font-dir", type=Path, help="Optional local DM Serif Display/Inter bundle, copied with licenses")
    ap.add_argument(
        "--iteration-notes", type=Path, help="Optional public visual-review JSON to bind into the final receipt"
    )
    args = ap.parse_args()
    d = json.loads(args.data.read_text())
    if d.get("schema") != "merlin.phase1.paper_exploration.v1":
        ap.error("Expected the existing public normalized Phase 1 projection")
    summary = evidence_summary(d)
    if args.out.exists():
        ap.error("Select a fresh output directory; prior visual iterations are immutable")
    args.out.mkdir(parents=True)
    shutil.copy2(args.data, args.out / "data.json")
    font_assets = select_fonts(args.font_dir, args.out)
    for plot in (run_story, execution, evidence):
        plot(d, args.out)
    contact(args.out)
    figures = [
        (
            "01_compiler_formation",
            "How did author activity, usage and qualification evolve on one clock?",
            "The comparable execution-plus-trace count moves 86→140→140; authoring ends near 130 min, "
            "while later verdicts remain separate. Subset feedback can overlap.",
        ),
        (
            "02_execution_and_capsules",
            "Which recorded work dominates, and where does it occur?",
            "447 gSIM processes account for 73.3% of retained subprocess wall intervals; one named "
            "capacity capsule contributes 46.6% of the gSIM total across five executions.",
        ),
        (
            "03_evidence_and_abstractions",
            "What closes an evidence gap, and what does tool use establish?",
            "42 numerical-only and 12 unresolved identities join 86 already evidenced cases; "
            "the 21 structurally matched requests establish observational adoption, not an abstraction effect.",
        ),
    ]
    public_figures = []
    for name, question, finding in figures:
        im = Image.open(args.out / f"{name}.png")
        preview = args.out / f"{name}_print_150dpi.png"
        im.resize((1080, round(im.height * 1080 / im.width)), Image.Resampling.LANCZOS).save(preview, dpi=(150, 150))
        public_figures.append(
            {
                "name": name,
                "question": question,
                "key_finding": finding,
                "exports": {ext: str(args.out / f"{name}.{ext}") for ext in ("png", "pdf", "svg")},
                "print_preview_7_2in_150dpi": str(preview),
            }
        )
    source = args.out / "source"
    source.mkdir()
    shutil.copy2(__file__, source / Path(__file__).name)
    shutil.copy2(STYLE_PATH, source / STYLE_PATH.name)
    if args.iteration_notes:
        shutil.copy2(args.iteration_notes, args.out / "iteration_notes.json")
    (args.out / "evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema": "merlin.phase1.editorial_render.v1",
        "run": d["run"],
        "data_sha256": digest(args.data),
        "generator_sha256": digest(__file__),
        "style_sha256": digest(STYLE_PATH),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "fonts": {
            role: {"family": family, "file_sha256": digest(font_manager.findfont(family, fallback_to_default=False))}
            for role, family in {"serif": SERIF, "sans": SANS, "mono": MONO}.items()
        },
        "physical_width_inches": 7.2,
        "figures": public_figures,
        "font_assets": font_assets,
        "reproduction_command": (
            f"python source/{Path(__file__).name} --data data.json --out REPRODUCED"
            + (" --font-dir fonts" if args.font_dir else "")
        ),
        "extraction_and_render_command": shlex.join(
            [
                "python",
                str(Path(__file__).resolve()),
                "--data",
                str(args.data.resolve()),
                "--out",
                str(args.out.resolve()),
                *(["--font-dir", str(args.font_dir.resolve())] if args.font_dir else []),
                *(["--iteration-notes", str(args.iteration_notes.resolve())] if args.iteration_notes else []),
            ]
        ),
        "visual_review_sha256": digest(args.iteration_notes) if args.iteration_notes else None,
        "evidence_summary_sha256": digest(args.out / "evidence_summary.json"),
        "outputs": {p.name: digest(p) for p in args.out.iterdir() if p.suffix in {".png", ".svg", ".pdf"}},
        "limits": d["limits"],
    }
    (args.out / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "REPRODUCE.txt").write_text(
        "PHASE1 EDITORIAL PLATES\n\nDirect PNG/PDF/SVG exports; no HTML required.\n"
        "Install matplotlib==3.11.2 numpy==2.5.4 Pillow==13.0.0"
        + (f" fontTools=={font_assets['fonttools_version']}" if font_assets else "")
        + ".\n"
        f"Selected fonts: {SERIF}; {SANS}; {MONO}. Bundled local fonts retain licenses and source hashes.\n"
        "Regenerate from this folder, selecting a fresh output directory:\n"
        f"  {manifest['reproduction_command']}\n\n"
        "The 7.2-inch-wide figures and 150 dpi previews use only the included public normalized data.\n"
        "render_manifest.json pins data, sources, font files and outputs. "
        "evidence_summary.json binds quantitative callouts.\n"
        "Any included iteration_notes.json records actual review critiques and retained alternatives.\n"
        "This historical campaign did not complete formal Phase 1; "
        "no private model evidence or new simulations are included.\n"
    )
    print(json.dumps({"output": str(args.out), "figures": 3, "data_sha256": digest(args.data)}))


if __name__ == "__main__":
    main()
