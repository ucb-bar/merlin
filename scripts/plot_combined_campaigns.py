"""Combined MX-Gemmini + Radiance(Muon) autocomp spend figure (Merlin house style).

Merges the two campaigns into one 'final' view in the radiance fig8 layout:
  left  = cumulative cost per beam iteration, per kernel (both campaigns overlaid)
  right = campaign-total cumulative cost over the shared calendar, both curves + grand total.

Reads:
  MX-Gemmini : the corpus bundle runs/*/cost_ledger.jsonl + cost/project-spend-total.json
  Radiance   : radiance_only_kernels/index/{cost_by_problem.csv, muon-spend_total.json}
"""
import csv
import json
import os
import sys

sys.path.insert(0, "/scratch/agustin/projects/oscar-merlin/scripts")
import matplotlib
matplotlib.use("Agg")
from merlin_plotstyle import *          # palette, helpers
use_merlin_style()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# reuse MX per-iteration segmentation + paths from the MX script (import is side-effect-safe)
from plot_mx_autocomp import segment_cost as mx_segment, BUNDLE as MX_BUNDLE

RAD = "/scratch/agustin/projects/oscar-merlin/tmp/kernels/radiance_only_kernels"
OUT = "/scratch/agustin/projects/oscar-merlin/artifacts/presentation/combined"
os.makedirs(OUT, exist_ok=True)

MX_TOTAL = json.load(open(f"{MX_BUNDLE}/cost/project-spend-total.json"))
RAD_TOTAL = json.load(open(f"{RAD}/index/muon-spend_total.json"))


def save(fig, name, dpi=180):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=dpi, facecolor=BG)
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# ---- per-kernel cumulative-$ curves ---------------------------------------------
def mx_curve(run):
    segs = mx_segment(run)
    return np.arange(len(segs)), np.cumsum([s["cost"] for s in segs])


def rad_curves():
    raw = {}
    for r in csv.DictReader(open(f"{RAD}/index/cost_by_problem.csv")):
        raw.setdefault((int(r["prob"]), r["run"]), []).append((int(r["iteration"]), float(r["usd"])))
    out = {}
    for k, rows in raw.items():
        rows.sort()
        out[k] = (np.array([x[0] for x in rows]), np.cumsum([x[1] for x in rows]))
    return out


RADC = rad_curves()


# (run/key, label, colour, linestyle, marker, significant)
MX_LINES = [
    (("mxpipe_gemmini-mx-matmul_0_iters5",), "MX · matmul (gen)",     NAVY,  "-",  "o", True),
    (("mxpipe_fork-mm-fp8-256_0_iters6",),   "MX · fp8-256 (golden)", NAVY,  "--", "s", True),
    (("mxpipe_real-pi05-ffn_0_iters7",),     "MX · pi05-ffn",         SLATE, "-",  "D", False),
    (("mxpipe_real-smolvla-ffn_0_iters5",),  "MX · smolvla-ffn",      SLATE, "--", "v", False),
]
RAD_LINES = [
    ((0, "built:muon_muon_0_beam_iters5_cyclotron"),            "Rad · matmul 64³",      MAUVE, "-",  "o", True),
    ((1, "built:muon_muon_1_beam_iters12_bw6harvest_cyclotron"), "Rad · conv patch (12it)", MAUVE, "--", "s", True),
    ((10, "built:muon_muon_10_beam_iters12_bw6harvest_cyclotron"), "Rad · matmul K384",   SAGE,  "-",  "D", False),
    ((7, "built:muon_muon_7_beam_iters5_cyclotron"),            "Rad · flash-attn",      SAGE,  "--", "v", False),
]


def _plot_line(ax, x, y, col, ls, m, sig, lab):
    ax.plot(x, y, ls, color=col, lw=2.9 if sig else 1.8, alpha=1.0 if sig else 0.75,
            zorder=5 if sig else 3, marker=m, ms=5.5 if sig else 3.6, mec=INK, mew=0.6,
            mfc=col, label=lab)


def fig_combined():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                                   gridspec_kw=dict(width_ratios=[1.3, 1]))

    # ---- left: per-kernel cumulative cost per iteration, both campaigns ----------
    style_ax(axL)
    for (key,), lab, col, ls, m, sig in MX_LINES:
        x, y = mx_curve(key)
        _plot_line(axL, x, y, col, ls, m, sig, lab)
    for key, lab, col, ls, m, sig in RAD_LINES:
        x, y = RADC[key]
        _plot_line(axL, x, y, col, ls, m, sig, lab)
    axL.set_xlabel("beam iteration")
    axL.set_ylabel("cumulative cost  (USD)")
    axL.set_xlim(-0.3, 12.6)
    axL.set_ylim(bottom=0)
    title(axL, "cost per iteration, per kernel")
    # two grouped legends: MX (cool) and Radiance (warm)
    mx_h = [Line2D([0], [0], color=c, ls=ls, lw=2.6, marker=m, mfc=c, mec=INK, label=lab)
            for (_,), lab, c, ls, m, _ in MX_LINES]
    rad_h = [Line2D([0], [0], color=c, ls=ls, lw=2.6, marker=m, mfc=c, mec=INK, label=lab)
             for _, lab, c, ls, m, _ in RAD_LINES]
    l1 = axL.legend(handles=mx_h, loc="upper left", fontsize=8.8, title="MX-Gemmini",
                    title_fontproperties={"weight": "bold"}, borderpad=0.6)
    axL.add_artist(l1)
    axL.legend(handles=rad_h, loc="upper left", bbox_to_anchor=(0.0, 0.66), fontsize=8.8,
               title="Radiance (Muon)", title_fontproperties={"weight": "bold"}, borderpad=0.6)

    # ---- right: combined campaign total over the shared calendar -----------------
    style_ax(axR)
    days = sorted(set(MX_TOTAL["by_day"]) | set(RAD_TOTAL["by_day"]))
    x = np.arange(len(days))
    labels = [d.replace("2026-", "") for d in days]

    def cum_over(byday):
        run = 0.0
        out = []
        for d in days:
            run += byday.get(d, 0.0)
            out.append(run)
        return np.array(out)

    mx_cum = cum_over(MX_TOTAL["by_day"])
    rad_cum = cum_over(RAD_TOTAL["by_day"])
    grand_total = MX_TOTAL["total_usd"] + RAD_TOTAL["total_usd"]
    grand_calls = MX_TOTAL["calls"] + RAD_TOTAL["calls"]

    axR.fill_between(x, 0, mx_cum, color=NAVY, alpha=0.10, zorder=2)
    axR.fill_between(x, 0, rad_cum, color=MAUVE, alpha=0.10, zorder=2)
    axR.plot(x, mx_cum, "-", color=NAVY, lw=3, marker="o", ms=6.5, mec=INK, mew=0.8,
             zorder=5, label="MX-Gemmini")
    axR.plot(x, rad_cum, "-", color=MAUVE, lw=3, marker="s", ms=6.0, mec=INK, mew=0.8,
             zorder=4, label="Radiance (Muon)")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9.5)
    axR.set_xlim(-0.3, len(days) - 1 + 0.9)
    axR.set_ylim(0, max(mx_cum[-1], rad_cum[-1]) * 1.22)
    axR.set_xlabel("active campaign day (2026)")
    axR.set_ylabel("cumulative cost  (USD)")
    axR.text(x[-1], mx_cum[-1], f"  ${MX_TOTAL['total_usd']:.2f}", color=NAVY, fontsize=10.5,
             fontweight="bold", va="bottom", ha="left")
    axR.text(x[-1], rad_cum[-1], f"  ${RAD_TOTAL['total_usd']:.2f}", color=MAUVE, fontsize=10.5,
             fontweight="bold", va="top", ha="left")
    axR.legend(loc="upper left", fontsize=9.5)
    emph(axR, len(days) - 1 + 0.35, max(mx_cum[-1], rad_cum[-1]) * 1.18,
         f"combined  ${grand_total:.0f}\n{grand_calls:,} LLM calls", color=GOLD, fs=12,
         ha="right", va="top")
    title(axR, "campaign total")

    suptitle(fig, "Autocomp search spend across both campaigns — MX-Gemmini + Radiance", y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_combined_campaigns")


def fig_combined_by_model():
    """Bonus: total spend by model, both campaigns stacked per model."""
    def norm(m):
        m = m.lower()
        if "sonnet" in m: return "Claude Sonnet 4.6"
        if "qwen" in m: return "Qwen3-Coder 480B"
        if "gemini-3.5-flash" in m: return "Gemini 3.5 Flash"
        if "gemini-3.1-pro" in m: return "Gemini 3.1 Pro"
        if "gemini-3-flash" in m: return "Gemini 3 Flash"
        if "gemini-2.5" in m: return "Gemini 2.5 Flash"
        return m
    mx, rad = {}, {}
    for k, v in MX_TOTAL["by_model"].items():
        mx[norm(k)] = mx.get(norm(k), 0) + v
    for k, v in RAD_TOTAL["by_model"].items():
        rad[norm(k)] = rad.get(norm(k), 0) + v
    models = sorted(set(mx) | set(rad), key=lambda m: -(mx.get(m, 0) + rad.get(m, 0)))
    models = [m for m in models if (mx.get(m, 0) + rad.get(m, 0)) >= 1.0]
    x = np.arange(len(models))
    mxv = np.array([mx.get(m, 0) for m in models])
    radv = np.array([rad.get(m, 0) for m in models])
    w = 0.62

    fig, ax = plt.subplots(figsize=(11, 6.2))
    style_ax(ax)
    for xi, tot in zip(x, mxv + radv):           # one block shadow per stacked total
        block_shadow(ax, xi - w / 2, 0, w, tot, z=2.0)
    ax.bar(x, mxv, w, color=NAVY, edgecolor=INK, lw=1.2, zorder=3, label="MX-Gemmini")
    ax.bar(x, radv, w, bottom=mxv, color=MAUVE, edgecolor=INK, lw=1.2, zorder=3,
           label="Radiance (Muon)")
    for xi, a, b in zip(x, mxv, radv):
        ax.text(xi, a + b + 3, f"${a+b:.0f}", ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5, rotation=12, ha="right")
    ax.set_ylabel("cumulative cost  (USD)")
    ax.set_ylim(0, (mxv + radv).max() * 1.2)
    grand = MX_TOTAL["total_usd"] + RAD_TOTAL["total_usd"]
    calls = MX_TOTAL["calls"] + RAD_TOTAL["calls"]
    emph(ax, len(models) - 0.5, (mxv + radv).max() * 1.12,
         f"combined ${grand:.0f} · {calls:,} calls", color=GOLD, fs=12, ha="right", va="top")
    ax.legend(loc="upper center", fontsize=9.5)
    title(ax, "Spend by model, both campaigns")
    suptitle(fig, "Where the combined $321 budget went — Sonnet dominant, Gemini+Qwen the workhorse", y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_combined_by_model")


def _load_outcomes():
    mx = [json.loads(l) for l in open(f"{MX_BUNDLE}/dataset/journeys.jsonl")]
    rad = [json.loads(l) for l in open(f"{RAD}/index/all_transforms.jsonl")]
    return mx, rad


def fig_combined_yield_funnel():
    """Combined candidate-yield + failure-mode comparison, both campaigns."""
    mx, rad = _load_outcomes()
    import collections
    mxn = len(mx); radn = len(rad)
    mx_oc = collections.Counter(r.get("outcome") for r in mx)
    rad_oc = collections.Counter(r.get("outcome") for r in rad)
    mx_corr = sum(1 for r in mx if r.get("correct"))
    rad_corr = sum(1 for r in rad if r.get("correct"))
    rad_lab = sum(rad_oc[k] for k in rad_oc if k)            # labeled subset size

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                                   gridspec_kw=dict(width_ratios=[1, 1.08]))

    # ---- left: yield as % of generated (fair across the size gap) ---------------
    style_ax(axL, grid="x")
    stages = [
        ("generated",            100.0, 100.0, mxn, radn, False),
        ("compiled & correct",   100 * mx_corr / mxn, 100 * rad_corr / radn, mx_corr, rad_corr, False),
        ("improved on parent",   100 * mx_oc["improved"] / mxn, 100 * rad_oc["improved"] / radn,
         mx_oc["improved"], rad_oc["improved"], True),
    ]
    y = np.arange(len(stages))[::-1]
    h = 0.36
    for yi, (lab, mxp, radp, mxc, radc, star) in zip(y, stages):
        hbars(axL, [yi + h / 2 + 0.02], [mxp], NAVY, height=h)
        hbars(axL, [yi - h / 2 - 0.02], [radp], MAUVE, height=h)
        axL.text(mxp + 1.5, yi + h / 2 + 0.02, f"{mxp:.0f}%  ({mxc})", va="center",
                 fontsize=9, color=INK)
        sfx = "*" if star else ""
        axL.text(radp + 1.5, yi - h / 2 - 0.02, f"{radp:.0f}%  ({radc}{sfx})", va="center",
                 fontsize=9, color=INK)
    axL.set_yticks(y)
    axL.set_yticklabels([s[0] + ("*" if s[5] else "") for s in stages], fontsize=10.5)
    axL.set_xlim(0, 118)
    axL.set_xlabel("share of generated candidates  (%)")
    emph(axL, 60, y[1], f"valid yield: {100*mx_corr/mxn:.0f}% vs {100*rad_corr/radn:.0f}%",
         color=GOLD, fs=11, ha="center", va="center")
    title(axL, "Candidate yield")

    # ---- right: how the invalid ones fail (% within labeled attempts) -----------
    style_ax(axR, grid="x")
    order = ["compile_error", "incorrect", "regressed", "correct_no_gain", "improved"]
    olab = {"compile_error": "compile error", "incorrect": "incorrect (wrong result)",
            "regressed": "regressed", "correct_no_gain": "correct, no gain", "improved": "improved"}
    yb = np.arange(len(order))[::-1]
    for yi, k in zip(yb, order):
        mxp = 100 * mx_oc.get(k, 0) / mxn
        radp = 100 * rad_oc.get(k, 0) / rad_lab if rad_lab else 0
        hbars(axR, [yi + h / 2 + 0.02], [mxp], NAVY, height=h)
        if k != "incorrect":          # radiance ledger has no separate 'incorrect' label
            hbars(axR, [yi - h / 2 - 0.02], [radp], MAUVE, height=h)
        axR.text(mxp + 1.2, yi + h / 2 + 0.02, f"{mxp:.0f}%", va="center", fontsize=9, color=INK)
        if k != "incorrect":
            axR.text(radp + 1.2, yi - h / 2 - 0.02, f"{radp:.0f}%", va="center", fontsize=9, color=INK)
        else:
            axR.text(1.2, yi - h / 2 - 0.02, "n/a (Radiance)", va="center", fontsize=8,
                     color=INK, fontstyle="italic")
    axR.set_yticks(yb); axR.set_yticklabels([olab[k] for k in order], fontsize=10.5)
    axR.set_xlim(0, 92)
    axR.set_xlabel("share of labeled attempts  (%)")
    title(axR, "How the invalid ones fail")
    axR.annotate("different failure mode:\nMX = numerically wrong,\nRadiance = won't compile\n(register oversubscription)",
                 (81, yb[0]), xytext=(58, yb[0] - 1.7), fontsize=9, color=BLUE,
                 fontweight="bold", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2,
                                 connectionstyle="arc3,rad=0.2"))

    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(fc=NAVY, ec=INK, label=f"MX-Gemmini  (n={mxn})"),
                        Patch(fc=MAUVE, ec=INK, label=f"Radiance/Muon  (n={radn})")],
               loc="lower right", fontsize=9.5)
    suptitle(fig, "Most generated kernels are invalid in both campaigns — but they fail differently", y=1.00)
    fig.text(0.5, -0.01,
             f"* improved counts: MX over all {mxn} attempts; Radiance over its ledger-labeled "
             f"subset (n={rad_lab}, probs 1/2/3/7) — a floor, not the full population.",
             ha="center", fontsize=8.3, color=INK, fontstyle="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig_combined_yield_funnel")


def fig_candidate_yield():
    """Standalone 'Candidate yield' panel (the left half of the funnel), enlarged for slides.

    Same data/series as fig_combined_yield_funnel's left panel; only legibility changes:
    larger numbers + labels, and value annotations pushed clear of the bars' block shadows.
    """
    from matplotlib.patches import Patch
    mx, rad = _load_outcomes()
    import collections
    mxn = len(mx); radn = len(rad)
    mx_oc = collections.Counter(r.get("outcome") for r in mx)
    rad_oc = collections.Counter(r.get("outcome") for r in rad)
    mx_corr = sum(1 for r in mx if r.get("correct"))
    rad_corr = sum(1 for r in rad if r.get("correct"))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    style_ax(ax, grid="x")
    stages = [
        ("generated",            100.0, 100.0, mxn, radn, False),
        ("compiled & correct",   100 * mx_corr / mxn, 100 * rad_corr / radn, mx_corr, rad_corr, False),
        ("improved on parent",   100 * mx_oc["improved"] / mxn, 100 * rad_oc["improved"] / radn,
         mx_oc["improved"], rad_oc["improved"], True),
    ]
    y = np.arange(len(stages))[::-1]
    h = 0.36
    # push annotations well past the bar end + its ~5.5pt block shadow so nothing touches
    for yi, (lab, mxp, radp, mxc, radc, star) in zip(y, stages):
        hbars(ax, [yi + h / 2 + 0.02], [mxp], NAVY, height=h)
        hbars(ax, [yi - h / 2 - 0.02], [radp], MAUVE, height=h)
        ax.text(mxp + 3.5, yi + h / 2 + 0.02, f"{mxp:.0f}%  ({mxc})", va="center",
                fontsize=13, color=INK)
        sfx = "*" if star else ""
        ax.text(radp + 3.5, yi - h / 2 - 0.02, f"{radp:.0f}%  ({radc}{sfx})", va="center",
                fontsize=13, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] + ("*" if s[5] else "") for s in stages], fontsize=13.5)
    ax.set_xlim(0, 135)
    ax.set_xlabel("share of generated candidates  (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    emph(ax, 64, y[1], f"valid yield: {100*mx_corr/mxn:.0f}% vs {100*rad_corr/radn:.0f}%",
         color=GOLD, fs=14, ha="center", va="center")
    title(ax, "Candidate yield", fs=18)
    ax.legend(handles=[Patch(fc=NAVY, ec=INK, label=f"MX-Gemmini  (n={mxn})"),
                       Patch(fc=MAUVE, ec=INK, label=f"Radiance/Muon  (n={radn})")],
              loc="lower right", fontsize=12)
    fig.text(0.5, -0.02,
             f"* improved counts: MX over all {mxn} attempts; Radiance over its ledger-labeled "
             f"subset (probs 1/2/3/7) — a floor, not the full population.",
             ha="center", fontsize=10, color=INK, fontstyle="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1.0))
    save(fig, "fig_candidate_yield", dpi=300)


if __name__ == "__main__":
    print("rendering ->", OUT)
    fig_combined()
    fig_combined_by_model()
    fig_combined_yield_funnel()
    fig_candidate_yield()
    print("done.")
