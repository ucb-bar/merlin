"""MX-Gemmini autocomp campaign — presentation figures (Merlin house style).

Reads the optimization corpus bundle (dataset/index.csv, journeys/, runs/*/cost_ledger.jsonl,
cost/project-spend*) and renders six figures into output/presentation/mx_autocomp/.

Run:  python scripts/plot_mx_autocomp.py
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, "/scratch/agustin/projects/oscar-merlin/scripts")
import matplotlib
matplotlib.use("Agg")
from merlin_plotstyle import *          # palette, vbars/hbars, style_ax, title, suptitle, emph, SHADOW
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

use_merlin_style()

BUNDLE = "/scratch/agustin/projects/chipyard-mx/generators/gemmini/mx-autocomp/bundle"
OUT = "/scratch/agustin/projects/oscar-merlin/artifacts/presentation/mx_autocomp"
os.makedirs(OUT, exist_ok=True)

# pipeline identity (consistent colour across every figure)
PIPE = {
    "sonnet": dict(c=NAVY,  lab="Sonnet-4.6"),
    "geminiqwen": dict(c=MAUVE, lab="Gemini-flash + Qwen-coder"),
    "geminipro": dict(c=SAGE,  lab="Gemini-3.1-pro"),
}
# truth-tier identity
TIER = {"RTL": NAVY, "faithful": BLUE, "spike": SLATE}


# ----------------------------------------------------------------- data loading
def load_index():
    rows = []
    with open(f"{BUNDLE}/dataset/index.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_journey(run):
    return json.load(open(f"{BUNDLE}/dataset/journeys/{run}.json"))


def classify_pipeline(j):
    m = " ".join(j.get("plan_models", []) + j.get("code_models", [])).lower()
    if "qwen" in m or ("gemini-3.5" in m):
        return "geminiqwen"
    if "gemini-3.1" in m:
        return "geminipro"
    return "sonnet"


def segment_cost(run):
    """Per-iteration (cost, in_tok, out_tok) by splitting cost_ledger on menu_generation groups.
    Segment 0 = context-selection/setup; segments 1..N = optimization iterations 1..N."""
    rows = [json.loads(l) for l in open(f"{BUNDLE}/runs/{run}/cost_ledger.jsonl")]
    segs, cur, prev = [], None, None
    for r in rows:
        ph = r["phase"]
        if cur is None or (ph == "menu_generation" and prev != "menu_generation"):
            cur = {"cost": 0.0, "in": 0, "out": 0}
            segs.append(cur)
        cur["cost"] += r["cost_usd"]
        cur["in"] += r["input_tokens"]
        cur["out"] += r["output_tokens"]
        prev = ph
    return segs


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=170, facecolor=BG)
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", facecolor=BG)
    print("  wrote", name)


# hard-coded silicon-truth numbers from results/MX_AUTOCOMP_RESULTS.md (not in index.csv)
RTL_ROWS = [   # (label, baseline_cyc, final_cyc, speedup)
    ("fp6 mm 128³×512", 73085, 46958, 1.56),
    ("fp4 mm 128³×512", 40383, 33253, 1.21),
    ("fp8 mm 128³×256", 41872, 36577, 1.14),
    ("fp8 mm 64³",       3019,  3024, 1.00),
]
FAITHFUL_ROWS = [
    ("pi05-ffn 256×256×512", 427143, 341347, 1.25),
    ("fp4 mm 128×128×256",    18909,  18698, 1.01),
    ("fp6 mm 128×128×256",    18914,  18703, 1.01),
]

# curated significant runs for the per-iteration spend figure (span both pipelines)
CURATED_SPEND = [
    ("mxpipe_gemmini-mx-matmul_0_iters5",  "matmul (generated)"),
    ("mxpipe_fork-mm-fp8-256_0_iters6",    "fp8-256 (golden)"),
    ("mxpipe_gemmini-mx-matmul-fp4_0_iters7", "fp4 mm (faithful)"),
    ("mxpipe_real-pi05-ffn_0_iters7",      "pi05-ffn (faithful)"),
    ("mxpipe_real-pi05-ffn_0_iters5",      "pi05-ffn (spike)"),
    ("mxpipe_real-smolvla-ffn_0_iters5",   "smolvla-ffn"),
    ("mxpipe_real-llm-attn_0_iters5",      "llm-attn"),
    ("mxpipe_real-bitvla_0_iters5",        "bitvla"),
]


# ============================================================ FIG 1 cum spend/iter
def fig_cum_spend_per_iter():
    fig, (axc, axt) = plt.subplots(1, 2, figsize=(15, 6.2))
    style_ax(axc, grid="y"); style_ax(axt, grid="y")
    lstyles = ["-", "--", "-.", ":"]
    legend_h = []
    pricey = (0, 0, "")  # (cost, x, label) of most expensive endpoint
    for k, (run, lab) in enumerate(CURATED_SPEND):
        j = load_journey(run)
        pipe = classify_pipeline(j)
        col = PIPE[pipe]["c"]
        ls = lstyles[k % len(lstyles)]
        segs = segment_cost(run)
        cum_c = np.cumsum([s["cost"] for s in segs])
        cum_t = np.cumsum([s["in"] + s["out"] for s in segs]) / 1e6
        x = np.arange(len(segs))            # 0 = setup, 1..N = iterations
        axc.plot(x, cum_c, ls=ls, color=col, lw=2.2, marker="o", ms=5,
                 mfc=col, mec=INK, mew=0.8, zorder=4)
        axt.plot(x, cum_t, ls=ls, color=col, lw=2.2, marker="o", ms=5,
                 mfc=col, mec=INK, mew=0.8, zorder=4)
        legend_h.append(Line2D([0], [0], color=col, ls=ls, lw=2.2, marker="o",
                               mfc=col, mec=INK, label=f"{lab}"))
        if cum_c[-1] > pricey[0]:
            pricey = (cum_c[-1], x[-1], lab)
    for ax, ylab in ((axc, "cumulative spend  (USD)"), (axt, "cumulative tokens  (millions)")):
        ax.set_xlabel("optimization iteration")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.2, 7.3)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(0, 8))
    title(axc, "Cumulative spend per iteration")
    title(axt, "Cumulative tokens per iteration")
    # pipeline-colour legend (left) + per-kernel line legend (right)
    pipe_h = [Line2D([0], [0], color=PIPE[p]["c"], lw=3, label=PIPE[p]["lab"])
              for p in ("sonnet", "geminiqwen", "geminipro")]
    axc.legend(handles=pipe_h, loc="upper left", fontsize=10, title="model pipeline",
               title_fontproperties={"weight": "bold"})
    axt.legend(handles=legend_h, loc="upper left", fontsize=9.0, ncol=1)
    emph(axc, pricey[1] - 0.15, pricey[0], f"${pricey[0]:.2f}", color=GOLD, fs=11,
         ha="right", va="bottom")
    suptitle(fig, "Autocomp spend grows ~linearly per beam iteration", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig1_cum_spend_per_iter")
    plt.close(fig)


# ============================================================ FIG 2 totals by model
def fig_total_by_model():
    tot = json.load(open(f"{BUNDLE}/cost/project-spend-total.json"))
    bym = tot["by_model"]
    # merge the two sonnet regions, give friendly labels
    groups = [
        ("Sonnet-4.6", bym.get("us.anthropic.claude-sonnet-4-6", 0)
                       + bym.get("global.anthropic.claude-sonnet-4-6", 0), NAVY),
        ("Gemini-3.5-flash", bym.get("gemini-3.5-flash", 0), SAGE),
        ("Qwen3-coder-480b", bym.get("qwen.qwen3-coder-480b-a35b-v1:0", 0), MAUVE),
        ("Gemini-3.1-pro", bym.get("gemini-3.1-pro-preview", 0), SLATE),
    ]
    groups.sort(key=lambda g: -g[1])
    labels = [g[0] for g in groups]
    vals = [g[1] for g in groups]
    cols = [g[2] for g in groups]
    x = np.arange(len(labels))

    fig, (axd, axr) = plt.subplots(1, 2, figsize=(14.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    style_ax(axd, grid="y")
    for xi, v, c in zip(x, vals, cols):
        vbars(axd, [xi], [v], c, width=0.62)
    for xi, v in zip(x, vals):
        axd.text(xi, v + 1.6, f"${v:.1f}", ha="center", va="bottom", fontsize=10.5,
                 color=INK, fontweight="bold")
    axd.set_xticks(x); axd.set_xticklabels(labels, fontsize=9.5, rotation=12, ha="right")
    axd.set_ylabel("spend  (USD)")
    axd.set_ylim(0, max(vals) * 1.18)
    title(axd, "Spend by model")
    emph(axd, len(labels) - 1.0, max(vals) * 1.05,
         f"total  ${tot['total_usd']:.2f}\n{tot['calls']:,} LLM calls",
         color=GOLD, fs=12, ha="right", va="top")

    # right: cumulative project ramp over calls
    style_ax(axr, grid="y")
    cum = [json.loads(l)["cumulative_usd"] for l in open(f"{BUNDLE}/cost/project-spend.jsonl")]
    axr.plot(np.arange(len(cum)), cum, color=NAVY, lw=2.6, zorder=4)
    axr.fill_between(np.arange(len(cum)), cum, color=NAVY, alpha=0.10, zorder=2)
    axr.set_xlabel("LLM call #")
    axr.set_ylabel("cumulative spend  (USD)")
    axr.set_xlim(0, len(cum) * 1.04); axr.set_ylim(0, max(cum) * 1.08)
    title(axr, "Cumulative project spend")
    emph(axr, len(cum) * 0.92, max(cum) * 0.99, f"${max(cum):.2f}", color=GOLD, fs=12,
         ha="right", va="top")
    suptitle(fig, "Project LLM cost — $167.96 across 5 models, Sonnet dominant", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig2_total_by_model")
    plt.close(fig)


# ============================================================ FIG 3 baseline vs final
def fig_baseline_vs_final():
    spike = [   # (label, speedup) — directional only
        ("pi05-ffn 256×256×512*", 2.07),
        ("smolvla-ffn 50×256×512*", 2.07),
        ("smolvla-vis 113×256×256*", 1.87),
        ("llm-attn 8×512×512*", 1.83),
        ("bitvla 32×256×256*", 1.59),
        ("attn-causal S128×D64*", 1.31),
    ]
    blocks = [("RTL (silicon truth)", [(l, s) for l, _, _, s in RTL_ROWS], "RTL"),
              ("faithful (±9% of RTL)", [(l, s) for l, _, _, s in FAITHFUL_ROWS], "faithful"),
              ("spike (directional only)", spike, "spike")]
    fig, ax = plt.subplots(figsize=(14, 6.6))
    style_ax(ax, grid="y")
    xpos, xticks, xlabs = [], [], []
    cur = 0
    gap = 1.0
    for bname, rows, tier in blocks:
        for lab, sp in rows:
            xpos.append((cur, lab, sp, tier, bname))
            xticks.append(cur); xlabs.append(lab)
            cur += 1
        cur += gap
    for x, lab, sp, tier, bname in xpos:
        hatch = "///" if tier == "spike" else ""
        col = TIER[tier]
        vbars(ax, [x], [sp], col, width=0.66, hatch=hatch, base=0.0)
        ax.text(x, sp + 0.03, f"{sp:.2f}×", ha="center", va="bottom", fontsize=9.5,
                color=INK, fontweight="bold")
    ax.axhline(1.0, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(xpos[-1][0] + 0.2, 1.0, "baseline 1.0×", fontsize=9, color=INK, va="center", ha="left")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs, rotation=32, ha="right", fontsize=9)
    ax.set_ylabel("speedup over claude-code baseline")
    ax.set_ylim(0.9, 2.45)
    # tier band labels
    for bname, rows, tier in blocks:
        xs = [p[0] for p in xpos if p[4] == bname]
        ax.text(np.mean(xs), 2.40, bname, ha="center", va="top", fontsize=10.5,
                color=TIER[tier], fontweight="bold")
    # hero
    hero_x = [p[0] for p in xpos if p[1] == "fp6 mm 128³×512"][0]
    emph(ax, hero_x, 1.82, "1.56×\nbest silicon", color=GOLD, fs=10.5, ha="center", va="bottom")
    title(ax, "Baseline to final speedup, by measurement fidelity")
    suptitle(fig, "Speedups collapse to silicon truth: 1.56× real, spike's 3–5× are mirages", y=1.00)
    leg = [Patch(facecolor=TIER["RTL"], edgecolor=INK, label="RTL — silicon truth"),
           Patch(facecolor=TIER["faithful"], edgecolor=INK, label="faithful (±9%)"),
           Patch(facecolor=TIER["spike"], edgecolor=INK, hatch="///", label="spike* — directional, not comparable")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig3_baseline_vs_final")
    plt.close(fig)


# ============================================================ FIG 4 model compare
def fig_model_compare():
    rows = load_index()
    fig, ax = plt.subplots(figsize=(12.5, 7))
    style_ax(ax, grid="both")
    ax.grid(True, axis="x", ls=":", lw=0.8, color=INK, alpha=0.22, zorder=0)
    marker = {"latency": "o", "faithful-cycles": "D"}
    seen_pipe = set(); seen_tier = set()
    pts = []
    for r in rows:
        if not r["speedup"] or not r["cost_usd"]:
            continue
        sp = float(r["speedup"]); cost = float(r["cost_usd"])
        if float(r["num_attempts"]) < 4:        # completed runs only
            continue
        j = load_journey(r["run_dir"])
        pipe = classify_pipeline(j)
        col = PIPE[pipe]["c"]
        mk = marker.get(r["metric"], "o")
        ax.scatter(cost, sp, s=150, c=col, marker=mk, edgecolors=INK, linewidths=1.1,
                   zorder=5, alpha=0.95)
        pts.append((cost, sp, r["prob_type"], pipe, r["metric"]))
        seen_pipe.add(pipe); seen_tier.add(r["metric"])
    ax.set_xlabel("run cost  (USD)")
    ax.set_ylabel("speedup achieved  (per run metric)")
    ax.set_ylim(0.9, 5.8); ax.set_xlim(0.4, 10.5)
    title(ax, "Cost vs speedup, by model pipeline")
    suptitle(fig, "What each model pipeline bought — gain per dollar", y=0.99)
    # callouts
    def callout(ptype, metric, text, dxy):
        for c, s, pt, pi, mt in pts:
            if pt == ptype and mt == metric:
                ax.annotate(text, (c, s), xytext=(c + dxy[0], s + dxy[1]),
                            fontsize=9, color=INK,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d9cfc0"),
                            arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
                return
    callout("gemmini-mx-matmul", "latency", "matmul 3–5×\n(spike, instr-count)", (0.4, 0.4))
    callout("real-pi05-ffn", "faithful-cycles", "pi05-ffn 1.25×\n(faithful, real)", (0.6, 0.9))
    pipe_h = [Line2D([0], [0], marker="o", ls="", mfc=PIPE[p]["c"], mec=INK, ms=11,
                     label=PIPE[p]["lab"]) for p in ("sonnet", "geminiqwen", "geminipro") if p in seen_pipe]
    tier_h = [Line2D([0], [0], marker="o", ls="", mfc="white", mec=INK, ms=11, label="spike latency"),
              Line2D([0], [0], marker="D", ls="", mfc="white", mec=INK, ms=10, label="faithful cycles")]
    l1 = ax.legend(handles=pipe_h, loc="upper right", fontsize=9.5, title="model pipeline",
                   title_fontproperties={"weight": "bold"})
    ax.add_artist(l1)
    ax.legend(handles=tier_h, loc="lower right", fontsize=9.5, title="metric",
              title_fontproperties={"weight": "bold"})
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig4_model_compare")
    plt.close(fig)


# ============================================================ FIG 5 outcome funnel
def fig_outcome_funnel():
    fams = {"golden-fork": "golden", "autocomp-generated": "generated", "real-model": "real-model"}
    order = ["improved", "correct_no_gain", "regressed", "incorrect", "compile_error"]
    ocol = {"improved": NAVY, "correct_no_gain": SLATE, "regressed": SAGE,
            "incorrect": MAUVE, "compile_error": BLUE}
    olab = {"improved": "improved", "correct_no_gain": "correct, no gain",
            "regressed": "regressed", "incorrect": "incorrect", "compile_error": "compile error"}
    counts = {f: {o: 0 for o in order} for f in fams}
    tot_improved = tot_all = 0
    for line in open(f"{BUNDLE}/dataset/journeys.jsonl"):
        a = json.loads(line)
        fam = a.get("family")
        if fam not in fams:
            continue
        oc = a.get("outcome")
        if oc not in counts[fam]:
            continue
        counts[fam][oc] += 1
        tot_all += 1
        if oc == "improved":
            tot_improved += 1
    fig, ax = plt.subplots(figsize=(11, 6.4))
    style_ax(ax, grid="y")
    x = np.arange(len(fams))
    famkeys = list(fams)
    width = 0.6
    totals = [sum(counts[f].values()) for f in famkeys]
    for xi, f, tot in zip(x, famkeys, totals):   # one block shadow per stacked total
        block_shadow(ax, xi - width / 2, 0, width, tot, z=2.0)
    bottoms = np.zeros(len(fams))
    for o in order:
        vals = np.array([counts[f][o] for f in famkeys], float)
        ax.bar(x, vals, width, bottom=bottoms, color=ocol[o], edgecolor=INK,
               linewidth=1.1, zorder=3, label=olab[o])
        bottoms += vals
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + max(totals) * 0.015, f"{tot} attempts", ha="center", va="bottom",
                fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([fams[f] for f in famkeys], fontsize=11)
    ax.set_ylabel("LLM plan→code→eval attempts")
    ax.set_ylim(0, max(totals) * 1.16)
    title(ax, "Attempt outcomes by kernel family")
    suptitle(fig, f"Only {100*tot_improved/tot_all:.0f}% of LLM attempts improved the kernel", y=1.00)
    emph(ax, x[-1] + 0.05, max(totals) * 1.10, f"{tot_improved}/{tot_all} improved",
         color=GOLD, fs=11, ha="right", va="top")
    ax.legend(loc="upper left", fontsize=9.5, ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig5_outcome_funnel")
    plt.close(fig)


# ============================================================ FIG 6 gain by iter
def fig_gain_by_iter():
    runs = [
        ("mxpipe_gemmini-mx-matmul_0_iters5", "matmul (generated)", NAVY, "-"),
        ("mxpipe_real-pi05-ffn_0_iters5", "pi05-ffn (spike)", MAUVE, "-"),
        ("mxpipe_real-pi05-ffn_0_iters7", "pi05-ffn (faithful)", BLUE, "--"),
        ("mxpipe_real-smolvla-ffn_0_iters5", "smolvla-ffn", SAGE, "-"),
        ("mxpipe_real-llm-attn_0_iters5", "llm-attn", SLATE, "-"),
        ("mxpipe_real-bitvla_0_iters5", "bitvla", MAUVE, ":"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    style_ax(ax, grid="y")
    maxsp = 1.0
    for run, lab, col, ls in runs:
        j = load_journey(run)
        base = j["baseline"]["score"]
        kb = j["kept_beam_per_iter"]
        xs, ys, best = [], [], base
        for it in sorted(kb, key=lambda s: int(s)):
            scores = [n["score"] for n in kb[it] if n.get("score")]
            if scores:
                best = min(best, min(scores))
            xs.append(int(it)); ys.append(base / best)
        maxsp = max(maxsp, max(ys))
        ax.plot(xs, ys, ls=ls, color=col, lw=2.4, marker="o", ms=6, mfc=col, mec=INK,
                mew=0.8, zorder=4, label=lab)
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("best speedup found so far")
    ax.set_xlim(-0.2, 7.2); ax.set_ylim(0.95, maxsp * 1.08)
    ax.set_xticks(range(0, 8))
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    title(ax, "Where the gain lands across beam iterations")
    suptitle(fig, "Most of the speedup is found in the first few iterations", y=1.00)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig6_gain_by_iter")
    plt.close(fig)


if __name__ == "__main__":
    print("rendering ->", OUT)
    fig_cum_spend_per_iter()
    fig_total_by_model()
    fig_baseline_vs_final()
    fig_model_compare()
    fig_outcome_funnel()
    fig_gain_by_iter()
    print("done.")
