#!/usr/bin/env python3
"""Reference-styled experiment plots for capsule_bench_v0.

Visual language borrowed from the NLA "language through transcript" figure the user shared: small
multiples, an x-axis that sweeps THROUGH the transcript, Gaussian-smoothed prevalence lines, and a
background shaded into coloured SECTIONS by phase. We render:

  fig1_activity_trajectory.png  — THE direct analog. One panel per run; x = progress through the run's
        transcript; smoothed prevalence lines for each agent activity (Read / Edit / Bash / Thinking /
        tool-result); background banded by round (alternating tan) with thinking stripes + rate-limit
        markers; a bold black line = fraction of capsules passing (the outcome) climbing across rounds.
  fig2_failure_planes.png       — stacked area of WHICH failure plane each capsule dies on, per round
        (trace_check / numeric / spike / command_buffer / ...). Shows the agent burning down failure
        classes round by round — and merlin's trace_check wall.
  fig3_capsule_heatmap.png      — capsule x round status grid (pass / fail / not-yet-graded), rows
        grouped by workload class. When does each capsule flip green.
  fig4_ab_summary.png           — cross-run A/B: per-class coverage bars + L3-cycle comparison +
        cost/effort + token economics, from the full-suite audit and process telemetry.

All inputs are artifacts already on disk (works on finished, frozen, OR mid-flight runs).
Usage: make_plots.py [--runs raw_baseline/rb_full_01,merlin_assisted/merlin_full_01,...] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import _trajectory as T
import _style as S

S.apply_theme()

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
RUNS = EXP / "runs"
REPORTS = EXP / "reports"

# ---- palette (muted ML-paper aesthetic; all colours come from _style) -----------------------------
BAND_A = S.CREAM        # tan round band
BAND_B = "#eef1f3"
RL_MARK = S.SALMON_E    # darker red for rate-limit markers
OUTCOME = S.INK         # near-black bold outcome line
ACT_COLORS = {
    T.KIND_READ:   S.GREEN,
    T.KIND_EDIT:   S.SALMON,
    T.KIND_BASH:   S.BLUE,
    T.KIND_THINK:  S.LAVENDER,
    T.KIND_RESULT: S.AMBER,
}
ACT_EDGES = {
    T.KIND_READ:   S.GREEN_E, T.KIND_EDIT: S.SALMON_E, T.KIND_BASH: S.BLUE_E,
    T.KIND_THINK:  S.LAVENDER_E, T.KIND_RESULT: S.AMBER_E,
}
ACT_LABEL = {T.KIND_READ: "Read", T.KIND_EDIT: "Edit/Write", T.KIND_BASH: "Bash",
             T.KIND_THINK: "Thinking", T.KIND_RESULT: "Tool result"}
PLANE_COLORS = {
    "trace_check": S.AMBER, "numeric_golden": S.SALMON, "numeric": S.SALMON,
    "spike": S.BLUE, "command_buffer": S.LAVENDER, "verilator": S.GREEN,
    "runner_internal": S.GREY, "oracle_unavailable": "#cfcfcf", "build": S.MUSTARD,
}
CLASS_ORDER = ["matmul", "acc_scale", "relu", "mlp", "attention", "movement", "conv"]


def _gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.5 or len(y) < 3:
        return y
    radius = int(max(1, round(sigma * 3)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def _global_x(timeline: list[dict]) -> np.ndarray:
    """Monotone position-through-transcript for the whole run: cumulative output tokens stitched
    across rounds when usage is present, else event ordinal. Returned normalised to [0, 100]."""
    if not timeline:
        return np.array([])
    xs = []
    offset = 0.0
    last_round = None
    last_cum = 0.0
    use_tok = any(e.get("cum_out", 0) for e in timeline)
    for i, e in enumerate(timeline):
        if use_tok:
            r = e.get("round")
            if last_round is not None and r != last_round:
                offset += last_cum
            cum = float(e.get("cum_out", 0) or 0)
            xs.append(offset + cum)
            last_cum = cum
            last_round = r
        else:
            xs.append(float(i))
    xs = np.array(xs, dtype=float)
    if xs.max() > xs.min():
        xs = (xs - xs.min()) / (xs.max() - xs.min()) * 100.0
    return xs


# ===================================================================================================
# fig1 — activity trajectory through the transcript (the reference analog)
# Backgrounds are SELECTABLE (bg=): each encodes a different meaning so they can be compared.
# ===================================================================================================
PHASE_ACT = "#f4ecda"     # tan  — model acting (text / tool call)
PHASE_ENV = "#d7e6f0"     # blue — environment responding (tool result)
PHASE_THINK = "#e3d7ef"   # purple — thinking
SIGMA_FRAC = 0.03         # smoothing window ≈ 3% of the transcript


def _phase_of(kind: str) -> str | None:
    if kind == T.KIND_THINK:
        return "think"
    if kind == T.KIND_RESULT:
        return "env"
    if kind in (T.KIND_READ, T.KIND_EDIT, T.KIND_BASH, T.KIND_TOOL_OTHER, T.KIND_TEXT):
        return "act"
    return None


def _draw_bg(ax, x, kinds, rounds, tl, sigma, bg: str) -> None:
    """Render the chosen background mode. Round labels are drawn for all modes."""
    bounds = []
    for r in np.unique(rounds):
        m = rounds == r
        x0, x1 = x[m].min(), x[m].max()
        ax.text((x0 + x1) / 2, 1.04, f"R{int(r)}", ha="center", va="bottom",
                fontsize=8, color="#999", transform=ax.get_xaxis_transform())
        bounds.append((x0, x1, int(r)))
    sep = lambda: [ax.axvline(b[1], color="#dcdcdc", lw=0.8, zorder=1) for b in bounds[:-1]]

    if bg == "rounds":
        # meaning: which loop ITERATION (round) — even rounds tinted, boundaries marked
        for x0, x1, r in bounds:
            if r % 2 == 0:
                ax.axvspan(x0, x1, color=BAND_A, alpha=0.55, lw=0, zorder=0)
        sep()
    elif bg == "reasoning":
        # meaning: WHERE the model reasoned hard — binned thinking density as a purple heat
        nb = 90
        edges = np.linspace(x.min(), x.max(), nb + 1)
        ind = (kinds == T.KIND_THINK).astype(float)
        for bi in range(nb):
            sel = (x >= edges[bi]) & (x < edges[bi + 1] if bi < nb - 1 else x <= edges[bi + 1])
            if sel.sum() == 0:
                continue
            d = ind[sel].mean()
            if d > 0:
                ax.axvspan(edges[bi], edges[bi + 1], color=PHASE_THINK,
                           alpha=float(min(0.7, d * 2.2)), lw=0, zorder=0)
        sep()
    elif bg == "turnphase":
        # meaning: transcript PHASE (faithful to the reference) — act / environment / thinking,
        # drawn as merged contiguous spans (will be finer here since agent tool-cycles are short)
        phase = [_phase_of(e["kind"]) for e in tl]
        col = {"act": PHASE_ACT, "env": PHASE_ENV, "think": PHASE_THINK}
        i = 0
        while i < len(tl):
            j = i
            while j + 1 < len(tl) and phase[j + 1] == phase[i]:
                j += 1
            if phase[i]:
                ax.axvspan(x[i], x[j], color=col[phase[i]], alpha=0.7, lw=0, zorder=0)
            i = j + 1
    elif bg == "plain":
        sep()
    ax.set_axisbelow(True)


def _draw_activity_panel(ax, run: dict, bg: str = "rounds", sigma_frac: float = SIGMA_FRAC):
    """One activity-trajectory panel. bg selects the background meaning. 'stream' switches the
    foreground from lines to a stacked-area composition. Returns the twin (outcome) axis."""
    tl = run["timeline"]
    if not tl:
        ax.set_title(f"{run['run_id']} — no transcript yet"); ax.axis("off"); return None
    x = _global_x(tl)
    sigma = max(1.0, sigma_frac * len(tl))
    kinds = np.array([e["kind"] for e in tl])
    rounds = np.array([e.get("round", 0) for e in tl])

    _draw_bg(ax, x, kinds, rounds, tl, sigma, "rounds" if bg == "stream" else bg)
    for i, e in enumerate(tl):
        if e["kind"] == T.KIND_RATELIMIT and e.get("rejected"):
            ax.axvline(x[i], color=RL_MARK, alpha=0.5, lw=1.3, ls=":", zorder=2)

    if bg == "stream":
        # foreground = stacked-area composition (fractions sum to 1) — zero line-crossing
        ys = {k: _gaussian_smooth((kinds == k).astype(float), sigma) for k in ACT_COLORS}
        tot = np.sum(list(ys.values()), axis=0); tot[tot == 0] = 1.0
        base = np.zeros_like(x, dtype=float)
        for kind, c in ACT_COLORS.items():
            frac = ys[kind] / tot
            ax.fill_between(x, base, base + frac, color=c, alpha=0.85, lw=0,
                            label=ACT_LABEL[kind], zorder=3)
            base = base + frac
        ax.set_ylim(0, 1.0); ax.set_ylabel("activity composition", fontsize=9)
    else:
        ymax = 0.0
        for kind, c in ACT_COLORS.items():
            ind = (kinds == kind).astype(float)
            if ind.sum() == 0:
                continue
            y = _gaussian_smooth(ind, sigma); ymax = max(ymax, float(y.max()))
            faint = kind == T.KIND_RESULT
            ax.plot(x, y, color=c, lw=1.6 if faint else 2.1, alpha=0.45 if faint else 0.95,
                    label=ACT_LABEL[kind], zorder=4)
        ax.set_ylim(0, max(0.45, ymax * 1.18)); ax.set_ylabel("activity prevalence", fontsize=9)

    ax.set_xlim(0, 100); ax.set_xlabel("Progress through transcript (%)", fontsize=9)
    ax.tick_params(labelsize=8); ax.spines["top"].set_visible(False)

    # outcome on its own right axis
    axr = ax.twinx()
    axr.set_ylim(0, 1.02); axr.set_ylabel("capsules passing", fontsize=9, color=OUTCOME)
    axr.tick_params(labelsize=8, colors=OUTCOME); axr.spines["top"].set_visible(False)
    rmeta = run["rounds"]
    if rmeta:
        ox, oy = [], []
        for r in np.unique(rounds):
            m = rounds == r
            rr = next((q for q in rmeta if q["round"] == int(r)), None)
            if rr and rr.get("n_passed") is not None and rr.get("n_capsules"):
                ox.append(x[m].max()); oy.append(rr["n_passed"] / rr["n_capsules"])
        if ox:
            axr.plot(ox, oy, color=OUTCOME, lw=2.8, marker="o", ms=6, zorder=7)
    return axr


def _fig1_legend(fig, ncol=4):
    handles = [Line2D([], [], color=c, lw=2.2, label=ACT_LABEL[k]) for k, c in ACT_COLORS.items()]
    handles += [Line2D([], [], color=OUTCOME, lw=2.8, marker="o", label="Capsules passing (right axis)"),
                Line2D([], [], color=RL_MARK, lw=1.4, ls=":", label="Rate-limit reject"),
                Patch(fc=PHASE_ACT, label="act/round"), Patch(fc=PHASE_ENV, label="environment"),
                Patch(fc=PHASE_THINK, label="thinking")]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def fig1_activity(runs: list[dict], out: Path, bg: str = "rounds") -> Path:
    n = len(runs); ncol = min(2, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.4 * ncol, 4.3 * nrow), squeeze=False)
    for idx, run in enumerate(runs):
        ax = axes[idx // ncol][idx % ncol]
        _draw_activity_panel(ax, run, bg=bg)
        conv = run.get("converged")
        ax.set_title(f"{run['run_id']}  ({run['arm']})" + ("  ✓ converged" if conv else ""),
                     fontsize=11, loc="left")
    for k in range(len(runs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    _fig1_legend(fig)
    fig.suptitle(f"Agent activity through the transcript — background: {bg}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


def fig1_compare(run: dict, out: Path,
                 modes=("rounds", "reasoning", "turnphase", "stream")) -> Path:
    """Same run rendered under every background mode, side by side, so the options can be compared."""
    titles = {"rounds": "A) Round bands — loop iteration #",
              "reasoning": "B) Reasoning-intensity heat — where the model thought hard",
              "turnphase": "C) Turn-phase sections — act / environment / thinking (reference-faithful)",
              "stream": "D) Stacked composition (streamgraph) — activity share, no crossings",
              "plain": "E) Plain — separators only"}
    n = len(modes); ncol = 2; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.6 * ncol, 4.3 * nrow), squeeze=False)
    for i, mode in enumerate(modes):
        ax = axes[i // ncol][i % ncol]
        _draw_activity_panel(ax, run, bg=mode)
        ax.set_title(titles.get(mode, mode), fontsize=10.5, loc="left")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    _fig1_legend(fig)
    fig.suptitle(f"fig1 background options — {run['run_id']} ({run['arm']})", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


# ===================================================================================================
# fig2 — failure-plane prevalence over rounds (stacked area)
# ===================================================================================================
def fig2_planes(runs: list[dict], out: Path) -> Path:
    n = len(runs); ncol = min(2, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 3.2 * nrow), squeeze=False)
    all_planes: list[str] = []
    for run in runs:
        for rd in run["rounds"]:
            for p in rd["failure_planes"]:
                if p not in all_planes:
                    all_planes.append(p)
    for idx, run in enumerate(runs):
        ax = axes[idx // ncol][idx % ncol]
        rmeta = run["rounds"]
        if not rmeta:
            ax.axis("off"); continue
        xr = [rd["round"] for rd in rmeta]
        series = {p: [rd["failure_planes"].get(p, 0) for rd in rmeta] for p in all_planes}
        series = {p: v for p, v in series.items() if any(v)}
        if series:
            ax.stackplot(xr, list(series.values()),
                         colors=[PLANE_COLORS.get(p, "#cccccc") for p in series],
                         labels=list(series.keys()), alpha=0.9)
        # overlay passing count
        npass = [rd["n_passed"] for rd in rmeta if rd["n_passed"] is not None]
        if npass:
            ax.plot([rd["round"] for rd in rmeta if rd["n_passed"] is not None], npass,
                    color=OUTCOME, lw=2.2, marker="o", ms=4, label="passing")
        ax.set_title(f"{run['run_id']} ({run['arm']})", fontsize=10, loc="left")
        ax.set_xlabel("Round", fontsize=8); ax.set_ylabel("# capsules", fontsize=8)
        ax.set_xticks(xr); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for k in range(len(runs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Failure plane by round — how each arm burns down (or stalls on) failure classes",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================================================
# fig3 — capsule x round status heatmap
# ===================================================================================================
def fig3_heatmap(runs: list[dict], out: Path) -> Path:
    n = len(runs); ncol = min(2, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.0 * nrow), squeeze=False)
    # status -> code: pass=2, fail/incomplete/error=1, not graded=0
    def code(s):
        return 2 if s == "pass" else (1 if s in ("fail", "incomplete", "error") else 0)
    cmap = matplotlib.colors.ListedColormap(["#ededed", S.SALMON, S.GREEN])
    for idx, run in enumerate(runs):
        ax = axes[idx // ncol][idx % ncol]
        caps = run["capsule_rounds"]
        if not caps:
            ax.set_title(f"{run['run_id']} — no per-capsule grades yet"); ax.axis("off"); continue
        order = sorted(caps.keys(), key=lambda c: (CLASS_ORDER.index(caps[c]["class"])
                       if caps[c]["class"] in CLASS_ORDER else 99, c))
        nr = max(len(v["status"]) for v in caps.values())
        M = np.zeros((len(order), nr))
        for i, c in enumerate(order):
            st = caps[c]["status"]
            for j in range(nr):
                M[i, j] = code(st[j]) if j < len(st) else 0
        ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{c}" for c in order], fontsize=6.3)
        ax.set_xticks(range(nr)); ax.set_xticklabels([f"R{j}" for j in range(nr)], fontsize=7)
        # class separators
        prev = None
        for i, c in enumerate(order):
            cl = caps[c]["class"]
            if prev is not None and cl != prev:
                ax.axhline(i - 0.5, color="white", lw=2)
            prev = cl
        ax.set_title(f"{run['run_id']} ({run['arm']})", fontsize=10, loc="left")
    for k in range(len(runs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles = [Patch(fc=S.GREEN, ec=S.GREEN_E, label="pass"),
               Patch(fc=S.SALMON, ec=S.SALMON_E, label="fail"),
               Patch(fc="#ededed", ec="#cccccc", label="not graded")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-capsule status by round (rows grouped by workload class)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================================================
# fig4 — cross-run A/B summary (coverage, cycles, cost, token economics)
# ===================================================================================================
def fig4_summary(audit: dict, runs: list[dict], out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    backends = audit.get("backends", {})
    cov = audit.get("class_coverage", {})

    # (a) per-class coverage grouped bars  (cov[cls] = {'n': N, '<backend>': passed_count})
    ax = axes[0][0]
    if cov and backends:
        bk = list(backends.keys())
        classes = [c for c in CLASS_ORDER if c in cov] + [c for c in cov if c not in CLASS_ORDER]
        w = 0.8 / max(1, len(bk))
        xpos = np.arange(len(classes))
        for bi, b in enumerate(bk):
            vals = [cov[cl].get(b, 0) / max(1, cov[cl].get("n", 1)) for cl in classes]
            S.bar(ax, xpos + bi * w, vals, w, ci=bi, label=b, shadow=True)
        ax.set_xticks(xpos + w * (len(bk) - 1) / 2)
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("fraction of class passing"); ax.set_ylim(0, 1.12)
        ax.legend(); ax.set_title("Coverage by workload class (full 25-capsule audit)")
        # callout: the classes a pilot-only backend never implemented
        zero = [c for c in classes if all(cov[c].get(b, 0) == 0 for b in bk)]
        if zero:
            S.callout(ax, "never implemented\n(" + ", ".join(zero) + ")",
                      xy=(xpos[classes.index(zero[0])] + w / 2, 0.02), xytext=(len(classes) * 0.45, 0.5),
                      color=S.SALMON_E)
        S.style_axes(ax)
    else:
        ax.axis("off"); ax.set_title("coverage: full_suite_audit.json not found", fontsize=9)

    # (b) L3 cycle comparison per capsule  (matrix = list of {capsule, '<b>__cycles', ...})
    ax = axes[0][1]
    mat = audit.get("matrix", [])
    if mat and backends:
        bk = list(backends.keys())
        rows = [r for r in mat if any(r.get(f"{b}__cycles") for b in bk)]
        rows = sorted(rows, key=lambda r: r.get("capsule", ""))[:16]
        xpos = np.arange(len(rows)); w = 0.8 / max(1, len(bk))
        for bi, b in enumerate(bk):
            ys = [r.get(f"{b}__cycles") or 0 for r in rows]
            S.bar(ax, xpos + bi * w, ys, w, ci=bi, label=b)
        ax.set_xticks(xpos + w * (len(bk) - 1) / 2)
        ax.set_xticklabels([r["capsule"][:14] for r in rows], rotation=70, ha="right", fontsize=6)
        ax.set_ylabel("L3 verilator cycles"); ax.legend()
        ax.set_title("Cycle A/B per capsule (lower = faster RTL)")
        S.style_axes(ax)
    else:
        ax.axis("off"); ax.set_title("cycles: audit matrix unavailable", fontsize=9)

    # (c) effort to converge: rounds + cost (value-labelled, paper style)
    ax = axes[1][0]
    labels, rounds_n, costs = [], [], []
    for run in runs:
        proc = run.get("process", {}) or {}
        labels.append(run["run_id"])
        rounds_n.append(run.get("n_rounds_seen", len(run["rounds"])))
        costs.append(proc.get("estimated_cost_usd"))
    xpos = np.arange(len(labels))
    br = S.bar(ax, xpos - 0.2, rounds_n, 0.4, ci=0, label="rounds", shadow=True)
    S.label_bars(ax, br, fmt="{:.0f}")
    ax2 = ax.twinx()
    bc = S.bar(ax2, xpos + 0.2, [c if c else 0 for c in costs], 0.4, ci=2, label="cost $", shadow=True)
    S.label_bars(ax2, [b for b, c in zip(bc, costs) if c], fmt="${:.0f}", color=S.MUSTARD_E)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rounds", color=S.BLUE_E); ax2.set_ylabel("est. cost $", color=S.MUSTARD_E)
    ax2.grid(False)
    ax.set_title("Effort: rounds & cost per run")
    S.style_axes(ax); ax2.spines["top"].set_visible(False)

    # (d) token economics (cache-read dominates)
    ax = axes[1][1]
    labs, series = [], {"input": [], "cache write": [], "cache read": [], "output": []}
    keymap = {"input": "input", "cache write": "cache_create", "cache read": "cache_read", "output": "output"}
    for run in runs:
        proc = run.get("process", {}) or {}
        nbm = proc.get("tokens_native_by_model", {})
        if not nbm:
            continue
        agg = {"input": 0, "cache_create": 0, "cache_read": 0, "output": 0}
        for v in nbm.values():
            for k in agg:
                agg[k] += v.get(k, 0)
        labs.append(run["run_id"])
        for disp, key in keymap.items():
            series[disp].append(agg[key])
    if labs:
        xpos = np.arange(len(labs))
        base = np.zeros(len(labs))
        ci_for = {"input": 0, "cache write": 4, "cache read": 1, "output": 3}
        for disp, vals in series.items():
            S.bar(ax, xpos, vals, 0.6, ci=ci_for[disp], bottom=base, label=disp)
            base = base + np.array(vals, dtype=float)
        for i, tot in enumerate(base):
            ax.annotate(f"{tot/1e6:.1f}M", (xpos[i], tot), ha="center", va="bottom",
                        fontsize=8.5, color="#333", xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(xpos); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("tokens"); ax.legend()
        ax.set_title("Token economics (cache-read dominates cost)")
        S.callout(ax, "cache reads ≈ 97% of tokens\n→ cheap on cache, $ from output",
                  xy=(xpos[-1], base[-1] * 0.6), xytext=(xpos[0] + 0.1, base.max() * 0.78),
                  color=S.GREEN_E)
        S.style_axes(ax)
    else:
        ax.axis("off"); ax.set_title("token economics: needs cost_time_toolcalls.yaml", fontsize=9)

    fig.suptitle("A/B summary — coverage · cycles · effort · tokens", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================================================
# fig5 — run scorecard (image-copy-7 card style: cream cards w/ hard shadow + bold horizontal bars)
# ===================================================================================================
def fig5_cards(runs: list[dict], out: Path) -> Path:
    names = [r["run_id"] for r in runs]
    rounds_n = [r.get("n_rounds_seen", len(r["rounds"])) for r in runs]
    best = []
    for r in runs:
        ps = [(rd["n_passed"], rd["n_capsules"]) for rd in r["rounds"] if rd["n_passed"] is not None]
        best.append(max(ps, key=lambda t: (t[0] or 0)) if ps else (0, 0))
    costs = [(r.get("process", {}) or {}).get("estimated_cost_usd") for r in runs]

    cards = [
        ("CONVERGENCE  ROUNDS", rounds_n, [f"{v}" for v in rounds_n], "rounds to all-pass"),
        ("PUBLIC  CAPSULES  PASSING", [p[0] or 0 for p in best],
         [f"{p[0] or 0}/{p[1] or '?'}" for p in best], "best round, of 20"),
        ("COST  PER  RUN", [c or 0 for c in costs],
         [f"${c:.0f}" if c else "in-flight" for c in costs], "USD (finished runs)"),
    ]
    n = len(cards)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 2.5 * n))
    y = np.arange(len(runs))[::-1]
    for ci, (title, vals, labels, unit) in enumerate(cards):
        ax = axes[ci]; ax.set_facecolor("none")
        bars = S.barh(ax, y, vals, height=0.5, shadow=True)
        for bi, b in enumerate(bars):
            b.set_color(S.FILLS[bi % len(S.FILLS)]); b.set_edgecolor(S.INK); b.set_linewidth(1.4)
        mx = max(vals) if max(vals) > 0 else 1
        for yi, (v, lab) in enumerate(zip(vals, labels)):
            inside = v > 0.28 * mx
            ax.text(v - 0.02 * mx if inside else v + 0.02 * mx, y[yi], lab,
                    ha="right" if inside else "left", va="center",
                    color="white" if inside else "#333", fontsize=10.5, weight="bold")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlim(0, mx * 1.2); ax.set_xticks([])
        ax.set_title(title, fontsize=12.5, loc="left", fontfamily="monospace",
                     weight="bold", color="#3a3a3a", pad=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(False)
        ax.text(0.995, 1.06, unit, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#999", style="italic")
    fig.suptitle("capsule_bench_v0 — run scorecard", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.6)
    fig.canvas.draw()
    for ax in axes:
        p = ax.get_position(); pad = 0.02
        S.card(fig, [p.x0 - pad, p.y0 - pad, p.width + 2 * pad, p.height + 2 * pad])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="raw_baseline/rb_full_01,merlin_assisted/merlin_full_01,"
                    "raw_baseline/rb_pilot_0002,raw_baseline/rb_pilot_cpp_01",
                    help="comma-sep arm/run_id paths under runs/")
    ap.add_argument("--out", default=str(REPORTS / "plots"))
    ap.add_argument("--bg", default="rounds",
                    help="fig1 background mode: rounds | reasoning | turnphase | stream | plain")
    ap.add_argument("--compare-run", default="raw_baseline/rb_full_01",
                    help="run to render the fig1 background-options comparison sheet for")
    a = ap.parse_args(argv)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    runs = []
    for rp in a.runs.split(","):
        rp = rp.strip()
        d = RUNS / rp
        if d.exists():
            runs.append(T.extract_run(d))
        else:
            print(f"  (skip missing {rp})")
    audit = T.load_full_suite(REPORTS)
    made = []
    # background-options comparison sheet (same run, every background mode side by side)
    cmp_run = next((r for r in runs if f"{r['arm']}/{r['run_id']}" == a.compare_run), runs[0] if runs else None)
    if cmp_run:
        try:
            fig1_compare(cmp_run, Path(a.out) / "fig1_compare_backgrounds.png")
            print(f"  wrote {Path(a.out) / 'fig1_compare_backgrounds.png'}")
        except Exception as e:
            import traceback
            print(f"  FAILED compare sheet: {e}\n{traceback.format_exc()}")
    for fn, name in [(lambda o: fig1_activity(runs, o, bg=a.bg), "fig1_activity_trajectory.png"),
                     (lambda o: fig2_planes(runs, o), "fig2_failure_planes.png"),
                     (lambda o: fig3_heatmap(runs, o), "fig3_capsule_heatmap.png"),
                     (lambda o: fig4_summary(audit, runs, o), "fig4_ab_summary.png"),
                     (lambda o: fig5_cards(runs, o), "fig5_scorecard.png")]:
        try:
            made.append(str(fn(outdir / name)))
            print(f"  wrote {outdir / name}")
        except Exception as e:
            import traceback
            print(f"  FAILED {name}: {e}\n{traceback.format_exc()}")
    print(f"\n{len(made)} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
