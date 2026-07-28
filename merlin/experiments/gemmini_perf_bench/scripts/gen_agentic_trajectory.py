"""Baseline-vs-merlin AUTHORING TRAJECTORY — what each AGENT does over its transcript + cumulative spend.

These ARE the two agentic-A/B agents (raw_baseline = regular tools; merlin_assisted = + RTL-compile/checks
tools) — distinct from the perf-bench codegen backends. Reuses _trajectory.extract_run for the per-event
activity timeline + per-round outcome.

Each panel: SMOOTHED stacked activity share (Reading/Writing/Bash/Thinking) over transcript progress, with
a bold CUMULATIVE-SPEND line on the right axis (reconstructed from per-message total-token usage, scaled to
the run's known total cost — output tokens are tiny, so spend tracks input/cache-read), round dividers, and
the run's total $ / tokens annotated for direct A/B comparison. -> reports/fig_agentic_trajectory.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import _pbcommon as PB
import perf_style as S

sys.path.insert(0, str(PB.REPO / "experiments/capsule_bench/targets/gemmini/scripts/plots"))
import _trajectory as T  # noqa: E402

CB_RUNS = PB.REPO / "experiments/capsule_bench/targets/gemmini/runs"
CB_REPORTS = PB.REPO / "artifacts" / "capsule-bench" / "gemmini"
# arm -> (run-dir subdir, panel label). Both merlin arms live under merlin_assisted/ (split by bundle_id).
ARM_META = {
    "baseline":         ("raw_baseline", "BASELINE agent — regular tools (no Merlin)"),
    "merlin":           ("merlin_assisted", "MERLIN agent — + authoring tools (no CIRCT)"),
    "merlin_rtlchecks": ("merlin_assisted", "MERLIN+CIRCT agent — + RTL-derived checks"),
}
PREFER_TAG = "abc1"  # prefer the fresh 3-arm batch; else fall back to any valid run with a transcript


def _select_runs():
    """One representative valid run per arm (prefer the fresh batch), only if it has round transcripts."""
    import json
    ag = json.loads((CB_REPORTS / "agentic_results.json").read_text())
    picks = []
    for arm, (sub, label) in ARM_META.items():
        valids = [r for r in ag["arms"].get(arm, []) if r.get("valid")]
        cands = [r for r in valids if PREFER_TAG in r["run_id"]] or valids
        for r in cands:
            d = CB_RUNS / sub / r["run_id"]
            if (d / "rounds").is_dir() and any((d / "rounds").glob("round_*.transcript.jsonl")):
                picks.append((sub, r["run_id"], label)); break
    return picks
BUCKET = {T.KIND_READ: "Reading", T.KIND_RESULT: "Reading", T.KIND_EDIT: "Writing code",
          T.KIND_BASH: "Running (Bash)", T.KIND_THINK: "Thinking", T.KIND_TEXT: "Thinking"}
ORDER = ["Reading", "Writing code", "Running (Bash)", "Thinking"]
CO = {"Reading": "#6E93B0", "Writing code": "#E6B84C", "Running (Bash)": "#9DB682", "Thinking": "#B49FCC"}
SPEND = "#7A4FA3"  # cumulative-spend line


def _hann_smooth(y, win):
    if win < 3 or len(y) < 3:
        return y
    win = min(win | 1, len(y) | 1)
    k = np.hanning(win); k /= k.sum()
    return np.convolve(np.pad(y, win // 2, mode="edge"), k, mode="valid")[:len(y)]


def _cumulative_spend(run_dir: Path, total_cost: float):
    """Per-message total-token usage (input+cache_creation+cache_read+output) across rounds in order,
    cumulative, scaled so the endpoint == the run's known total cost. Returns (x%-progress, cum_usd)."""
    rdir = run_dir / "rounds"
    toks = []
    if rdir.exists():
        for f in sorted(rdir.glob("round_*.transcript.jsonl")):
            for line in f.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") == "assistant":
                    u = (o.get("message", {}) or {}).get("usage", {}) or {}
                    t = sum(int(u.get(k, 0) or 0) for k in
                            ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                             "output_tokens"))
                    if t:
                        toks.append(t)
    if not toks:
        return None, None
    cum = np.cumsum(toks, dtype=float)
    x = 100.0 * np.arange(len(cum)) / max(len(cum) - 1, 1)
    usd = (total_cost or 0.0) * cum / cum[-1]
    return x, usd


def _panel(ax, run, run_dir, title):
    tl = run["timeline"]
    if not tl:
        ax.text(0.5, 0.5, f"{run['run_id']}\n(no transcript)", ha="center", va="center"); ax.axis("off"); return
    n = len(tl)
    x = 100.0 * np.arange(n) / max(n - 1, 1)
    bucket = np.array([BUCKET.get(e["kind"], "") for e in tl])
    rounds = np.array([e.get("round", 0) for e in tl])
    win = max(7, int(0.08 * n))
    series = {b: _hann_smooth((bucket == b).astype(float), win) for b in ORDER}
    tot = np.sum(list(series.values()), axis=0); tot[tot == 0] = 1.0
    ax.stackplot(x, *[series[b] / tot for b in ORDER], colors=[CO[b] for b in ORDER],
                 labels=ORDER, alpha=0.92, zorder=2)
    ax.set_ylim(0, 1); ax.set_xlim(0, 100); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("activity share", fontsize=9)
    ax.set_title(title, fontsize=12, loc="left", pad=18, fontweight="bold")

    # round dividers + labels + final-pass marker
    bounds = []
    for r in np.unique(rounds):
        xr = x[rounds == r]
        if len(xr):
            bounds.append((int(r), xr.min(), xr.max()))
    for r, lo, hi in bounds:
        ax.axvline(hi, color=S.INK, lw=1.0, alpha=0.45, zorder=4)
        rr = next((q for q in run["rounds"] if q["round"] == r), None)
        tag = f"round {r}"
        if rr and rr.get("n_passed") is not None and rr.get("n_capsules"):
            tag += f"  ({rr['n_passed']}/{rr['n_capsules']}✓)"
        ax.text((lo + hi) / 2, 1.05, tag, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # cumulative spend on the right axis (bold), total annotated
    proc = run.get("process", {}) or {}
    total_cost = proc.get("estimated_cost_usd") or 0.0
    total_tok = proc.get("tokens_total") or 0
    sx, sy = _cumulative_spend(run_dir, total_cost)
    axr = ax.twinx(); axr.set_ylabel("cumulative spend ($)", color=SPEND, fontsize=9)
    axr.tick_params(colors=SPEND, labelsize=8); axr.spines["top"].set_visible(False)
    if sx is not None:
        axr.plot(sx, sy, color=SPEND, lw=3.0, zorder=8, solid_capstyle="round")
        axr.set_ylim(0, max(sy) * 1.15)
        axr.annotate(f"  ${total_cost:.2f} · {total_tok/1e6:.1f}M tok",
                     xy=(sx[-1], sy[-1]), xytext=(-4, 6), textcoords="offset points",
                     ha="right", va="bottom", fontsize=10, fontweight="bold", color=SPEND)
    ax.text(0.995, 1.05, f"{len(bounds)} rounds", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def main():
    S.use_style()
    picks = _select_runs()
    if not picks:
        print("  (no valid runs with transcripts yet — skipping trajectory)"); return 0
    runs = [(T.extract_run(CB_RUNS / arm / rid), CB_RUNS / arm / rid, lab) for arm, rid, lab in picks]
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 3.2 * len(runs) + 0.4), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (run, rdir, lab) in zip(axes, runs):
        _panel(ax, run, rdir, lab)
    axes[-1].set_xlabel("progress through the agent's transcript (%)")
    handles = [Patch(fc=CO[b], label=b) for b in ORDER]
    handles.append(Line2D([], [], color=SPEND, lw=3, label="cumulative spend ($, right axis)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Authoring trajectory & cumulative spend — baseline vs merlin vs merlin+CIRCT (PILOT)",
                 fontsize=15, fontweight="bold", y=1.0)
    totals = []
    for (run, _rd, lab) in runs:
        p = run.get("process", {}) or {}
        short = lab.split(" agent")[0].lower()
        totals.append(f"{short} ${p.get('estimated_cost_usd',0):.2f}/{(p.get('tokens_total') or 0)/1e6:.1f}M ({run['run_id']})")
    S.caption(fig, "Smoothed activity share (Read/Write/Bash/Think) over each agent's transcript; bold line = "
              "cumulative $ (from per-message total-token usage — input+cache dominate; scaled to each run's "
              "known total). Totals: " + " · ".join(totals) + ". Round dividers labelled with capsules-"
              "passing. One representative valid run per arm. These are the AGENTS (authoring), not codegen "
              "backends; same task/model/grading — only authoring aids differ (see ARMS.md).")
    out = PB.REPORTS / "fig_agentic_trajectory.png"
    S.save_fig(fig, out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
