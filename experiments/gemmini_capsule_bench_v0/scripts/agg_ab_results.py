"""Aggregate the N>=3 A/B/C experiment (arm × condition × repeats) with error bars.

The N=1 abc4 gave point estimates; this aggregates the `--repeats N` runs from launch_ab_batch into
mean ± std per (arm, condition) cell so the magnitude claims (cost / wall / rounds / sim-runs-skipped /
25-pass) have the dispersion the comparison needs. Honest about N: each cell records n_valid; cells with
n<2 print mean only (std undefined) and are flagged.

Two axes:
  • arm        — baseline (C++) · merlin (xDSL) · merlin_rtlchecks (xDSL+CIRCT)
  • condition  — kernels (hwbringup + example kernels) vs no-kernels (RTL+ISA+README only); the run-id
                 carries `_nk` for the no-kernels cell (set by launch_ab_batch --condition).

Reuses agg_agentic_results.load_run (cost/tokens/rounds/fullsuite) and additionally reads, per run:
  • timing_detailed.json  -> think+gen vs tool/wait split, CIRCT sims_skipped/sims_run
  • full_suite_audit.json  -> passed X/25 (completeness)

-> reports/ab_results.json (+ reports/figs/fig_ab_*.png with error bars). Reads on-disk artifacts only.
Usage: agg_ab_results.py [--tag abc5]   (tag filters run-ids; default = all tagged runs found)
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import yaml

import agg_agentic_results as AAR  # reuse arm detection + per-run loader

EXP = Path("/scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0")
ARM_ORDER = ["baseline", "merlin", "merlin_rtlchecks"]
ARM_LABEL = {"baseline": "baseline (C++)", "merlin": "merlin (xDSL)",
             "merlin_rtlchecks": "merlin+CIRCT"}
COND_ORDER = ["kernels", "no-kernels"]


def _condition_of(run_id: str) -> str:
    # launch_ab_batch tags the no-kernels cell with `_nk` (e.g. merlincirct_abc5_nk_r2)
    return "no-kernels" if "_nk" in run_id else "kernels"


def _timing(d: Path) -> dict:
    t = d / "timing_detailed.json"
    if not t.is_file():
        return {}
    try:
        return json.loads(t.read_text())
    except Exception:
        return {}


def _stat(vals: list[float]) -> dict:
    xs = [v for v in vals if isinstance(v, (int, float))]
    if not xs:
        return {"mean": None, "std": None, "n": 0, "values": []}
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(xs), "values": xs}


# metrics aggregated per cell: key -> (extractor(run_dict, timing_dict), label, unit)
def _passed(r):
    fs = r.get("fullsuite") or {}
    return (fs.get("all") or {}).get("passed")


METRICS = {
    "cost_usd":     (lambda r, t: r.get("cost_usd"),                      "cost", "$"),
    "wall_s":       (lambda r, t: (r.get("wall_s") or 0) / 60.0,          "active wall", "min"),
    "n_rounds":     (lambda r, t: r.get("n_rounds"),                      "rounds", "rounds"),
    "passed":       (lambda r, t: _passed(r),                            "capsules passed", "/25"),
    "think_pct":    (lambda r, t: t.get("think_pct"),                     "think+gen share", "%"),
    "sims_skipped": (lambda r, t: (t.get("circt_gate") or {}).get("sims_skipped"), "CIRCT sims skipped", "#"),
    "sims_run":     (lambda r, t: (t.get("circt_gate") or {}).get("sims_run"),     "sims actually run", "#"),
}


def collect(tag: str | None) -> dict:
    fa = EXP / "reports/full_suite_audit.json"
    audit = json.loads(fa.read_text()) if fa.is_file() else {}
    # cells[(arm, cond)] = list of per-run records
    cells: dict[tuple, list] = {(a, c): [] for a in ARM_ORDER for c in COND_ORDER}
    for sub in AAR.RUN_DIRS:
        base = EXP / "runs" / sub
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if tag and tag not in d.name:
                continue
            arm = AAR._arm_of(d)
            if arm is None:
                continue
            r = AAR.load_run(d, audit)
            if not r:
                continue
            cond = _condition_of(d.name)
            r["_timing"] = _timing(d)
            cells[(arm, cond)].append(r)
    return cells


def aggregate(cells: dict) -> dict:
    out = {"arm_order": ARM_ORDER, "cond_order": COND_ORDER, "cells": {}, "metrics": list(METRICS)}
    for (arm, cond), runs in cells.items():
        valid = [r for r in runs if r.get("valid")]
        cell = {"n_runs": len(runs), "n_valid": len(valid), "run_ids": [r["run_id"] for r in runs],
                "metrics": {}}
        for mk, (fn, label, unit) in METRICS.items():
            cell["metrics"][mk] = {**_stat([fn(r, r.get("_timing", {})) for r in valid]),
                                   "label": label, "unit": unit}
        out["cells"][f"{arm}|{cond}"] = cell
    return out


def plot(agg: dict, outdir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}); JSON written, skipping figs.")
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {"baseline": "#7a7a7a", "merlin": "#4878a8", "merlin_rtlchecks": "#3a8a5a"}
    # one grouped bar chart per metric: x = condition, grouped bars = arm, error bar = std
    for mk in ("cost_usd", "wall_s", "n_rounds", "sims_skipped"):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        width = 0.25
        xs = range(len(COND_ORDER))
        any_data = False
        for i, arm in enumerate(ARM_ORDER):
            means, errs = [], []
            for cond in COND_ORDER:
                m = agg["cells"][f"{arm}|{cond}"]["metrics"][mk]
                means.append(m["mean"] if m["mean"] is not None else 0)
                errs.append(m["std"] if (m["std"] is not None and m["n"] > 1) else 0)
                any_data = any_data or (m["mean"] is not None)
            offs = [x + (i - 1) * width for x in xs]
            ax.bar(offs, means, width, yerr=errs, capsize=4, label=ARM_LABEL[arm],
                   color=colors[arm], edgecolor="white")
        unit = next(v for k, v in [(mk, METRICS[mk][2])])
        label = METRICS[mk][1]
        ax.set_xticks(list(xs)); ax.set_xticklabels(COND_ORDER)
        ax.set_ylabel(f"{label} ({unit})")
        n_per = agg["cells"][f"{ARM_ORDER[0]}|kernels"]["n_valid"]
        ax.set_title(f"{label} by arm × condition  (mean ± std, error bars)")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = outdir / f"fig_ab_{mk}.png"
        if any_data:
            fig.savefig(p, dpi=130); written.append(p)
        plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="filter run-ids by tag substring (e.g. abc5); default=all")
    a = ap.parse_args(argv)
    cells = collect(a.tag)
    agg = aggregate(cells)
    agg["tag_filter"] = a.tag
    p = EXP / "reports/ab_results.json"
    p.write_text(json.dumps(agg, indent=2))
    print(f"wrote {p}")
    figs = plot(agg, EXP / "reports/figs")
    for f in figs:
        print(f"  fig: {f}")
    # console summary
    for arm in ARM_ORDER:
        for cond in COND_ORDER:
            c = agg["cells"][f"{arm}|{cond}"]
            if not c["n_runs"]:
                continue
            cm = c["metrics"]
            def fmt(mk):
                m = cm[mk]
                if m["mean"] is None:
                    return "—"
                s = f"{m['mean']:.1f}" + (f"±{m['std']:.1f}" if m["n"] > 1 else "")
                return s
            print(f"  {arm:16s} [{cond:10s}] n={c['n_valid']}/{c['n_runs']}  "
                  f"${fmt('cost_usd')}  {fmt('wall_s')}min  {fmt('n_rounds')}rd  "
                  f"{fmt('passed')}/25  skips={fmt('sims_skipped')}")
    n_total = sum(c["n_valid"] for c in agg["cells"].values())
    if n_total < 6:
        print(f"\n  ⚠ only {n_total} valid runs across all cells — N<2 per cell means std is undefined; "
              f"run launch_ab_batch with --repeats>=3 before quoting magnitudes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
