"""Optional matplotlib plots for DSE guidance. No-op (returns False) if matplotlib is absent.

Follows the headless pattern used elsewhere in the repo (``dse/experiment.py``): try-import,
``matplotlib.use("Agg")``, save to disk. CSV/Markdown reports are always written regardless;
plots are a convenience, never a dependency.
"""
from __future__ import annotations

from pathlib import Path

from merlin.dse_guidance.axes import AXIS_FAMILY

_FAMILY_COLOR = {
    "hardware": "#d62728",
    "memory_residency": "#1f77b4",
    "dispatch": "#2ca02c",
    "datapath": "#9467bd",
}


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def axis_triage_plot(triage_result: dict, path: Path) -> bool:
    """Bubble: x=gap_closure, y=cost_tier, size=confidence, color=axis family."""
    plt = _mpl()
    if plt is None:
        return False
    rows = [r for r in triage_result["axes"] if r["gap_closure"] is not None]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        fam = r["family"]
        ax.scatter(r["gap_closure"], r["cost_tier"],
                   s=80 + 320 * float(r["confidence"]),
                   color=_FAMILY_COLOR.get(fam, "#777777"), alpha=0.7, edgecolors="k")
        ax.annotate(r["axis"], (r["gap_closure"], r["cost_tier"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.axhline(3, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("gap_closure (fraction of target gap closed)")
    ax.set_ylabel("cost_tier (1=SW … 5=major RTL)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 5.5)
    ax.invert_yaxis()
    ax.set_title(f"DSE axis triage — {triage_result['workload']} "
                 f"({triage_result['representation']})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def bottleneck_plot(baseline, path: Path) -> bool:
    """Stacked bar of the baseline cost components."""
    plt = _mpl()
    if plt is None:
        return False
    comps = sorted(baseline.components, key=lambda c: -baseline.components[c])
    if not comps:
        return False
    fig, ax = plt.subplots(figsize=(4, 6))
    bottom = 0.0
    for comp in comps:
        ms = baseline.components[comp]
        ax.bar(baseline.workload, ms, bottom=bottom, label=f"{comp} ({baseline.evidence_for(comp)})")
        bottom += ms
    ax.set_ylabel("ms")
    ax.set_title(f"Bottleneck breakdown — {baseline.workload}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def flat_vs_multirate_plot(flat, multirate, path: Path) -> bool:
    """Grouped bar: visible reuse and dispatches/replan, flat vs multi-rate."""
    plt = _mpl()
    if plt is None:
        return False
    metrics = ["visible_weight_reuse", "visible_prefix_kv_reuse", "dispatches_per_replan"]
    flat_vals = [getattr(flat, m) for m in metrics]
    multi_vals = [getattr(multirate, m) for m in metrics]
    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], flat_vals, width=0.4, label="flat")
    ax.bar([i + 0.2 for i in x], multi_vals, width=0.4, label="multirate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=15, fontsize=8)
    ax.set_title(f"Flat vs multi-rate — {flat.workload}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
