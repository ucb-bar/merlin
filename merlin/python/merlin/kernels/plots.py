"""Evaluation plots: each figure answers one "does this result make sense?" question.

1. ``motif_source_heatmap`` — is the cross-source signal real? (fractions, not raw counts,
   since sources differ 100x in size)
2. ``motif_prevalence`` — how common is each motif, and how broadly attested?
3. ``promotion_funnel`` — how hard does the ladder filter observations into policies?
4. ``reuse_distribution`` — does the *measured* L2 reuse metric look sane per source?
5. ``dispatch_scatter`` — is the L7 runtime candidate visually justified?
6. ``motif_cooccurrence`` — which decisions travel together (justifies composites)?
7. ``motif_op_heatmap`` — does any marker over-fire on op families where it makes no sense?

All figures are static PNGs under ``plots/`` next to the report, generated only with
``--plots`` and skipped gracefully when matplotlib is missing (``pip install -e
.[kernels-plots]``). They visualize evidence *frequency* — never measured speedup.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path

from merlin.kernels.policy import MotifStat, PromotionResult

log = logging.getLogger(__name__)


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover - exercised via generate_all warning path
        log.warning("plots requested but matplotlib unavailable (%s); skipping. "
                    "Install with `pip install -e .[kernels-plots]`.", e)
        return None


def _motif_order(stats: dict[str, MotifStat]) -> list[str]:
    return sorted(stats, key=lambda m: -stats[m].kernel_count)


def _save(fig, out_dir: Path, name: str, paths: list[Path]) -> None:
    p = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    paths.append(p)


def _heatmap(plt, ax, matrix, xlabels, ylabels, title, fmt="{:.2f}"):
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    vmax = max((max(row) for row in matrix if row), default=1) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v > 0:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 * vmax else "black")
    ax.set_title(title, fontsize=10)
    return im


def plot_motif_source_heatmap(plt, records, stats, out_dir, paths):
    per_source = collections.Counter(r.get("source", "?") for r in records)
    per_ms = collections.Counter()
    for r in records:
        for m in (r.get("evidence", {}) or {}).get("motifs", []):
            per_ms[(m, r.get("source", "?"))] += 1
    motifs, sources = _motif_order(stats), sorted(per_source)
    matrix = [[per_ms[(m, s)] / per_source[s] for s in sources] for m in motifs]
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(sources), 0.7 + 0.4 * len(motifs)))
    _heatmap(plt, ax, matrix, sources, motifs,
             "Motif × source — fraction of each source's kernels")
    _save(fig, out_dir, "motif_source_heatmap", paths)


def plot_motif_prevalence(plt, records, stats, min_kernels, out_dir, paths):
    motifs = _motif_order(stats)[::-1]
    total = max(len(records), 1)
    fracs = [stats[m].kernel_count / total for m in motifs]
    nsrc = [len(stats[m].sources) for m in motifs]
    cmap = {1: "#c6dbef", 2: "#6baed6", 3: "#2171b5"}
    colors = [cmap.get(min(n, 3), "#08306b") if n < 4 else "#08306b" for n in nsrc]
    fig, ax = plt.subplots(figsize=(7, 0.7 + 0.35 * len(motifs)))
    bars = ax.barh(motifs, fracs, color=colors)
    for b, m in zip(bars, motifs):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{stats[m].kernel_count} ({len(stats[m].sources)} src)",
                va="center", fontsize=7)
    ax.axvline(min_kernels / total, color="red", ls="--", lw=0.8,
               label=f"promotion gate (≥{min_kernels} kernels)")
    ax.set_xlabel("fraction of corpus")
    ax.set_title("Motif prevalence (color = #sources; gate also clears at ≥2 sources)",
                 fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "motif_prevalence", paths)


def plot_promotion_funnel(plt, records, stats, promo, validation, out_dir, paths):
    n_validated = sum(
        1 for name, info in (validation or {}).items()
        if any(st == "holds" for st in info.get("workloads", {}).values()))
    stages = [
        (f"kernels ({len(records)})", len(records)),
        (f"kernels w/ ≥1 motif ({sum(1 for r in records if (r.get('evidence') or {}).get('motifs'))})",
         sum(1 for r in records if (r.get("evidence") or {}).get("motifs"))),
        (f"motifs ({len(stats)})", len(stats)),
        (f"promoted motifs ({len(promo.promoted)})", len(promo.promoted)),
        (f"policy rules ({len(promo.rules)})", len(promo.rules)),
        (f"Stage-D validated ({n_validated})", n_validated),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = range(len(stages))
    vals = [max(s[1], 1) for s in stages]
    ax.bar(xs, vals, color="#4292c6", log=True)
    ax.set_xticks(list(xs), [s[0] for s in stages], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("count (log)")
    ax.set_title("Promotion funnel: observations → motifs → policies → validated", fontsize=10)
    _save(fig, out_dir, "promotion_funnel", paths)


def plot_reuse_distribution(plt, records, out_dir, paths):
    by_source: dict[str, list[int]] = collections.defaultdict(list)
    for r in records:
        rc = ((r.get("features", {}) or {}).get("memory_behavior", {})
              .get("rhs", {}).get("reuse_count"))
        if rc and rc > 0:
            by_source[r.get("source", "?")].append(rc)
    if not by_source:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for src, vals in sorted(by_source.items()):
        ax.hist(vals, bins=range(1, max(max(vals) + 2, 10)), alpha=0.55,
                label=f"{src} (n={len(vals)})")
    ax.set_xlabel("measured rhs reuse count (RVV: register-block MR; "
                  "Gemmini: compute per weight load)")
    ax.set_ylabel("kernels")
    ax.set_title("Measured RHS reuse — the L2 evidence behind resident_packed_tensor",
                 fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "reuse_distribution", paths)


def plot_dispatch_scatter(plt, records, out_dir, paths):
    pts = [(dm["n_dispatches"], dm["small_dispatch_fraction"], r.get("source", "?"))
           for r in records
           if (dm := (r.get("features", {}) or {}).get("dispatch_metrics"))
           and dm.get("n_dispatches")]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for src in sorted({p[2] for p in pts}):
        xs = [p[0] for p in pts if p[2] == src]
        ys = [p[1] for p in pts if p[2] == src]
        ax.scatter(xs, ys, s=6, alpha=0.4, label=f"{src} (n={len(xs)})")
    ax.axvline(20, color="red", ls="--", lw=0.8)
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("dispatches per kernel (log)")
    ax.set_ylabel("small-dispatch fraction")
    ax.set_title("L7 dispatch metrics — upper-right quadrant is the\n"
                 "many_small_dispatches motif behind command_buffer_batching", fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "dispatch_scatter", paths)


def plot_motif_cooccurrence(plt, records, stats, out_dir, paths):
    motifs = _motif_order(stats)
    sets: dict[str, set[int]] = {m: set() for m in motifs}
    for i, r in enumerate(records):
        for m in (r.get("evidence", {}) or {}).get("motifs", []):
            sets[m].add(i)
    matrix = []
    for a in motifs:
        row = []
        for b in motifs:
            union = len(sets[a] | sets[b])
            row.append(len(sets[a] & sets[b]) / union if union else 0.0)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(1.5 + 0.45 * len(motifs), 1.0 + 0.42 * len(motifs)))
    _heatmap(plt, ax, matrix, motifs, motifs,
             "Motif co-occurrence (Jaccard) — decisions that travel together")
    _save(fig, out_dir, "motif_cooccurrence", paths)


def plot_motif_op_heatmap(plt, records, stats, out_dir, paths):
    op_counts = collections.Counter(r.get("op", "?") for r in records)
    ops = [op for op, _ in op_counts.most_common(12)]
    per_mo = collections.Counter()
    for r in records:
        for m in (r.get("evidence", {}) or {}).get("motifs", []):
            per_mo[(m, r.get("op", "?"))] += 1
    motifs = _motif_order(stats)
    matrix = [[per_mo[(m, op)] / op_counts[op] for op in ops] for m in motifs]
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * len(ops), 0.8 + 0.4 * len(motifs)))
    _heatmap(plt, ax, matrix, ops, motifs,
             "Motif × op family — sanity: reuse motifs should not fire on elementwise ops")
    _save(fig, out_dir, "motif_op_heatmap", paths)


def generate_all(records: list[dict], stats: dict[str, MotifStat], promo: PromotionResult,
                 validation: dict | None, out_dir: Path,
                 min_kernels: int = 10) -> list[Path]:
    """Write every plot into ``out_dir``; returns the paths written (possibly empty)."""
    plt = _matplotlib()
    if plt is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plotters = (
        lambda: plot_motif_source_heatmap(plt, records, stats, out_dir, paths),
        lambda: plot_motif_prevalence(plt, records, stats, min_kernels, out_dir, paths),
        lambda: plot_promotion_funnel(plt, records, stats, promo, validation, out_dir, paths),
        lambda: plot_reuse_distribution(plt, records, out_dir, paths),
        lambda: plot_dispatch_scatter(plt, records, out_dir, paths),
        lambda: plot_motif_cooccurrence(plt, records, stats, out_dir, paths),
        lambda: plot_motif_op_heatmap(plt, records, stats, out_dir, paths),
    )
    for fn in plotters:
        try:
            fn()
        except Exception as e:  # one bad plot must not kill the extract run
            log.warning("plot failed: %s", e)
    plt.close("all")
    return paths
