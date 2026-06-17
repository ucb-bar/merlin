"""Neutral PNG renderers for the insight-mining plot manifest.

Renders the plot-manifest entries that have a known renderer + present source columns into PNGs
under a run's ``generated_plots/``. Styling is neutral and every axis is a structural quantity
(counts / bytes / fractions / evidence tiers) — nothing here plots latency, throughput, or any
performance/speedup quantity. Output lands in the non-committed ``results/`` run folder, so PNG
determinism is not required (the byte-stable guarantee is on the committed case_study, not here).

If matplotlib is unavailable the renderer no-ops and returns ``[]`` (the manifest is still emitted).
"""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from pathlib import Path


def _have_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception:
        return False


def _rows(p: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(p.read_text()))) if p.is_file() else []


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (8, 4.5), "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)


# each renderer: (cs_dir, facts, ax) -> bool drew_something

def _stacked_count(ax, facts, key, series_key, title):
    groups = defaultdict(Counter)
    for f in facts:
        groups[f[key]][f[series_key]] += 1
    cats = sorted(groups)
    series = sorted({s for c in groups.values() for s in c})
    bottoms = [0] * len(cats)
    for s in series:
        vals = [groups[c].get(s, 0) for c in cats]
        ax.bar(cats, vals, bottom=bottoms, label=s)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title(title)
    ax.set_ylabel("fact count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=6, ncol=2)
    return True


def _r_evidence_by_workload(cs, facts, ax):
    return _stacked_count(ax, facts, "workload", "evidence_type", "Evidence type by workload")


def _r_evidence_by_phase(cs, facts, ax):
    return _stacked_count(ax, facts, "source_phase", "evidence_type", "Evidence type by phase")


def _r_shape_mac_share(cs, facts, ax):
    rows = _rows(cs / "shape_summary_by_workload.csv")
    if not rows:
        return False
    g = defaultdict(dict)
    for r in rows:
        g[r["workload"]][r["shape_class"]] = float(r["mac_fraction"])
    wls = sorted(g)
    classes = sorted({c for d in g.values() for c in d})
    bottoms = [0.0] * len(wls)
    for c in classes:
        vals = [g[w].get(c, 0.0) for w in wls]
        ax.bar(wls, vals, bottom=bottoms, label=c)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Shape-class MAC share by workload")
    ax.set_ylabel("MAC fraction")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=6)
    return True


def _r_primitive_coverage(cs, facts, ax):
    rows = _rows(cs / "primitive_coverage_matrix.csv")
    if not rows:
        return False
    prims = sorted({r["primitive"] for r in rows})
    wls = sorted({r["workload"] for r in rows})
    mat = [[0.0] * len(wls) for _ in prims]
    idx = {p: i for i, p in enumerate(prims)}
    widx = {w: j for j, w in enumerate(wls)}
    for r in rows:
        mat[idx[r["primitive"]]][widx[r["workload"]]] = float(r["coverage_under_10pct"])
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(wls)), wls, rotation=20)
    ax.set_yticks(range(len(prims)), prims, fontsize=7)
    ax.set_title("Primitive x workload coverage (<=10% pad waste)")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return True


def _r_primitive_regret(cs, facts, ax):
    rows = _rows(cs / "primitive_regret_table.csv")
    if not rows:
        return False
    prims = [r["primitive"] for r in rows]
    x = range(len(prims))
    ax.bar([i - 0.2 for i in x], [float(r["coverage_under_10pct"]) for r in rows], 0.4,
           label="coverage <=10%")
    ax.bar([i + 0.2 for i in x], [float(r["max_regret"]) for r in rows], 0.4, label="max regret")
    ax.set_xticks(list(x), prims, rotation=35, fontsize=7)
    ax.set_title("Primitive coverage + cross-workload regret")
    ax.set_ylabel("MAC fraction")
    ax.legend(fontsize=7)
    return True


def _r_boundary_heatmap(cs, facts, ax):
    rows = _rows(cs / "hw_sw_boundary_matrix.csv")
    if not rows:
        return False
    levels = ["compiler_transform", "runtime_hal_object", "command_buffer_or_command_isa",
              "accelerator_isa", "device_microcode_or_controller", "fixed_hardware_datapath"]
    score = {"strong_candidate": 4, "possible": 3, "weak_candidate": 2, "blocked": 1,
             "not_applicable": 0, "unavailable": 0}
    rows = sorted(rows, key=lambda r: -int(r["boundary_pressure_score"]))[:14]
    abst = [r["abstraction"] for r in rows]
    mat = [[score.get(r[lv], 0) for lv in levels] for r in rows]
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=4)
    ax.set_xticks(range(len(levels)), [lv.replace("_", "\n") for lv in levels], fontsize=6)
    ax.set_yticks(range(len(abst)), abst, fontsize=6)
    ax.set_title("Boundary placement: abstraction x level (status)")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return True


def _r_resident_capacity(cs, facts, ax):
    rows = _rows(cs / "data_movement_table.csv")
    if not rows:
        return False
    labels = [f"{r['workload']}/{r['region']}" for r in rows]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [int(r["resident_int8_B"]) for r in rows], 0.4, label="int8")
    ax.bar([i + 0.2 for i in x], [int(r["resident_bf16_B"]) for r in rows], 0.4, label="bf16")
    ax.set_xticks(list(x), labels, rotation=30, fontsize=6)
    ax.set_title("Resident weight capacity by dtype (per region)")
    ax.set_ylabel("bytes")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    return True


def _r_avoidable_reload(cs, facts, ax):
    rows = _rows(cs / "data_movement_table.csv")
    if not rows:
        return False
    labels = [f"{r['workload']}/{r['region']}" for r in rows]
    ax.bar(labels, [int(r["avoidable_weight_reload"]) for r in rows])
    ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=6)
    ax.set_title("Avoidable weight reload by region")
    ax.set_ylabel("bytes")
    ax.set_yscale("log")
    return True


def _r_measurement_priority(cs, facts, ax):
    rows = _rows(cs / "measurement_priority_table.csv")
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: -int(r["n_candidates_unblocked"]))
    ax.barh([r["measurement"][:34] for r in rows][::-1],
            [int(r["n_candidates_unblocked"]) for r in rows][::-1])
    ax.set_title("Candidates unblocked per measurement")
    ax.set_xlabel("candidates")
    ax.tick_params(axis="y", labelsize=6)
    return True


def _r_critical_path(cs, facts, ax):
    rows = _rows(cs / "critical_path_table.csv")
    if not rows:
        return False
    ax.bar([r["workload"] for r in rows], [float(r["available_parallelism"]) for r in rows])
    ax.set_title("Available inter-op parallelism (work/span)")
    ax.set_ylabel("work / span")
    ax.tick_params(axis="x", rotation=20)
    return True


_RENDERERS = {
    "evidence_type_by_workload": _r_evidence_by_workload,
    "evidence_type_by_phase": _r_evidence_by_phase,
    "shape_class_mac_share": _r_shape_mac_share,
    "primitive_coverage_heatmap": _r_primitive_coverage,
    "primitive_regret_bar": _r_primitive_regret,
    "boundary_placement_heatmap": _r_boundary_heatmap,
    "resident_capacity_by_dtype": _r_resident_capacity,
    "avoidable_reload_by_region": _r_avoidable_reload,
    "measurement_priority_bar": _r_measurement_priority,
    "critical_path_parallelism": _r_critical_path,
}


def render_plots(plot_manifest, cs_dir, facts, out_dir) -> list[str]:
    """Render every manifest plot that has a renderer; return the list of rendered plot_ids."""
    if not _have_mpl():
        return []
    import matplotlib.pyplot as plt
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cs_dir = Path(cs_dir)
    rendered = []
    for p in plot_manifest:
        r = _RENDERERS.get(p["plot_id"])
        if r is None or p.get("recommendation") == "omit":
            continue
        fig, ax = plt.subplots()
        try:
            ok = r(cs_dir, facts, ax)
        except Exception:
            ok = False
        if ok:
            _save(fig, out_dir / f"{p['plot_id']}.png")
            rendered.append(p["plot_id"])
        else:
            plt.close(fig)
    return rendered
