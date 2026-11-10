"""Declare evidence-backed plots and prepare their source-bound manifest.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows
from .facts import unified_facts

# --------------------------------------------------------------------------- P13-f plot manifest

# (plot_id, title, source_artifact, required_columns, x, y, series, kind, rec)
_PLOTS = [
    (
        "evidence_type_by_workload",
        "Evidence type by workload",
        "unified_fact_table.csv",
        ["workload", "evidence_type"],
        "workload",
        "count",
        "evidence_type",
        "stacked_bar",
        "backup",
    ),
    (
        "evidence_type_by_phase",
        "Evidence type by analysis phase",
        "unified_fact_table.csv",
        ["source_phase", "evidence_type"],
        "source_phase",
        "count",
        "evidence_type",
        "stacked_bar",
        "backup",
    ),
    (
        "shape_class_mac_share",
        "Shape-class MAC share by workload",
        "shape_summary_by_workload.csv",
        ["workload", "shape_class", "mac_fraction"],
        "workload",
        "mac_fraction",
        "shape_class",
        "stacked_bar",
        "main",
    ),
    (
        "shape_class_opcount_share",
        "Shape-class op-count share by workload",
        "shape_summary_by_workload.csv",
        ["workload", "shape_class", "op_count"],
        "workload",
        "op_count",
        "shape_class",
        "stacked_bar",
        "backup",
    ),
    (
        "primitive_coverage_heatmap",
        "Primitive x workload structural coverage",
        "primitive_coverage_matrix.csv",
        ["primitive", "workload", "coverage_under_10pct"],
        "workload",
        "primitive",
        "coverage_under_10pct",
        "heatmap",
        "main",
    ),
    (
        "primitive_regret_bar",
        "Primitive coverage + max regret",
        "primitive_regret_table.csv",
        ["primitive", "coverage_under_10pct", "max_regret"],
        "primitive",
        "coverage_under_10pct",
        "max_regret",
        "grouped_bar",
        "backup",
    ),
    (
        "abstraction_pressure_bar",
        "Abstraction pressure (workloads supporting)",
        "abstraction_pressure_ranking.csv",
        ["system_abstraction", "n_workloads"],
        "system_abstraction",
        "n_workloads",
        "",
        "bar",
        "backup",
    ),
    (
        "boundary_placement_heatmap",
        "Boundary placement: abstraction x level",
        "hw_sw_boundary_matrix.csv",
        [
            "abstraction",
            "compiler_transform",
            "runtime_hal_object",
            "command_buffer_or_command_isa",
            "accelerator_isa",
            "device_microcode_or_controller",
            "fixed_hardware_datapath",
        ],
        "level",
        "abstraction",
        "status",
        "heatmap",
        "backup",
    ),
    (
        "resident_capacity_by_dtype",
        "Resident capacity by dtype (per region)",
        "data_movement_table.csv",
        ["workload", "region", "resident_int8_B", "resident_bf16_B"],
        "workload_region",
        "bytes",
        "dtype",
        "grouped_bar",
        "backup",
    ),
    (
        "avoidable_reload_by_region",
        "Avoidable weight reload by region",
        "data_movement_table.csv",
        ["workload", "region", "avoidable_weight_reload"],
        "workload_region",
        "avoidable_weight_reload",
        "",
        "bar",
        "main",
    ),
    (
        "measurement_priority_bar",
        "Candidates unblocked per measurement",
        "measurement_priority_table.csv",
        ["measurement", "n_candidates_unblocked"],
        "measurement",
        "n_candidates_unblocked",
        "",
        "bar",
        "main",
    ),
    (
        "critical_path_parallelism",
        "Available parallelism by workload",
        "critical_path_table.csv",
        ["workload", "available_parallelism"],
        "workload",
        "available_parallelism",
        "",
        "bar",
        "main",
    ),
    (
        "epilogue_pattern_counts",
        "Epilogue patterns by workload",
        "epilogue_pattern_table.csv",
        ["workload", "pattern"],
        "workload",
        "count",
        "pattern",
        "stacked_bar",
        "backup",
    ),
    # ---- decision-impact ("what-if") plots: how an outcome changes under a DSE knob choice ----
    (
        "decision_primitive_choice",
        "Decision: single primitive choice -> coverage",
        "primitive_coverage_matrix.csv",
        ["primitive", "workload", "coverage_under_10pct"],
        "primitive",
        "coverage_under_10pct",
        "workload",
        "decision_bar",
        "main",
    ),
    (
        "decision_weight_residency",
        "Decision: weight residency -> bytes moved vs loop count",
        "data_movement_table.csv",
        ["workload", "region", "weight_bytes", "invocations"],
        "loop_count",
        "bytes_moved",
        "policy",
        "decision_curve",
        "main",
    ),
    (
        "decision_capacity_dtype",
        "Decision: on-chip capacity + dtype -> weights resident",
        "dtype_capacity_table.csv",
        ["workload", "bf16_B", "int8_B", "int4_B"],
        "capacity_budget",
        "workloads_resident",
        "dtype",
        "decision_curve",
        "main",
    ),
    (
        "decision_sharding_cost",
        "Decision: shard axis + count -> extra data-movement bytes",
        "sharding_table.csv",
        ["axis", "shardable_2", "shardable_4", "shardable_8", "per_extra_shard_bytes"],
        "shard_count",
        "extra_bytes",
        "axis",
        "decision_bar",
        "main",
    ),
    # ---- P16 decision-frontier & robustness plots ----
    (
        "primitive_set_frontier",
        "Primitive-set frontier (worst vs mean coverage)",
        "tile_waste_table.csv",
        ["workload", "op_index", "primitive", "covered_under_10pct"],
        "mean_coverage",
        "worst_coverage",
        "set",
        "scatter",
        "main",
    ),
    (
        "operator_cumulative_mac",
        "Operator cumulative MAC share (few giant vs many even ops)",
        "operator_shape_table.csv",
        ["workload", "macs"],
        "top_k_ops",
        "cumulative_mac",
        "workload",
        "line",
        "main",
    ),
    (
        "boundary_necessity_matrix",
        "Abstraction necessity (necessary/useful/possible/blocked)",
        "hw_sw_boundary_matrix.csv",
        ["abstraction"],
        "workload",
        "abstraction",
        "necessity_class",
        "categorical",
        "main",
    ),
    (
        "decision_sharding_per_top_op",
        "Decision: shard top-MAC ops -> extra bytes / output bytes",
        "sharding_table.csv",
        ["workload", "op_index", "axis", "per_extra_shard_bytes"],
        "top_op",
        "extra_over_output",
        "axis",
        "decision_bar",
        "main",
    ),
    # ---- P17 adversarial-audit + requirements-envelope plots ----
    (
        "primitive_frontier_by_threshold",
        "Frontier robustness: worst coverage vs set size by threshold",
        "tile_waste_table.csv",
        ["workload", "op_index", "primitive", "covered_under_10pct"],
        "set_size",
        "worst_coverage",
        "threshold",
        "line",
        "main",
    ),
    (
        "macro_vs_micro_primitive_coverage",
        "Macro vs micro vs worst primitive coverage",
        "tile_waste_table.csv",
        ["workload", "op_index", "primitive", "covered_under_10pct"],
        "set_size",
        "coverage",
        "aggregation",
        "line",
        "main",
    ),
    (
        "required_compute_envelope",
        "Required compute envelope (requirement, not measured)",
        "requirements_table.csv",
        ["workload", "region", "requirement", "value"],
        "deadline_ms",
        "required_GMAC_per_s",
        "workload",
        "line",
        "main",
    ),
    (
        "required_memory_movement_envelope",
        "Required memory-movement envelope (residency removes Kx)",
        "requirements_table.csv",
        ["workload", "region", "requirement", "value"],
        "workload",
        "required_weight_B_per_s",
        "residency",
        "grouped_bar",
        "main",
    ),
    (
        "required_command_rate_envelope",
        "Required command-rate envelope (proxy; not measured)",
        "requirements_table.csv",
        ["workload", "region", "requirement", "value"],
        "deadline_ms",
        "required_dispatch_per_s",
        "workload",
        "line",
        "backup",
    ),
    (
        "workload_influence_loo_delta",
        "Workload influence: leave-one-out micro delta",
        "shape_summary_by_workload.csv",
        ["workload", "shape_class", "mac_fraction"],
        "metric",
        "max_loo_micro_delta",
        "",
        "bar",
        "main",
    ),
    # ---- P18 operator-recovery plots ----
    (
        "work_coverage_by_workload",
        "Recovered work: linear-GEMM vs attention MAC mass",
        "work_coverage_table.csv",
        ["workload", "linear_gemm_macs", "attention_macs"],
        "workload",
        "recovered_macs",
        "op_class",
        "stacked_bar",
        "main",
    ),
    (
        "visible_linear_fraction",
        "Visible linear fraction (linear / (linear+attention))",
        "work_coverage_table.csv",
        ["workload", "visible_linear_fraction"],
        "workload",
        "visible_linear_fraction",
        "",
        "bar",
        "main",
    ),
]


_PLOT_CAPTION = {
    "evidence_type_by_workload": "How much of each workload's evidence is IR-recovered vs assumed — "
    "where the search space rests on recovered structure vs reference values.",
    "evidence_type_by_phase": "Evidence provenance per analysis phase (traceability).",
    "shape_class_mac_share": "MAC mass per shape class — which matmul shapes a DSE primitive set must "
    "cover to capture most compute.",
    "shape_class_opcount_share": "Op-count share per shape class (context for the MAC share).",
    "primitive_coverage_heatmap": "Which candidate primitives tile each workload's shapes under 10% "
    "pad waste — the primitive search space, not a performance ranking.",
    "primitive_regret_bar": "Coverage vs worst-case cross-workload regret per primitive — primitives "
    "with high regret are corpus-overfit candidates DSE should treat cautiously.",
    "abstraction_pressure_bar": "How many workloads imply each system abstraction (support breadth).",
    "boundary_placement_heatmap": "At which HW/SW levels each abstraction could sit — the boundary "
    "search space DSE must explore (Merlin enumerates, does not choose).",
    "resident_capacity_by_dtype": "Resident weight bytes per region by dtype — the on-chip capacity "
    "the residency search space is sized against.",
    "avoidable_reload_by_region": "Weight bytes re-read across the K-loop that residency could avoid "
    "— where a residency/packed-store axis has the most to act on.",
    "measurement_priority_bar": "How many blocked candidates each missing input would unblock — what "
    "to capture/measure next, not a result.",
    "critical_path_parallelism": "Inter-op work/span per workload — the unit-multiplicity the "
    "heterogeneity search space could exploit.",
    "epilogue_pattern_counts": "Epilogue/fusion patterns per workload (numerical-contract context).",
    "decision_primitive_choice": "If DSE builds only ONE compute primitive, how much of each "
    "workload's MACs it can tile under 10% waste — the worst-case bar shows no single primitive "
    "covers every workload, so the search space needs both a tile and a GEMV lane.",
    "decision_weight_residency": "Weight bytes moved as the head loop count grows: reload-every-step "
    "(linear) vs keep-resident (flat). The vertical gap at each workload's real K is the "
    "avoidable reload a residency knob removes (bytes, not bandwidth).",
    "decision_capacity_dtype": "How many workloads become fully weight-resident as the on-chip "
    "capacity budget grows, per storage dtype — low-bit dtypes reach full residency at a smaller "
    "budget, quantifying the capacity-vs-dtype trade in the search space.",
    "decision_sharding_cost": "Extra data-movement bytes added by sharding 2/4/8 ways along M, N, or "
    "K: M/N shards are reduction-free (broadcast only) while K shards add partial-sum traffic — "
    "the cost side of the parallelization decision.",
    "primitive_set_frontier": "Each point is a primitive (or best set): x=mean coverage, y=worst-"
    "workload coverage. Upper-right = broadly useful; high-x/low-y = corpus-overfit. The best "
    "single primitive sits low on y; a {tile + GEMV-lane} set reaches the top-right — DSE should "
    "search primitive SETS.",
    "operator_cumulative_mac": "Cumulative MAC share vs top-k operators per workload: a steep curve "
    "(rdt: 1 op = 87%) means DSE sizes for a few giant ops; a gradual curve means many even ops.",
    "boundary_necessity_matrix": "Strict necessity per abstraction × workload "
    "(necessary/useful/possible/blocked/N-A) — what DSE should commit to, not merely what is "
    "possible; low-bit abstractions are blocked by the dequantized capture.",
    "decision_sharding_per_top_op": "For the top-MAC ops, extra sharding bytes normalized by the op's "
    "output bytes, per M/N/K axis — which axis partitions a hot op cheaply (the per-operator view, "
    "not a corpus aggregate).",
    "primitive_frontier_by_threshold": "Worst-workload coverage vs primitive-set size at 5/10/20% pad "
    "waste — whether the 'a 2-set suffices' claim survives threshold perturbation (the specific "
    "pair may shift; structural coverage only).",
    "macro_vs_micro_primitive_coverage": "Macro (equal-weight) vs micro (MAC-weighted) vs worst "
    "coverage as the primitive set grows — the second primitive is where worst-workload coverage "
    "jumps; structural, no performance.",
    "required_compute_envelope": "Required compute rate (= configured-K replan MACs / deadline) vs "
    "replan deadline — a REQUIREMENT a future accelerator must exceed, not a measured rate.",
    "required_memory_movement_envelope": "Required weight bandwidth at a 100 ms deadline, weights "
    "reloaded every step vs kept resident — residency removes a K× bandwidth requirement (the "
    "residency search axis); a requirement, not a measured rate.",
    "required_command_rate_envelope": "Required dispatch rate vs deadline — a PROXY (matmul-count, "
    "~12× undercount), measured only for small_llama; not a hardware command rate.",
    "workload_influence_loo_delta": "Largest leave-one-out micro swing per cross-workload metric — "
    "red bars are metrics whose winner is stable but whose magnitude is not (drop one workload "
    "and the number moves sharply).",
    "work_coverage_by_workload": "Recovered MAC mass split into linear-GEMM vs attention (both from "
    "IR shapes, no config) — attention is NOT erased, just lowered to generic and re-parsed.",
    "visible_linear_fraction": "Fraction of recovered MAC work that is the linear-GEMM geometry this "
    "study analyzes (rest = attention) — answers 'are we analyzing most of the compute?'.",
}


def plot_manifest(cs_dir: Path, scope: str) -> list[dict]:
    out = []
    for pid, title, art, cols, x, y, series, kind, rec in _PLOTS:
        # unified_fact_table.csv is the run's own output (always produced) — treat as available
        present = (cs_dir / art).is_file() or art == "unified_fact_table.csv"
        have_cols = True
        n_rows = 0
        if present and art.endswith(".csv") and (cs_dir / art).is_file():
            hdr = (cs_dir / art).read_text().splitlines()[:1]
            header = hdr[0].split(",") if hdr else []
            have_cols = all(
                c in header for c in cols if c not in ("level", "count", "workload_region", "bytes", "dtype")
            )
            n_rows = len(_csv_rows(cs_dir / art))
        elif art == "unified_fact_table.csv":
            n_rows = len(unified_facts(cs_dir, scope))  # the run's own fact table (B6: non-empty)
        out.append(
            {
                "plot_id": pid,
                "title": title,
                "source_artifact": art,
                "required_columns": cols,
                "x_axis": x,
                "y_axis": y,
                "series": series,
                "plot_type": kind,
                "available": bool(present and have_cols and n_rows > 0),
                "n_source_rows": n_rows,
                "evidence_tier": "B",
                "recommendation": rec if (present and have_cols and n_rows > 0) else "omit",
                "caveat": "structural; axes are counts/bytes/fractions, not a performance metric",
                "dse_caption": _PLOT_CAPTION.get(pid, ""),
                "why_useful": f"shows {title.lower()} from {art}",
            }
        )
    return out


def _plot_md(plots, scope, rendered) -> str:
    L = [
        f"# Plot manifest ({scope})\n",
        "> Candidate plots derived from the data. Structural axes only (counts/bytes/fractions). "
        f"Rendered PNGs: {len(rendered)} under generated_plots/.\n",
        "| plot_id | title | type | source | rec | rendered |",
        "|---|---|---|---|---|---|",
    ]
    for p in plots:
        L.append(
            f"| {p['plot_id']} | {p['title']} | {p['plot_type']} | {p['source_artifact']} | "
            f"{p['recommendation']} | {'yes' if p['plot_id'] in rendered else 'no'} |"
        )
    return "\n".join(L) + "\n"


def _plots_index_md(plots, scope, rendered, folder) -> str:
    L = [
        f"# Presentation plots index ({scope})\n",
        "> Each rendered plot with its one-sentence DSE-search-space implication. Structural axes "
        "only (counts / bytes / fractions) — none is a performance metric.\n",
    ]
    for p in plots:
        if p["plot_id"] not in rendered:
            continue
        L.append(f"### {p['title']}\n")
        L.append(f"![{p['plot_id']}]({folder}/{p['plot_id']}.png)\n")
        L.append(f"**DSE implication:** {p.get('dse_caption') or '—'}\n")
    return "\n".join(L) + "\n"
