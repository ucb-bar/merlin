#!/usr/bin/env python3
"""Generate plot_inventory.{csv,md} + final_plot_classification.{csv,md} for the P26 pass.

Reproducible: the curated per-plot metadata lives in PLOTS below; running this emits the four docs.
Source artifacts / chart types match presentation_plots.py `_RENDERERS`; class follows the 8-rule
final-main test in the approved plan. Scale ∈ {captured-config, deployment-composition, structural,
proxy, QA}. risk ∈ {low, medium, high}.
"""
from __future__ import annotations
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# plot_id: (source, tier, scale, risk, use, dse_question, claim, caveat, restyled)
PLOTS = {
 "table_capture_summary": ("loop_aware_contract.csv", "A", "structural", "low", "main",
   "What does Merlin actually recover from a capture?",
   "Merlin recovers a loop/region/state contract (K, repeated ops, carried state, KV), not just op shapes.",
   "structural facts; no performance implied", True),
 "capture_fidelity": ("IM.capture_fidelity", "A/B", "structural", "low", "main",
   "Which DSE axes does the capture enable vs block?",
   "Recovered shapes/roles/K/KV vs erased low-bit vs not-claimed latency — fidelity gates the axes.",
   "'blocked/erased' is a capture limit, not a workload property", True),
 "capture_level_ablation": ("IM.capture_level_ablation", "A", "structural", "low", "main",
   "What does each capture level unlock?",
   "flat→high_level→quant_qdq progressively names attention/norm then low-bit.",
   "counts of named ops; structural", True),
 "primitive_set_frontier": ("IM.primitive_set_frontier", "A", "structural", "low", "main",
   "Is one tiling primitive enough across workloads?",
   "One primitive is not robust; a 2-set reaches near-full worst-case coverage.",
   "structural pad-waste coverage; not hardware performance", True),
 "operator_cumulative_mac": ("operator_shape_table.csv", "A", "structural", "low", "main",
   "Is compute hot-op-dominated or diffuse?",
   "Workloads split into hot-op-dominated (few giant ops) vs diffuse regimes.",
   "cumulative MAC share from IR shapes", True),
 "decision_weight_residency": ("data_movement_table.csv", "A", "captured-config", "medium", "main",
   "How does weight traffic scale with the loop?",
   "Non-resident weight traffic grows ×K; residency loads once; dot = the model's IR K.",
   "weight BYTES MOVED (not bandwidth); captured-config scale", True),
 "decision_capacity_dtype": ("dtype_capacity_table.csv", "A", "captured-config", "low", "main",
   "When do repeated-head weights fit on-chip?",
   "dtype sets the residency budget: int4/int8 fit more workloads at a given on-chip capacity.",
   "captured-config weight sizes; feasibility envelope", True),
 "realtime_requirement": ("realtime_requirement.csv", "A/B", "deployment-composition", "medium", "main",
   "What weight bandwidth must a machine provide for 30 Hz?",
   "Residency lowers the required weight bandwidth to hit the target rate.",
   "requirement floor under the workload model; not a chip measurement", True),
 "lever_ablation": ("arithmetic_intensity.csv", "A/B", "deployment-composition", "medium", "main",
   "How do system levers lower the 30 Hz requirement?",
   "Action chunking (/H) then residency (/K) each cut the required weight bandwidth.",
   "requirement reduction (H,K from source/config); not a speedup", True),
 "boundary_necessity_matrix": ("IM.abstraction_necessity", "B", "structural", "low", "main",
   "Which abstractions should DSE search first?",
   "A few abstractions are necessary across all workloads; others are workload-specific or blocked.",
   "'blocked' = capture/evidence blocked; 'possible' = not discriminating", True),
 "arithmetic_intensity_roofline": ("arithmetic_intensity.csv", "A/B", "deployment-composition", "high", "main",
   "How does residency shift intensity across machine-balance regimes?",
   "Residency raises weight-stream AI, shifting workloads across a hypothetical balance band.",
   "weight-stream AI; hypothetical balance; modeling view, not measured performance, not full-memory AI", True),
 "visible_linear_fraction": ("work_coverage_table.csv", "A", "structural", "low", "main",
   "How much recovered work is linear-GEMM vs attention?",
   "Most workloads are GEMM-geometry dominated; smolVLA is attention-heavy.",
   "linear/(linear+attention); excludes erased/unmodeled work", True),
 # ---- backup ----
 "work_coverage_by_workload": ("work_coverage_table.csv", "A", "captured-config", "medium", "backup",
   "Linear-GEMM vs attention MAC mass per workload?",
   "Per-workload split of recovered linear vs attention MACs.",
   "captured-config; not deployment scale", True),
 "deployment_magnitude": ("real_config_magnitudes.csv", "B", "deployment-composition", "low", "backup",
   "Deployment params/MACs scale per workload?",
   "Deployment magnitudes by config-composition (embed + per-layer × real n_layers).",
   "deployment-composition; small_llama (synthetic) omitted", True),
 "sharding_scalability": ("sharding_table.csv", "A", "structural", "low", "backup",
   "What is the transfer cost of parallelism?",
   "Extra comm bytes per unit output rise with PU count; M broadcast > K reduction > N.",
   "structural communication bytes; not a performance result", True),
 "sharding_comm_tradeoff": ("sharding_table.csv", "A", "captured-config", "low", "backup",
   "Absolute comm cost vs shard count?",
   "Absolute extra communication (GB) by axis × shard count.",
   "structural bytes; not performance", False),
 "decision_sharding_per_top_op": ("operator_shape_table.csv + sharding_table.csv", "A", "captured-config", "medium", "backup",
   "Which top-MAC ops are expensive to shard?",
   "Sharding extra bytes for the top-MAC ops, normalized by output bytes.",
   "captured-config bytes", False),
 "shape_class_mac_share": ("shape_summary_by_workload.csv", "A", "captured-config", "medium", "backup",
   "MAC distribution across GEMM shape classes?",
   "Per-workload MAC split by shape class (square/skinny/...).",
   "captured-config", False),
 "primitive_frontier_by_threshold": ("IM.primitive_frontier_robustness", "A", "structural", "low", "backup",
   "Is the primitive-set frontier robust to the pad-waste threshold?",
   "Worst-workload coverage vs set size across 5/10/20% thresholds.",
   "structural coverage", False),
 "macro_vs_micro_primitive_coverage": ("IM.primitive_frontier_robustness", "A", "structural", "low", "backup",
   "Macro vs micro vs worst coverage by set size?",
   "Coverage measured three ways across set size.",
   "structural coverage", False),
 "resident_capacity_by_dtype": ("data_movement_table.csv", "A", "captured-config", "medium", "backup",
   "Resident weight bytes per region by dtype?",
   "Resident weight bytes per region, int8 vs bf16.",
   "captured-config scale", False),
 "required_compute_envelope": ("IM.timing_requirement_envelope", "A/C", "captured-config", "medium", "backup",
   "Required compute vs replan deadline?",
   "Required GMAC/s as a function of deadline (a requirement).",
   "requirement, not measured performance; configured K", False),
 "required_memory_movement_envelope": ("IM.timing_requirement_envelope", "A/C", "captured-config", "medium", "backup",
   "Required weight bandwidth @100ms, resident vs reload?",
   "Residency removes a K× factor from the required weight bandwidth.",
   "requirement floor; not a chip", False),
 "critical_path_parallelism": ("critical_path_table.csv", "A", "structural", "low", "backup",
   "How much inter-op parallelism is available?",
   "work/span per workload (low ⇒ mostly serial chains; parallelism is intra-op).",
   "structural dependency graph", False),
 "table_deployment_magnitudes": ("real_config_magnitudes.csv", "B", "deployment-composition", "low", "backup",
   "Exact deployment magnitudes?",
   "Layers, params, MACs/token by config-composition; openVLA/tiny_llama are exact anchors.",
   "deployment-composition", True),
 "table_arithmetic_intensity": ("arithmetic_intensity.csv", "A/B", "deployment-composition", "medium", "backup",
   "Exact arithmetic intensity + residency gain?",
   "Weight-stream AI resident vs reload + residency gain per workload.",
   "weight-stream (modeling) AI, not full-memory; not measured performance", True),
 "table_low_bit_tiers": ("low_bit_visibility.csv", "A/B", "structural", "low", "backup",
   "Low-bit tier per workload?",
   "native/qdq_int8/dequant_only per workload; int8 ratified by the measured gate.",
   "fp8/int4 never assumed", True),
 "table_realtime_requirement": ("realtime_requirement.csv", "A/B", "deployment-composition", "medium", "backup",
   "Exact real-time requirement per regime?",
   "Required compute + weight bandwidth per VLA/VLM regime.",
   "requirement floor; not a chip", False),
 "measurement_priority_bar": ("measurement_priority_table.csv", "B", "structural", "low", "backup",
   "What missing measurement unblocks the most candidates?",
   "Ranks missing measurements by how many abstraction candidates they unblock.",
   "meta/methodology view", False),
 "workload_influence_loo_delta": ("IM.macro_micro_influence", "B", "structural", "low", "backup",
   "How stable are corpus metrics to leave-one-out?",
   "Leave-one-out stability of corpus metrics (winner-stable vs magnitude-unstable).",
   "robustness diagnostic", False),
 # ---- QA-only ----
 "evidence_type_by_workload": ("unified_facts", "A/B/C", "QA", "low", "QA",
   "Provenance mix of facts per workload?", "Fact counts by evidence tier per workload.",
   "QA/methodology only", False),
 "evidence_type_by_phase": ("unified_facts", "A/B/C", "QA", "low", "QA",
   "Provenance mix of facts per phase?", "Fact counts by evidence tier per analysis phase.",
   "QA/methodology only", False),
 "required_command_rate_envelope": ("IM.timing_requirement_envelope", "C", "proxy", "high", "QA",
   "Required dispatch/s vs deadline?", "Required dispatch rate (a PROXY).",
   "PROXY ~12× undercount; measured only for small_llama", False),
 "primitive_regret_bar": ("primitive_coverage_matrix.csv", "A", "structural", "low", "QA",
   "Per-primitive mean vs worst coverage?", "Redundant with primitive_set_frontier / decision_primitive_choice.",
   "structural coverage", False),
 "decision_primitive_choice": ("primitive_coverage_matrix.csv", "A", "structural", "low", "QA",
   "Single-primitive worst vs mean coverage?", "Subsumed by primitive_set_frontier for the talk.",
   "structural coverage", False),
 "avoidable_reload_by_region": ("data_movement_table.csv", "A", "captured-config", "medium", "QA",
   "Avoidable reload bytes per region?", "Subsumed by decision_weight_residency.",
   "captured-config bytes", False),
 "boundary_placement_heatmap": ("hw_sw_boundary_matrix.csv", "B", "structural", "medium", "QA",
   "Boundary placement per abstraction × level?",
   "Raw numeric-colored version; replaced by boundary_placement_simplified (categorical).",
   "use the simplified categorical version for the talk", False),
 # ---- drop / backup-only ----
 "realtime_requirement_surface": ("arithmetic_intensity.csv", "A/B", "deployment-composition", "medium", "drop",
   "3D required compute vs rate × workload?",
   "On a live slide; restyled clean. 2D realtime_requirement + lever_ablation carry the same message more legibly.",
   "backup-only; requirement floor under the workload model, not a chip", True),
 "decision_sharding_cost": ("sharding_table.csv", "A", "captured-config", "low", "drop",
   "Absolute extra sharding bytes?",
   "Log axis wastes space; superseded by sharding_scalability (normalized).",
   "backup-only; structural bytes", False),
}

INV_COLS = ["plot_id", "source_artifact", "evidence_tier", "scale", "overclaim_risk", "proposed_use",
            "restyled_in_final_pass", "dse_decision", "main_claim", "caveat"]
CLS_COLS = ["plot_id", "decision", "dse_question", "safe_takeaway", "required_caveat"]


def main():
    inv, cls = [], []
    for pid, (src, tier, scale, risk, use, q, claim, caveat, restyled) in PLOTS.items():
        inv.append({"plot_id": pid, "source_artifact": src, "evidence_tier": tier, "scale": scale,
                    "overclaim_risk": risk, "proposed_use": use,
                    "restyled_in_final_pass": "yes" if restyled else "no",
                    "dse_decision": q, "main_claim": claim, "caveat": caveat})
        cls.append({"plot_id": pid, "decision": use, "dse_question": q, "safe_takeaway": claim,
                    "required_caveat": caveat})
    for name, cols, rows in [("plot_inventory", INV_COLS, inv), ("final_plot_classification", CLS_COLS, cls)]:
        with open(HERE / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        md = [f"# {name.replace('_', ' ').title()}\n",
              f"{len(rows)} plots. Generated by `_gen_inventory.py` (reproducible).\n",
              "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            md.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
        (HERE / f"{name}.md").write_text("\n".join(md) + "\n")
    # ---- the four list docs (Phase 4) ----
    AVOID = "speedup / faster / cycles / area / energy / throughput / optimal / measured hardware performance"
    lists = {
        "final_main_plot_list": ("main", "Final MAIN plots (on the slides)"),
        "final_backup_plot_list": ("backup", "BACKUP plots (Q&A / appendix)"),
        "internal_QA_plot_list": ("QA", "INTERNAL QA-only plots (methodology / not for the talk)"),
        "drop_or_replace_plot_list": ("drop", "DROPPED / replaced plots (backup-only at most)"),
    }
    for fname, (use, title) in lists.items():
        sel = [(pid, v) for pid, v in PLOTS.items() if v[4] == use]
        md = [f"# {title}\n", f"{len(sel)} plots. Generated by `_gen_inventory.py`.\n"]
        for pid, (src, tier, scale, risk, _u, q, claim, caveat, restyled) in sel:
            md.append(f"## {pid}")
            md.append(f"- **DSE question:** {q}")
            md.append(f"- **Say (claim):** {claim}")
            md.append(f"- **Evidence tier / scale:** {tier} / {scale}  ·  **source:** `{src}`"
                      + ("  ·  restyled (clean) ✓" if restyled else ""))
            md.append(f"- **Required caveat:** {caveat}")
            if use == "main":
                md.append(f"- **Avoid saying:** {AVOID}")
            md.append("")
        (HERE / f"{fname}.md").write_text("\n".join(md) + "\n")
    n = {u: sum(1 for v in PLOTS.values() if v[4] == u) for u in ("main", "backup", "QA", "drop")}
    print(f"emitted inventory + classification + 4 lists for {len(PLOTS)} plots; class counts = {n}")


if __name__ == "__main__":
    main()
