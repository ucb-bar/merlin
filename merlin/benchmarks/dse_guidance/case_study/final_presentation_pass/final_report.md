# Final presentation pass — report (P26)

A curated, clean, decision-relevant plot set for the talk + paper, with safe wording and an explicit honesty
discipline. This report is the index; details are in the linked artifacts.

**Closing story:** *Merlin recovers a loop/region/state workload contract from model captures and translates
it into DSE search axes — primitive sets, residency/capacity/dtype, sharding communication, real-time
requirement envelopes, and HW/SW boundary placement — and flags which axes remain blocked by capture fidelity.*

## 1. Deliverables
- **Methodology:** [`METHODOLOGY.md`](METHODOLOGY.md) — end-to-end flow, what we capture, why a compiler, how
  the mining works, **where/whether an LLM is used and how it is verified**, and per-plot data lineage.
- **Clean figures:** `figures/*.png` (19 curated) + `boundary_placement_simplified.png`; index in
  [`figure_manifest.csv`](figure_manifest.csv).
- **Classification:** [`plot_inventory.md`](plot_inventory.md) (39 plots) ·
  [`final_plot_classification.md`](final_plot_classification.md).
- **Lists:** [`final_main_plot_list.md`](final_main_plot_list.md) ·
  [`final_backup_plot_list.md`](final_backup_plot_list.md) ·
  [`internal_QA_plot_list.md`](internal_QA_plot_list.md) ·
  [`drop_or_replace_plot_list.md`](drop_or_replace_plot_list.md).
- **Slides:** [`final_slide_skeleton.md`](final_slide_skeleton.md) (15-min, 10 slides).
- **Wording audits:** [`roofline_semantics_audit.md`](roofline_semantics_audit.md) ·
  [`residency_plot_semantics.md`](residency_plot_semantics.md) ·
  [`primitive_frontier_main.md`](primitive_frontier_main.md).
- **Checker:** [`check_final_plots.py`](check_final_plots.py).

## 2. Classification summary (39 plots)
- **Main (12):** table_capture_summary, capture_fidelity, capture_level_ablation, primitive_set_frontier,
  operator_cumulative_mac, decision_weight_residency, decision_capacity_dtype, realtime_requirement,
  lever_ablation, boundary_necessity_matrix, arithmetic_intensity_roofline (relabeled), visible_linear_fraction.
- **Backup (18):** sharding_scalability, sharding_comm_tradeoff, decision_sharding_per_top_op,
  work_coverage_by_workload, deployment_magnitude, shape_class_mac_share, primitive_frontier_by_threshold,
  macro_vs_micro_primitive_coverage, resident_capacity_by_dtype, required_compute_envelope,
  required_memory_movement_envelope, critical_path_parallelism, table_deployment_magnitudes,
  table_arithmetic_intensity, table_low_bit_tiers, table_realtime_requirement, measurement_priority_bar,
  workload_influence_loo_delta.
- **QA-only (7):** evidence_type_by_workload, evidence_type_by_phase, required_command_rate_envelope (proxy),
  primitive_regret_bar, decision_primitive_choice, avoidable_reload_by_region, boundary_placement_heatmap.
- **Drop / backup-only (2):** realtime_requirement_surface (3D hard to read), decision_sharding_cost
  (superseded by the normalized sharding_scalability).

## 3. Regenerated (clean restyle) + what changed
A new module `presentation_final.py` redraws the curated set in one clean identity — background `#FDF7EF`,
ink `#2E2D2C`, palette `#333351/#0F3759/#8B93A6/#815E5E/#7D886C/#AB9A89`, per-series hatches + soft drop
shadows on bars, Noto Serif Display titles + Inter body, gold/blue emphasis, booktabs tables, callout boxes,
smart y-limits. **Removed all on-figure tier badges / scale chips / mid-axes text**; a short italic caveat
subtitle appears only on high-risk plots. The earlier (P24/P25) `manual_validation/figures/` set is unchanged.
Notable fixes during the visual pass: `operator_cumulative_mac` relabeled curves were colliding → now a
hot-op bundle + 3 labeled diffuse curves; `decision_capacity_dtype` got human-readable MB/GB ticks.

## 4. Wording fixes (high-risk)
- **arithmetic_intensity_roofline** → relabeled to **weight-stream AI** under a **hypothetical machine-balance
  band**; subtitle "modeling view, not measured performance, not full-memory AI". See the roofline audit.
- **realtime_requirement / lever_ablation** → "requirement floor / requirement reduction, not a chip /
  not a speedup".
- **decision_weight_residency** → "bytes moved (not bandwidth); K = IR scf.for".
- **boundary_placement** → replaced the raw numeric heatmap with a **categorical** search-space matrix.
- **boundary_necessity** → categorical N/U/P/B title; "blocked = capture/evidence blocked".

## 5. Remaining caveats (stated, not hidden)
- Captured-config magnitudes (residency, work_coverage, envelopes) are reduced-capture scale; deployment
  scale comes only from config-composition (deployment_magnitude / tables). bitvla deployment config is not
  sourceable → omitted from deployment magnitudes (no guess).
- The roofline is **weight-stream**, not full-memory AI; the machine balance is hypothetical.
- K is IR-recovered for the loop corpus; H (action horizon) is source/config.
- One synthetic toy (small_llama) is excluded from the analyzed/curated figures.

## 6. Claims that are SAFE to present
- "We recover K, the repeated region, loop-carried state and KV directly from the IR `scf.for`."
- "One tiling primitive is not robust; a 2-primitive set covers the corpus (structural pad-waste coverage)."
- "Residency removes a K× factor from weight traffic / the weight-bandwidth requirement to hit a target rate."
- "Action chunking (÷H) and residency (÷K) reduce the 30 Hz requirement floor."
- "Capture fidelity gates which DSE axes are available; low-bit packing and scales remain erased."
- All magnitudes labeled captured-config vs deployment-composition.

## 7. Claims that are NOT safe (never made)
- Any speedup / latency-met / cycles / area / energy / throughput / optimality / "best design".
- "Compute-bound on hardware X" / attainable throughput / predicted performance.
- Treating a requirement floor or the roofline band as a measurement.

## 8. Tests / checks run
- `check_final_plots.py` → **139/139** (manifest complete; no forbidden wording; roofline/real-time/K
  qualifiers present; QA-only ∉ main; every slide caveated).
- `verify_implementation.py` → **631/631** (additive; dangerous-terms scan now excludes the curated
  `final_presentation_pass/` subtree, exactly like `manual_validation/`).
- `pytest test_dse_guidance.py` → **181 passed, 1 skipped**.
- No unrelated files changed; the existing study figures + pipeline are untouched.

## 9. Unresolved issues
- None blocking. The 3D `realtime_requirement_surface` is intentionally backup-only. Several backup plots were
  not restyled (they stay in `manual_validation/figures/`); restyle on demand if any is promoted to a slide.
