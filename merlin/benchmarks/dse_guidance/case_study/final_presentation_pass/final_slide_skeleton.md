# Final slide skeleton — 15 min (~10 slides)

Thesis: **Merlin recovers a loop/region/state workload contract from model captures and turns it into DSE
search axes — it does NOT do DSE and claims no measured performance.** Figures are in `figures/`; each slide
names a caveat sentence to say aloud.

---

## Slide 1 — Thesis (no figure / pipeline cartoon)
- **Claim:** flat single-pass model captures are not enough; Merlin extracts a *workload contract* that
  defines the DSE search axes.
- Bullet: inputs = real model captures (MLIR); outputs = facts, requirement envelopes, boundary candidates.
- Bullet: we emit *requirements & search spaces*, not designs.
- **Caveat:** no speedup / cycles / area / energy / measured performance is claimed anywhere.
- Speaker note: set expectations — this is a workload-contract extractor feeding DSE, not a DSE optimizer.
- Backup: `final_report.md` story paragraph.

## Slide 2 — What is recovered  ·  `figures/table_capture_summary.png`
- **Claim:** Merlin recovers K, the repeated region, loop-carried operands and KV cache — a loop/region/state
  contract, not just op shapes.
- Bullet: every value is read from the `scf.for` loop in the IR (Tier A).
- **Caveat:** structural facts only; no performance implied.
- Speaker note: this table is the contribution in one glance.
- Backup: `figures/capture_level_ablation.png`.

## Slide 3 — What capture fidelity enables / blocks  ·  `figures/capture_fidelity.png`
- **Claim:** which DSE axes are available depends on what the capture preserves vs erases.
- Bullet: recovered = shapes/roles/K/KV/loop-state; erased = packed-lowbit + scales; not-claimed = latency.
- **Caveat:** "blocked/erased" is a capture-fidelity limit, not a workload property.
- Speaker note: the erased rows are exactly the next capture-level to build.
- Backup: `figures/capture_level_ablation.png`, `boundary_placement_simplified.png`.

## Slide 4 — Primitive-set frontier  ·  `figures/primitive_set_frontier.png`
- **Claim:** one primitive is not robust across workloads; a small set is required.
- Bullet: best single primitive leaves the worst workload poorly covered; 2 primitives → ~full coverage.
- **Caveat:** structural pad-waste MAC coverage — does not rank hardware performance.
- Speaker note: motivates a primitive *set* as a DSE axis.
- Backup: `final_backup_plot_list.md` (frontier-by-threshold, macro-vs-micro).

## Slide 5 — Operator concentration  ·  `figures/operator_cumulative_mac.png`
- **Claim:** workloads split into hot-op-dominated vs diffuse regimes.
- Bullet: steep curves → a few giant ops; diffuse → broad operator-family coverage needed.
- **Caveat:** cumulative MAC share from IR shapes; no deployment-scale claim.
- Speaker note: tells you where acceleration effort pays off per workload.
- Backup: `figures/work_coverage_by_workload.png`.

## Slide 6 — Residency as a loop/rate abstraction  ·  `figures/decision_weight_residency.png`
- **Claim:** non-resident weight traffic grows with the loop count K; residency is loop/rate-aware.
- Bullet: reload-every-step scales ×K; resident loads once; dot = the model's actual K from `scf.for`.
- **Caveat:** weight *bytes moved* (not bandwidth); captured-config scale.
- Speaker note: residency is a recovered structural decision, not a guess.
- Backup: `figures/arithmetic_intensity_roofline.png`.

## Slide 7 — Capacity × dtype  ·  `figures/decision_capacity_dtype.png`
- **Claim:** dtype determines whether repeated-head weights fit on-chip (residency feasibility).
- Bullet: int8/int4 push more workloads under a given on-chip budget than bf16.
- **Caveat:** captured-config weight sizes; a feasibility envelope, not a chip.
- Speaker note: ties the residency axis to a concrete capacity budget.
- Backup: `final_backup_plot_list.md` (resident_capacity_by_dtype).

## Slide 8 — Real-time requirement envelope  ·  `figures/realtime_requirement.png` (+ `lever_ablation.png`)
- **Claim:** Merlin emits a HW-independent requirement floor for a target control rate; residency + chunking
  lower it.
- Bullet: 30 Hz needs X GB/s reload vs much less resident; chunking (÷H) + residency (÷K) cut it further.
- **Caveat:** requirement floor under the workload model — not a chip measurement, not a speedup.
- Speaker note: this is the bridge from workload contract to system sizing.
- Backup: `figures/lever_ablation.png`, `figures/arithmetic_intensity_roofline.png`.

## Slide 9 — HW/SW boundary & necessity  ·  `figures/boundary_necessity_matrix.png`
- **Claim:** DSE should search *which* abstractions are necessary/useful/possible/blocked — and *where* they
  can live in the stack.
- Bullet: a few abstractions are necessary across all workloads; others are workload-specific or blocked.
- **Caveat:** "blocked" = capture/evidence blocked; "possible" = not discriminating, not a recommendation.
- Speaker note: pair with `boundary_placement_simplified.png` for the where-in-the-stack view.
- Backup: `boundary_placement_simplified.png`, `boundary_necessity_full_backup.png`.

## Slide 10 — What remains blocked / next capture level
- **Claim:** the honest residuals are the roadmap: packed-lowbit layout + scales (erased), full attention
  internals, and any K taken from source/config rather than IR.
- Bullet: each blocked axis names the exact capture upgrade that unblocks it.
- **Caveat:** these are stated limits, not silently dropped; nothing here is a performance claim.
- Speaker note: close on "Merlin turns captures into DSE axes and flags what fidelity still blocks."
- Backup: `final_report.md` (claims safe vs not-safe), `internal_QA_plot_list.md`.
