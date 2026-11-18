---
name: roofline-and-utilization
description: "Read scoped compute ceilings and resource bookings: what roofline_bound_cycles, bound_by, gap_ratio, headroom_cycles, array_utilization_pct and dominant_modeled_resource mean and how each is computed, how to see utilization per test case, operator family, size class, connected model and unit (scripts/perf_summary.py), and which compiler lever moves each signal. Open when choosing performance work or after any check that returned cycles."
---

# Roofline and utilization

The objective is fewer cycles for the same results. Every passing case with cycles gets an entry in a
self-check's `performance_brief.largest_headroom` and in the grade's
`feedback_brief.performance_priority.development_headroom` (in `qa/latest_completed_grade.json`).

## What the fields mean

- `roofline_bound_cycles`, `bound_by`: the largest of two structural floors of `structural_roofline`:
  `compute` = `structural_compute_floor_cycles` = ceil(`useful_macs` / `structural_peak_macs_per_cycle`),
  and `read requests`/`write requests` = `logical_request_ideal_service.<side>.ideal_requested_service_cycles`
  = the workload's logical bytes / `client_peak_bytes_per_cycle`. A lower bound, not a prediction: it omits
  fill/drain, issue, host work and dependencies (`missing_terms`).
- `gap_ratio` = cycles / bound; `headroom_cycles` = cycles - bound (absolute, so large shapes lead).
- `array_utilization_pct` = 100 * `useful_macs` / (cycles * `structural_peak_macs_per_cycle`): compute-ceiling attainment, not measured PE occupancy. `lowest_utilization` ranks by it;
  `structural_floor_to_estimate_ratio` instead uses the rounded compute floor.
- `dominant_modeled_resource` {`name`, `modeled_activity`}: the unit with the largest
  `capacity_normalized_activity` (busy unit-cycles / (capacity * cycles)) in
  `performance.scoped_resource_feedback.resources`. Always from the Spike timing model (also on gSIM rows);
  a modeled reservation share, not a hardware counter.
- `evidence` says whether cycles are measured (gSIM) or modeled; see the `cycle-model-and-gsim` skill.

Field-by-field reference with the writer of each: [reference/fields.md](reference/fields.md).

## Procedure

1. Summarize your own evidence (read-only):
   `python .agents/skills/roofline-and-utilization/scripts/perf_summary.py [--top N] [--json]`:
   utilization quartiles; utilization and headroom per family, size class, bound and cohort; top cases
   with bound and busiest unit; unit activity; modeled/measured ratios; cases without gSIM; whole-model
   kernel shares; idle tuning; your request latencies.
2. Choose where to work. A lever that serves a whole group (family, size class, a model's dominant
   kernel) moves more cycles than one case. Absolute headroom finds big wins; lowest utilization finds
   low useful-compute attainment; it does not identify which overhead caused the gap.
3. Diagnose one case: `bound_by`, the busiest unit, the traffic ratio (issued `target.requested_*_bytes`
   in `feature_counts` over the logical bytes), command counts and `useful_fraction_of_padded_work`.
   Read its command buffer with the entry's `inspect` command (`read-command-buffer` skill).
4. Pick a lever from [reference/levers.md](reference/levers.md); `suggested_mechanisms` already names
   candidate catalogue mechanisms to investigate, not a proven bottleneck (`pass-library` skill).
5. Build it as a variant or a `submission/TUNING_KNOBS.json` knob and compare on the same cases
   (`ab-library-pass` skill). Keep it only when numerics still pass and cycles fall beyond the model's
   spread or on gSIM (`cycle-model-and-gsim` skill).
6. Re-run the summary; the next grade and idle timing refresh the ranking.

## Utilization at each level

- Test case: entries above.
- Family / size class: script groups (declared operation, else the ops of `public_sources/<case>.mlir`;
  size class = decade of `useful_macs`).
- Connected model: `python agent_model_check.py check --model MODEL` reports per-kernel `cycles`
  (kind `functional` is an instruction-derived count, not elapsed cycles; `rtl` is gSIM) and `census.device_mac_fraction`; the script
  weights each kernel by its source regions to show which kernel dominates the model.
- Unit: the script's unit table; the `overlaps` rows show coincident named bookings, not physical engine overlap.
  Controller bookings may cover dispatch only; missing engine lifetimes remain unknown.

Read target facts (array dimensions, peak rate, storage sizes) from `rtl_facts`, the capability
manifest and these fields; never assume them.
