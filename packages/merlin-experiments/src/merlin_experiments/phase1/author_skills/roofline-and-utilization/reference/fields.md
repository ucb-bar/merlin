# Performance fields: definition, where you see them, who writes them

Every number below is computed by the harness from your own executable's checks; nothing is a target
constant written into this skill. "Writer" names the harness module, for traceability.

## Per test case (self-check `performance_brief.largest_headroom[]`, grade `development_headroom.largest_headroom[]`)

| Field | Definition | Writer |
|---|---|---|
| `cycles` | gSIM `barrier_cycles` when the evidence is measured, else the Spike timing model's `estimated_whole_soc_cycles` | performance_brief / performance_headroom |
| `evidence` | `measured: gSIM cycles of this request` (this response's L3 row), `measured: latest gSIM certificate` (the cycles of the L3 certificate in `qa/tier_state.json`), `modeled: Spike timing model (provisional\|qualified)` (the grade row's record), `modeled: Spike timing model (self-check)` (newest submission self-check that timed the case) | same |
| `roofline_bound_cycles` | max over the positive terms `structural_compute_floor_cycles`, `logical_request_ideal_service.read.ideal_requested_service_cycles`, `...write...` | performance_headroom `_bound` |
| `bound_by` | name of that term: `compute`, `read requests`, `write requests` | same |
| `gap_ratio` | `cycles / roofline_bound_cycles` | same |
| `headroom_cycles` | `cycles - roofline_bound_cycles`; entries rank by it, largest first | same |
| `array_utilization_pct` | `100 * useful_macs / (cycles * structural_peak_macs_per_cycle)`; compute-ceiling attainment, not physical array occupancy | same |
| `dominant_modeled_resource` | resource with the largest `capacity_normalized_activity`; `modeled_activity` is that value | performance_headroom `_dominant_resource` |
| `reading` | one sentence joining utilization and the busiest unit | performance_brief |
| `suggested_mechanisms` | up to four catalogue mechanisms for the busiest unit's facets, or for `bound_by` when no unit is reported; units named in a mechanism's own entry first | mechanism_hints |
| `measured_on_earlier_source` | the certificate's cycles belong to an earlier executable than the newest L2 | performance_headroom |
| `calibrated_estimate`, `calibrated_interval`, `model_confidence` | modeled cycles times this run's measured/modeled ratio (see `cycle-model-and-gsim`) | model_calibration |
| `inspect` | the `isa_tools.py debug` command for the copied command buffer | performance_brief |

Ranked lists: `largest_headroom` (by `headroom_cycles`), `lowest_utilization` (by `array_utilization_pct`),
`without_evidence` (passing cases with no complete timing or no structural bound), `counts`,
`median_array_utilization_pct` (grade block only, over ranked entries).

## The structural roofline (`performance.structural_roofline`, schema `merlin.perf.partial_roofline.v1`)

| Field | Definition |
|---|---|
| `useful_macs` | sum of M*K*N over the case's declared contractions |
| `structural_peak_macs_per_cycle` | sum over the extracted arrays of rows * columns * multipliers per element, all issuing at once: an upper bound on the rate |
| `structural_compute_floor_cycles` | ceil(`useful_macs` / peak) |
| `logical_request_ideal_service.<read\|write>` | `requested_bytes` (the workload's logical operand/result bytes), `client_peak_bytes_per_cycle` (from the memory path facts), `ideal_requested_service_cycles` = bytes / rate |
| `padded_work`, `storage_requests` | optional premises; usually empty |
| `missing_terms` | what the floor does not charge (traffic amplification, contention, issue, host, fill/drain, dependencies) |

The request floors use the workload's logical bytes, not what your program issued. Compare them with the
program's own requests in `feature_counts` (`target.requested_read_bytes`, `target.requested_write_bytes`):
a ratio above 1 is re-fetched or amplified traffic.

## Timing record (`performance`, schema `merlin.phase1.joint_performance.v1`)

| Field | Meaning |
|---|---|
| `status` | `UNKNOWN` (no complete estimate; see `reason`, `unavailability_stage`, `unresolved_events`), `PROVISIONAL` (untuned model, complete accounting), `QUALIFIED` (tuned and qualified model), `REFUTED` (estimate below the compute floor) |
| `complete_event_accounting` | every event in the kernel-call window was priced; only then are cycles used |
| `estimated_whole_soc_cycles` | modeled cycles of the complete kernel-call window (`scope`, `scope_excludes`) |
| `structural_floor_to_estimate_ratio` | compute floor / modeled cycles |
| `feature_counts` | observed counts in the window: `target.commands`, `target.class.<class>`, `target.requested_*_bytes`, `target.useful_shape_macs`, `target.nominal_padded_macs`, `cpu.*` |
| `scoped_resource_feedback.resources[]` | per named resource: modeled capacity reservations / (capacity * scoped cycles), booking active fraction and generic queue entries; not physical DMA/array busy time or dependency stalls; at most 16 rows kept |
| `scoped_resource_feedback.overlaps[]` | `left`/`right` resource ids, fraction with both named bookings active; zero does not prove absent engine overlap |
| `scoped_resource_feedback.compute` | `useful_fraction_of_padded_work` = useful / nominal padded MACs |
| `modeled_resource_activity` | whole-trace totals; not a decomposition of the kernel call |
| `search_priority` | grade ordering inside a matched model group (`comparison_groups`): `diagnostic_headroom` = cycles - compute floor |

## Grade-level blocks (`feedback_brief.performance_priority`)

`correctness_first` (fix these before optimizing), `comparison_groups[].qualified` / `.provisional_diagnostics`
(ranked by `diagnostic_headroom` within one model and scope), `unranked`, the cohort lanes
(`primary` is the selected scalability cohort), and `development_headroom` (all passing cases, any cohort).

Physical engine occupancy, host-cycle attribution and storage high-water remain unknown without their own scoped producer. Traffic rates need an explicit byte boundary; bandwidth utilization also needs a matching sourced capacity. Never sum overlapping resource times.
