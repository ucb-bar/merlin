---
name: performance-feedback-levels
description: "Map of every performance signal by level and freshness: the per-request performance_brief, the per-grade performance_priority and development_headroom, idle-time timing and idle tuning (qa/idle_tuning.json, TUNING_KNOBS.json, variants/), whole-model checks, certification tiers and L3 gSIM evidence, and the final campaign gates; what to check after each kind of compiler change. Open to find the freshest cycles for a case or to plan how to verify a performance change."
---

# Performance feedback levels

Correctness: `feedback-files` skill. Cycle fields: `roofline-and-utilization` and `cycle-model-and-gsim`.

| Level | Where | Fresh for | Arrives |
|---|---|---|---|
| Request | `performance_brief` in the response your check prints (`.qa_channel/resp_<id>.json`, gSIM: `simresp_<id>.json`) | the `submission_sha256` it graded (`graded_newer_bytes`: it took newer bytes) | per check; latencies: `perf_summary.py` |
| Idle timing | `resp_op_idle_*.json` (origin `operator_idle_performance`) | your newest submission digest, passing performance-cohort cases | while CPUs are free, without asking |
| Idle tuning | `qa/idle_tuning.json`, `performance_brief.idle_tuning`, per entry `idle_tuning_experiments` | candidate bytes and stated baseline/model; edits retain comparisons with freshness labels | beside baseline timing while CPUs are free |
| Grade | `qa/latest_completed_grade.json` `feedback_brief.performance_priority` (`correctness_first`, `comparison_groups`, `development_headroom`); `qa/latest_grade_summary.txt` (`perf=STATUS cycles=N` per case); `qa/verdict.json` while current | `graded_source_sha256`; rows from older grades say `graded=` | each background grade |
| Whole model | `python agent_model_check.py check --model MODEL --out REPORT.json`: per-kernel `cycles` (`functional` or `rtl`), `census.device_mac_fraction`, gates | the checked source | per request |
| Certification | `qa/tier_state.json` (`L2`, `L3` and `<certs>` per case, bound to `execution_digest`); L3 = elaborated RTL on gSIM, the measured cycles | the certified executable | automatic promotion after L2 passes, or an admitted `--sim gsim` |
| Campaign | `joint_campaign` in the grade, `qa/joint_physical_correctness.json` (`obligations`, `pending`), `qa/progress.json` | the whole run | the final post-freeze grade and validation are not visible |

Tiers: L0 independent numeric golden, L1 command buffer self-consistency, L2 functional model on Spike
(where timing is modeled), L3 cycle-accurate RTL (gSIM). Performance is never a pass/fail tier; the
objective is cycles of the same correct results. Details: [reference/signal-map.md](reference/signal-map.md).

## Getting candidates timed

- Timing missing? Use `explore-schedules` for traffic, storage and synchronization; check Spike
  numerics and retain alternatives.
- Your own request: `python agent_selfcheck.py --suite scalability --capsules A,B --wait-budget 5`
  (Spike timing of large shapes), or `--submission variants/NAME` to time an alternative.
- Without asking: keep alternatives in `variants/NAME/` or declare choices in
  `submission/TUNING_KNOBS.json` (schema `merlin.tuning_knobs.v1`; see the `offer` in
  `qa/idle_tuning.json`). Idle tuning times them on the cases with the most headroom.
  Its `measured_cycles_*` fields are Spike model estimates despite the name.
- Parallel comparison: `agent_selfcheck.py --sweep variants/a,variants/b --capsules A,B --wait-budget 5`;
  collect with `--attach SWEEP_ID`. Knobs, bundles and retained baseline labels:
  `ab-library-pass` [sweeps](../ab-library-pass/reference/sweeps.md).
- `performance_brief.tuning_knob_candidates` lists policy switches already in `submission/`
  (location, default, values seen); `fence_bound` lists cases fenced after every output tile.

## After each kind of change

1. Schedule, tiling or lowering: check affected cases together, including large scalability cases;
   compare `performance_brief` (cycles, `array_utilization_pct`, `bound_by`) with the previous response.
2. A library pass or mechanism on or off: a variant or a knob, the same cases, the comparison rule
   of `cycle-model-and-gsim` (`ab-library-pass`, `pass-library` skills).
3. Command or issue structure: inspect the command buffer per command (`read-command-buffer` skill);
   then cycles as in 1.
4. Host code, fusion or connected regions: `agent_model_check.py check` for each affected model; compare
   kernel cycles and `device_mac_fraction`, then the connected cases' cycles.
5. Before keeping a change: check `correctness_first`, `development_headroom` and admitted gSIM
   confirmation; `perf_summary.py` reports utilization and timing freshness.
