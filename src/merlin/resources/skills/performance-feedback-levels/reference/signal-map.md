# Performance signals in detail

## Request level: `performance_brief` (schema `merlin.selfcheck_performance_brief.v1`)

In every self-check and simulator response that has a passing case. Covers only that request's cases.
`counts` (`passing`, `with_evidence`, `without_evidence`), `largest_headroom` (up to 12 entries),
`lowest_utilization` (up to 6), `model_calibration`, `without_evidence`, `inspect_note`, and on
`submission/` responses the `idle_tuning` block. A gSIM response's entries say
`measured: gSIM cycles of this request`; other entries are modeled.

## Idle timing (origin `operator_idle_performance`, request ids `op_idle_`)

When no request of yours is waiting and shared CPUs are free, the harness submits the same Spike
scalability request you could make, for passing members of the grade's performance cohort whose
estimate does not describe your newest `submission/` digest (never-timed cases and the largest stale
estimates alternate). Results are ordinary responses (`resp_op_idle_*.json`); the grade's
`development_headroom` uses the newest of them as `modeled: Spike timing model (self-check)`. They
certify nothing and start no promotion.

## Idle tuning (origin `operator_idle_tuning`, ledger `qa/idle_tuning.json`, schema `merlin.idle_tuning_experiments.v1`)

Once idle timing is current, the same idle capacity times candidates on the passing cohort cases with
the most modeled headroom:

- every `variants/<name>/` whose bytes differ from `submission/` (the catalogue mechanisms it newly
  imports are named);
- every single-knob change of `submission/TUNING_KNOBS.json`
  (`{"schema": "merlin.tuning_knobs.v1", "settings_file": ..., "knobs": {name: {"values": [...],
  "mechanism": <catalogue id>}}}`), set only in a private copy.

Candidates exercising a mechanism suggested for those cases go first; none is ranked by expected gain.
Each experiment row: `test_case`, `mechanism`, `api`, `candidate`, `measured_cycles_before` and
`measured_cycles_after` (Spike model estimates of `submission/` and of the candidate, same model and
timing epoch), `delta_pct`, `numerics` (whether both still pass the case's numerical check),
`how_to_apply`, `source_sha256`; `not_compared` when the estimates are not comparable. Results bound to
an older digest are dropped. Nothing is applied to `submission/`: adopt a winner by making the change there.

## Grade level (`qa/latest_completed_grade.json`)

- `feedback_brief.performance_priority.correctness_first`: cases to repair before optimizing.
- `comparison_groups[]`: one per timing model and scope; `qualified` and `provisional_diagnostics`
  ranked by `diagnostic_headroom` (cycles minus compute floor, times the invocation weight).
- `primary_cohort`, `correctness_regression`, `calibration_diagnostics`: the cohort lanes;
  the primary lane is the selected scalability cohort.
- `development_headroom` (schema `merlin.development_performance_headroom.v1`): every case passing L2
  in any cohort, best evidence first (latest gSIM certificate, else the grade row's model estimate,
  else the newest self-check estimate), `counts`, `median_array_utilization_pct`,
  `model_calibration`, `largest_headroom` (15), `lowest_utilization` (8), `without_evidence` (20).
- `qa/latest_grade_summary.txt`: the same grade one line per case, a few kilobytes.
- `snapshot_current` and `graded_source_sha256` tell whether the grade describes your current bytes.

## Whole-model level (`agent_model_check.py`)

`check --model MODEL` builds and replays every kernel of a model: `gates` (`build`, `kernel_exact`,
`model_numerics`), `programs[].kernels[]` with `form`, `source_region_ids` (where the kernel is used),
`cycles` (`kind: functional` is Spike's count, `kind: rtl` gSIM's; `min`, `max`, `calls`), and
`census.device_mac_fraction` (share of the model's MACs on the device). `replay-kernel` replays one
kernel group. Weight a kernel's cycles by how many source regions use it to see where a model's
cycles go.

## Certification tiers (per case, `tiers`, `qa/tier_state.json`)

L0 independent numeric golden; L1 the command buffer's reference equals its simulation; L2 the
functional model (Spike) agrees, which is where timing is modeled; L3 elaborated RTL, cycle-accurate,
on gSIM: an L3 certificate is bound to the `execution_digest` that earned it, and its `barrier_cycles`
are the measured cycles. A mandatory tier that did not run keeps the case incomplete, never passed.

## Campaign level

`joint_campaign` (`scope`, `physical_certificate`, `final_only_pending`, `physical_correctness_required`)
and `qa/joint_physical_correctness.json` (`obligations` per case with `status` and certificate reuse,
`passed`, `pending`, `complete`) track the required gSIM correctness. After the freeze a final public and
hidden grade and an independent validation on the selected hardware configuration run outside the
workspace; correctness and performance are reported separately.
