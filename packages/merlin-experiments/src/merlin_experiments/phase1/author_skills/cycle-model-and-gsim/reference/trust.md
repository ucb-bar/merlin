# Trusting modeled cycles: the fields

## Model calibration (`model_calibration`, schema `merlin.timing_model_calibration.v1`)

Built from your own `.qa_channel/resp_*.json` and `simresp_*.json` (newest files first): a pair is the
newest modeled estimate and the newest gSIM L3 cycles of the same test case and the same
`execution_digest`. Modeled values count only with `complete_event_accounting: true`; measured values
only from passing gSIM L3 rows. Scalability-suite rows carry no `execution_digest`, so they never pair.

| Field | Meaning |
|---|---|
| `pairs` | number of same-executable pairs |
| `basis` | the pairing rule |
| `levels.family.<family>` | ratios of cases whose public declaration (`isa/<case>/capsule.yaml`) names an operation of that family |
| `levels.bound_class.<compute\|memory requests>` | ratios by which roofline term limits the case |
| `levels.all.all` | all pairs |
| `median_measured_over_modeled` | median of measured / modeled; above 1 the model is optimistic |
| `interquartile` | [q1, q3] of the ratio |
| `spread` | q3 - q1; descriptive dispersion, not a confidence guarantee |
| `status: uncalibrated` | fewer pairs than the level needs; no ratio offered |
| `use` | the comparison rule, as published |

Per modeled entry (`largest_headroom[]`):

| Field | Meaning |
|---|---|
| `calibrated_estimate` | legacy overlay: modeled cycles * the most specific descriptive median (family, else bound class, else all) |
| `calibrated_interval` | legacy overlay: modeled cycles * [q1, q3]; not a statistical confidence interval |
| `model_confidence` | `level`, `group`, `pairs`, `median_measured_over_modeled`, `spread`; or `status: uncalibrated` |
| `calibration_label` | an estimate scaled by this run's pairs; never a measurement |

## What the ratios do and do not tell you

- A narrow `spread` describes the recorded ratios. It does not establish correct ranking of
  alternative schedules, large-case accuracy, an accepted fitted runtime or hardware-domain
  equivalence. Those require independent matched validation with explicit coverage.
- Pairs come mostly from small correctness cases (they are the ones gSIM admits); large scalability
  shapes may sit outside that range. Treat their calibrated interval as a lower confidence estimate.
- A case whose ratio is far from the group (the summary script lists newest pairs off by more than
  2x) has a cost the model does not price for that schedule: check `unresolved_events` of related
  rows, its traffic ratio and command counts, and prefer gSIM evidence for decisions on it.
- `timing_epoch` and `model_sha256` identify the model. Compare two estimates only under the same
  model, scope (`scope_sha256`) and epoch; idle tuning results already require this.

## Other evidence fields

| Field | Where | Meaning |
|---|---|---|
| `barrier_cycles` | gSIM rows | cycles of the measured window on elaborated RTL |
| `execution_digest` | every per-case row, `qa/tier_state.json` | identity of the executable that produced the row; binds certificates and calibration pairs |
| `qa/tier_state.json` `<case>.L3` | grade state | the latest L3 certificate (`status`, `execution_digest`); `<certs>.L3` holds earlier slots, `joined` when an unchanged device boundary reused an earlier certificate |
| `structural_qualification`, `tier_boundary` | gSIM rows | which boundary was measured and under which toolchain pins |
| `observation_seconds` | `performance` | host wall time of the timing observation, not target cycles |
| `observation_diagnostics` | `performance` | where an observation failed (`failure_stage`, `deadline_exhausted`) |
