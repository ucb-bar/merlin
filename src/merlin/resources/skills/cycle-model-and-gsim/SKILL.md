---
name: cycle-model-and-gsim
description: "The two sources of cycles: the Spike cycle-approximate timing model (estimated_whole_soc_cycles, evidence modeled: Spike timing model) and gSIM RTL simulation (barrier_cycles, evidence measured: gSIM). When each is produced, how to request it, what it costs, how to read per-unit modeled activity, how far to trust the model (model_calibration, measured/modeled ratios, calibrated_interval) and the rule explore with the model, decide with gSIM on the same cases. Open before comparing two variants by cycles."
---

# Cycle model and gSIM

## Available evidence

| | Spike timing model | gSIM |
|---|---|---|
| Field | `performance.estimated_whole_soc_cycles` | `barrier_cycles` with `barrier_tier: L3` |
| Label | `modeled: Spike timing model (provisional\|qualified\|self-check)` | `measured: gSIM cycles of this request`, `measured: latest gSIM certificate` |
| Usable when | PROVISIONAL/QUALIFIED, `complete_event_accounting: true` | passing row of the matched executable |
| Resources | `scoped_resource_feedback` models reservations/overlap | model fields alongside gSIM remain modeled |

PROVISIONAL is not an accuracy certificate; complete accounting does not establish accuracy.
UNKNOWN exposes `reason`, `unavailability_stage`, `unavailability_detail` and `unresolved_events`.
Missing costs never become zero. Compare model revision, hardware domain and completed-call scope.

## Keep optimizing while coverage improves

Do not wait for calibration. Select legal alternatives by inspecting emitted operand traffic/reloads,
packing frequency, tile coverage, storage lifetimes/spills, completion boundaries and CPU work.
Use actual loop multiplicities. Keep distinct resource tradeoffs: fewer bytes need not mean fewer
cycles. Instructions are secondary diagnostics. Retain a correct baseline and promising variants
or knobs, check affected Spike outputs and independent shapes/tails, and record structural hypotheses
with source/emission identities. Static tile coverage is not measured PE occupancy.

Work across representative contraction, convolution and connected regions; preserve all visible
outputs and quantization boundaries. Public optimized examples may suggest mechanisms; private
references/scores remain excluded. See `explore-schedules` and `ab-library-pass`.

## Requesting feedback

- Spike model: `python agent_selfcheck.py --suite scalability --capsules A,B --wait-budget 5`,
  core self-checks, background grades and idle requests on passing primary performance cases.
- gSIM: admitted automatic L3 promotion, `python agent_selfcheck.py --sim gsim --capsules NAME`
  or `python simjob.py submit --sim gsim --capsules NAME`. TASK.md controls eligibility and budgets.
  Prefer small/medium mechanism witnesses. Large shapes remain Spike-only in development unless
  explicitly authorized; do not run large gSIM simply because modeled timing is missing.
- Cost: `perf_summary.py`; see `roofline-and-utilization` for signals and request latency.

## Trust and decisions

`model_calibration` contains legacy same-executable descriptive ratios: `median_measured_over_modeled`,
`interquartile`, `spread`; too few pairs means `uncalibrated`. Per-entry `calibrated_estimate`,
`calibrated_interval`, `model_confidence` are descriptive overlays, not accepted fitted runtimes or
statistical confidence guarantees. A narrow spread does not prove schedule ranking or large-case
accuracy. Inspect independent matched validation and the accuracy card, including coverage/UNKNOWNs.
Details: [reference/trust.md](reference/trust.md).

Use complete same-provider, same-model, same-ROI cycles for provisional A/B comparisons; preserve
uncertainty. Seek admitted public witnesses to distinguish uncertain mechanisms. Only matched
measurements establish a measured speedup in their hardware domain. Where the complete model has
not run, decide on the case's calibrated `fast_estimate` (labelled, with its band) and revert if a
complete estimate or measurement later disagrees. Record the evidence
and outstanding questions; background model improvements apply to future requests.
