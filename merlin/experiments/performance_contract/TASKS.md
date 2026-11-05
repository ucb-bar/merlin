# Performance-study validation requirements

The scripts in this directory measure workload costs and accelerator occupancy. This checklist describes how to qualify their output. It is separate from the Gemmini paper's saved-result registry and does not report completed paper runs.

## Required input identities

- Select the target description, hardware elaboration, simulator or hardware runtime, compiler source and toolchain explicitly.
- Retain the workload program, numerical policy, concrete inputs, expected output, compiled object and executed binary identities.
- Record the timed interval, warmup policy, concurrency and number of repeats. Serial query latency and throughput under concurrent jobs are different metrics.

## Measurements

- Validate complete numerical output before assigning performance credit.
- Bind cycle counts to the submitted executable and execution fidelity. Do not treat functional estimates as measured hardware cycles.
- For occupancy, identify the measured engine signals and check that exclusive compute, transfer, overlap and idle buckets sum to the observed interval.
- For transfer amplification, distinguish bytes actually transferred from useful operand bytes and state how padding and repeated loads are counted.
- Measure cost at both test and layer scale before extrapolating between them.
- Preserve unavailable observations and rejected inputs. They are not zero costs or passing tests.

## Tools

- `layer_workload.py` constructs layer-scale workload inputs.
- `cycle_sweep.py` and `counter_occupancy.py` collect cycle and activity records.
- `movement_volume.py` and `dependence.py` examine movement and dependencies.
- `gemmini_occupancy.py` and `gemmini_dma_occupancy.py` provide Gemmini adapters.
- `headline.py` assembles comparisons from selected records.

Shared performance analyses belong to `src/merlin/perf/`. Hardware-specific observations must expose the traits needed by those analyses rather than select behavior from a target name. A second target requires its own numerical and measurement checks, not reuse of another target's recorded constants.

## Reporting

Report source and input hashes, numerical status, execution tier, timed scope, sample counts and aggregation rules with each result. A script's presence or an infrastructure test pass does not establish a new accelerator measurement.
