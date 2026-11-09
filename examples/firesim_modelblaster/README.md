# Historical ModelBlaster FireSim operator adapter

This independently installable package retains the historical staging and queue
implementation for external operators and research reproduction. Merlin core
no longer discovers or invokes `merlin.firesim_runners` entry points.
`run_on_firesim` now requires explicit prepare/collect; `select_runner` refuses
legacy launch requests. See [the handoff contract](../../docs/guides/firesim.md).

An operator maintaining the historical integration can install this package in
its own environment and use its adapter directly. Source, workload identity,
queue policy, environment settings and terminal marker are retained here; their
presence does not authorize or qualify execution from Merlin. Image builders
accept an explicit `completion_metric_prefix` when required by the operator.
The operator must return the handoff's request hash and actual output artifacts
in `merlin.external_result.v1` format for Merlin collection.

The package requires `MERLIN_MODELBLASTER` to select the external ModelBlaster
checkout containing `src/modelblaster/validation/firesim_runner.py` or
`validation/firesim_runner.py`. Its implementation remains unchanged. The old
entry-point registration is retained for historical standalone consumers, not
as a supported core launch interface. Preflight alone establishes no FPGA,
bitstream, numerical or timing correctness.
