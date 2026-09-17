# AGENT.md — merlin/targets/gemmini/tools

## Purpose
Tools whose every fact belongs to this target and that shared code therefore must not name.

- `baremetalc_corroborate.py` builds this target's own reference programs (the upstream bareMetalC
  `mvin_mvout` test and the library `tiled_matmul_auto`) with the backend's bareMetalC toolchain, runs
  them on spike or verilator, and compares the output to merlin's `Tensor` goldens. The contract
  declares it as `plugin.reference_programs`. Callers load it with
  `merlin.targetgen.plugins.load_declared(target, "reference_programs")`. Run it directly as a script
  to write the corroboration report.

## Invariants
- These are reference ORACLES only. Nothing here is copied into or called from a graded submission.
- Imports are absolute (`merlin.*`). The package is loaded by file path under a synthetic namespace,
  never by putting this directory on `sys.path`.
