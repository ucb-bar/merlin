# ReplayTrace and reuse defaults: local verification

TVM source: `64bed0cb9c876b89707e69584909abe5e13b0384`; Merlin runner: `9794880b465a8fa456bd0949914187ea192a45e3`.
Use the published branches `grammy117/tvm:gemmini/direct-intrinsics` and
`ucb-bar/merlin:comparisons/tvm-workload-tuning`, pinned by these commits.
The gzip-compressed patches retain exact changes from TVM 3d5e9c353 and Merlin f0b2d0681.
No model payload or ELF is supplied by this small verification product.

The new baseline favors large legal exact tiles and operand reuse. Its choices
are ordinary TensorIR traces and pass physical SRAM lowering. It is a heuristic,
not a cycle predictor. Explicit group settings remain available.

Checks (single-threaded BLAS, no native compiler rebuild):
- Schedule/TensorIR/graph-plan/ordered-database suites: 53 passed, 21 subtests.
- Workload-wrapper and tuning-library suites: 36 passed.
- Original operator inventory plus stock ReplayTrace/no-XGB controls: 3 passed,
  51 operator subtests. These check host/emulated semantics, not new Spike runs.
- FireSim local control tests: 57 passed.
- Structure gate: nine unchanged module-size failures in unrelated files.

`default-shapes.json` reports shape-derived group choices and allocated rows.
It is compile inspection, not measured performance. The reuse regression checks
numerical equality, exact replay, small SRAM and odd/one-row shapes; a reuse
case issues fewer emulator commands than the old one-group baseline.

The no-ML test rejects any XGBoost import, construction or update callback.
The wrapper/publication tests check that ReplayTrace records reach compilation
and retain the measured trace evidence. Inspect compressed logs for warnings.

Reproduce using the TVM Python tree at the pin, a compatible native libtvm,
and its pytest dependencies. Set OMP/OPENBLAS/MKL thread counts to one.
Run the changed `tests/python/relax/test_backend_contrib_gemmini_{sketch,tensorir,search,operator_inventory}.py`
and `test_apps_gemmini_{graph_plans,ordered_databases,workload_chunks,tuning_library}.py`
suites. Select the replay cases from the search suite for a no-XGB environment;
its explicit legacy evolutionary cases still require the optional XGBoost package.
Merlin controls are under `merlin/tests/gemmini/test_tvm_firesim_*.py`.

No FireSim performance is claimed. Rebuild the changed schedules, bind the actual
hardware/spec proofs, qualify remote guest checks/cancellation/occupancy, and
finish complete kernel-call builders before publishing benchmark results.
The older 71-case diagnostic Spike release retains its old source attribution.
