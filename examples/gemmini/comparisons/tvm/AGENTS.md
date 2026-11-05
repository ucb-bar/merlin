# Working in the TVM comparison (2026-11-04)

The ResNet50 + 16-kernel campaign ran on AWS F2 FireSim on 2026-10-28.
Start with [README.md](README.md); the results and recipe are in
[REPRODUCING_F2_RESULTS.md](REPRODUCING_F2_RESULTS.md). Read
[ARCHITECTURE.md](ARCHITECTURE.md) and [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md)
before changing compiler behaviour, and the companion TVM `apps/gemmini/AGENTS.md`.

## Evidence rules

- The F2 numbers bind TVM `f3cfd1a7d`, Merlin `fe0306676`, GCC 16.1.0 and AGFI
  `agfi-02410ef1a8461b435` (F2, 60 MHz, DDR3+LLC). Never relabel them as the
  selected U250 30 MHz target or as results of a later source revision.
- The submodule `third_party/tvm` pins that F2 TVM revision. Move it only
  together with new measurements that bind the new revision.
- Frozen products under `out/artifacts/` keep their original source, input,
  compiler and hardware identities. Write new results to new products
  (`merlin.common.artifacts.new_product` / `new_measurement`); never edit a receipt.
- Report only measured cycles of rebuilt complete units. Never sum chunk or
  task minima into complete-call or end-to-end numbers. No cycles from estimates.
- Keep failures, timeouts, duplicates and unused budget in every report. A
  timeout is not a cycle label.
- Spike results are functional evidence only. The repaired diagnostic oracle is
  not the spec-pinned libgemmini provider.
- The sole model protocol is Merlin `8ec99308`; use its quality policy.

## Compiler rules

- Keep mathematical TensorIR, stock graph passes, stock MetaSchedule
  (ReplayTrace, JSONDatabase, MetaScheduleApplyDatabase) and declared
  TensorIntrins. Do not restore automatic bias folding, prepacking, scale/readout
  rewrites or DMA peepholes.
- No model name, captured ID or expected output may choose an optimization.
  Unsupported work stays on the CPU with a placement reason or is refused;
  kernel builds must not fall back to the CPU for the measured operation.
- Device code belongs in the TVM fork. This directory owns inputs, builders,
  runner orchestration and evidence.
- Full TinyLlama/SmolVLA remain backlogged. Their session tools stay here as
  future work, not launch instructions.

## Practicalities

- On the shared VM use bounded workers, single-threaded BLAS and the
  `tvm-tuning-env.sh` variables; check `tvm.__file__` and the loaded `libtvm`.
- Run the relevant tests (`merlin/tests/gemmini/test_tvm_*.py`, TVM
  `tests/python/relax/test_*gemmini*.py`) and
  `python build_tools/scripts/check_structure.py`. Report inherited failures
  separately (see [USAGE.md](USAGE.md#9-tests)).
- No sudo, user namespaces, destructive actions, pushes or PRs without the
  user's approval. Preserve other worktrees.
