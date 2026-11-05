# TVM handoff regression and sanity checks (a26, 2026-11-04)

Run before the submission handoff of `comparisons/tvm-workload-tuning` and `grammy117/tvm` `gemmini/direct-intrinsics`. `summary.json` has the counts; the logs are next to it.

- Merlin F2-path tests (FireSim runner, kernels, shared chunks, ResNet, target, package session, payload): all pass.
- All Merlin TVM-comparison tests: 33 inherited failures, all in backlogged TinyLlama/SmolVLA session tests (missing optional packages, host BF16 dispatch guards).
- TVM Gemmini tests: the same 10 failures at the F2 pin and the tip (FASED helper removed, removed features, stale preset, one uninvestigated ValueError expectation).
- Gates: docs and doc paths pass; the remaining failures are inherited and in files this handoff did not touch.
- Reproduction smoke at the F2 pins: G01/T01 inputs byte-identical to the F2 study; adapter sources/headers/symbols identical to the F2 receipt (objects differ: GCC 13.2.0 here vs 16.1.0 on F2); G01/T01 baseline and tuned (F2 database) ELFs pass Spike and reject the wrong oracle with the plain libgemmini `ea8f7ed7` plugin (`simulator-a26`; the repaired-RNE plugin behind SPIKE_HANDOFF_EVIDENCE was not found on a26); the merged ResNet database equals the union of the ten owner databases. No cycle counts were reproduced (no F2 access).

Builds used for the smoke are under `out/build/baselines/tvm-gemmini/f2-repro-a26-20261104/` and `adapter-a26-f3cfd1a7d/` (not tracked).
