# AGENT.md — merlin/experiments/gemmini_perf_bench

Cross-approach **Gemmini perf benchmark**: runs the kernel corpus through backends (golden /
generated / IREE-dialect) and reports cycles/GFLOPs. Consumes merlin (`scripts/run_perf_bench.py`
imports `merlin.targetgen`).

- **Tracked source**: `scripts/` (harness + reporting), `kernels/` (capsule test corpus:
  `capsule.yaml` + `capsule.interface.mlir` per kernel), tracked `reports/`.
- **Generated output**: runs → `runs/gemmini/perf-bench/`, plots → `artifacts/plots/gemmini/`
  (constants `RUNS`/`REPORTS` in `scripts/_pbcommon.py`).
- Reproduce: `python scripts/run_perf_bench.py --help`.
