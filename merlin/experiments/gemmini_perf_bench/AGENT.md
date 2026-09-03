# AGENT.md — merlin/experiments/gemmini_perf_bench

Cross-approach **Gemmini perf benchmark**: runs the kernel corpus through backends (golden /
generated / IREE-dialect) and reports cycles/GFLOPs. Consumes merlin (`scripts/run_perf_bench.py`
imports `merlin.targetgen`).

- **Tracked source**: `scripts/` (harness + reporting), `kernels/` (capsule test corpus:
  `capsule.yaml` + `capsule.interface.mlir` per kernel), tracked `reports/`.
- **Generated output**: runs → `runs/gemmini/perf-bench/`, plots → `artifacts/plots/gemmini/`
  (constants `RUNS`/`REPORTS` in `scripts/_pbcommon.py`).
- Reproduce: `python scripts/run_perf_bench.py --help`.
- **Functional input gate**: the campaign consumes exactly one frozen functional submission
  (`scripts/perf_campaign.py:inspect_functional_run`). A run that predates the immutable
  bundle-input snapshot v2 schema is re-verified — snapshot re-materialized, public + hidden
  grades re-run at L3 against the same submission bytes — by
  `scripts/refreeze_functional_run.py --source-run-id <run> --new-run-id <run>_refreeze_<date>`.
  It carries the original run's authoring provenance and never back-fills evidence.
