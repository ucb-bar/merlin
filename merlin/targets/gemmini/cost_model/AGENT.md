# AGENT.md — merlin/targets/gemmini/cost_model

## Purpose

This target's calibrated per-command linear cost model — the DATA half of
`merlin.perf.linear_cost.LinearCostModel` (the target-agnostic regressor). Callers never import
anything from here: they resolve it by target name through
`merlin.perf.linear_cost.cost_model_artifact(target)` / `LinearCostModel.for_target(target)`.

## What lives here

- `coefficients.json` — the fitted model (`const`, `coeff`, `error` band, `meta`). Written by
  `calibrate.py`; a bare `LinearCostModel.load(path)` reads only this file.
- `vocabulary.json` — the declared command vocabulary in summation order (`events`) and the `folds`
  (a command kind priced as a multiple of another, scaled by a ratio of RTL-fact datapath widths —
  never a literal). Hand-authored; `for_target` reads it together with the coefficients.
- `calibrate.py` — the calibration driver: builds `calib/` microbenchmarks and the
  `merlin/benchmarks/cost_calib/` ablations for this target, runs them on the cycle-exact Verilator
  sim, fits, validates. Run `python merlin/targets/gemmini/cost_model/calibrate.py` (`--refit
  <calibration.json>` refits offline, no RTL).
- `calib/` — the calibration microbenchmark sources.

## Invariants

- Fidelity is serial with no overlap: screening only, never a certified cycle count.
- Reports go to `out/artifacts/cache/cost_model/` (via `cache_dir`), never here.
- Every subdirectory must also contain an AGENT.md.
