# AGENT.md — packages/merlin-experiments/src/merlin_experiments/timing_probes

Operator calibration stage between Phase 1 and Phase 2: derive, run and archive the smallest set of
timing probes that makes the timing model's parameters identifiable over the Phase 2 domain.

- `space.py` resolves a target's timing space (its data directory's `timing_space.yaml`) against the
  selected RTL facts and the timing provider's profile. Funct codes, field positions and sizes come
  from the facts; a name the facts lack leaves its class or quantity UNKNOWN with the reason.
- `decode.py` turns any command stream (archived static trace, emitted probe) into classes and
  whole-call feature columns. `coverage.py` reads timing archives without extracting them.
- `domain.py` turns the Phase 2 corpus ledger into quantity ranges; unquantified entries are counted.
- `selection.py` derives requirements (presence, pooled-variant coverage, two points, rank
  identifiability, domain range, the fit's own `merlin_timing.fitting.design_check`) and greedily
  covers them with backend-realized requests. Every chosen probe lists what it covers; every gap
  left says why.
- `backends.py` loads the per-target backend the timing space names, from its own directory only.
- `runner.py` builds with the backend's `harness_build_recipe`, runs its selected gSIM command (the
  capsule L3 path), checks determinism, caches by hardware identity and writes the label archive.

Never name a target here; select it by parameter and data. Never grant this stage's inputs or
outputs to an author, stage them into a release, or feed them to a grade. Outputs are 0700.
