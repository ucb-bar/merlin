# AGENT.md — merlin/experiments

## Purpose
In-repo experiments + benchmark harnesses. They **consume** merlin (add `merlin/python` to
`sys.path`); **nothing in the library depends on them** (one-way — safe to move/prune).

- Small workstream experiments: `kernel_policy/`, `gemmini_cert/`.
- Benchmark harnesses: `capsule_bench/` (the multi-target capsule benchmark — six targets, one
  harness), `agent_bench/` (target-agnostic reference scaffold), `gemmini_perf_bench/`,
  `muon_perf_bench_v0/`, `targetgen_evals/` (import-isolated; 0 real runs — do not cite it).
- Cross-compiler studies: `llm_kernel_vs_compiler_v0/`, `voyager_h2h/` (merlin vs the Voyager
  compiler, same-hardware and home-turf planes).

## Status (enforced)
Every experiment's AGENT.md carries a `Status:` line in its first 15 lines: `active`, `frozen`
(finished; its results are in `FINDINGS.md`, which must exist) or `reference` (a scaffold other
experiments copy, with no runs of its own). `check_structure.py` "experiment status" enforces it.
There is no `retired` status: retiring an experiment means deleting its directory, and git history is
the archive. Decide by citation, not commit age -- a quiet experiment a guide still cites is frozen,
not abandoned.

## What lives here (curated inputs only)
- Task specs, `input_bundles/`, method specs, per-target guides, and the harness drivers that run them.
- Kernel/capsule corpora that are experiment-specific inputs.

## What does NOT belong here
- **Reusable library code** → lift into `merlin/python/merlin/`.
- **Generated output** → `out/runs/<target>/<suite>/` (runs) or `out/artifacts/` (products). Never
  in-tree — the `check_artifact_layout` gate forbids `experiments/*/reports/` and
  `experiments/*/runs/`. The top-level `runs/`/`artifacts/`/`build/` roots are retired.

## The rule (consumption direction)
Experiments **only consume** the library; **nothing in `merlin/python/merlin/` may read an
`experiments/` path** (one-way). If the library needs an input an experiment currently holds (a corpus,
kernels), that input is a benchmark — move it to `benchmarks/` (or `contract/`) and have the harness
reference it by location. Enforced by `check_structure.py` "library boundary".

## Invariants
- Resolve the repo root by discovery (`git rev-parse --show-toplevel` / walk to `merlin/python`),
  never hardcoded `parents[N]` or absolute paths.
- `targetgen_evals/` is import-isolated by design (zero `merlin.*` imports) — keep it that way.
- `agent_bench/` is the clean target-agnostic model the other benches should converge toward.

## In progress
The gemmini/muon/perf harnesses are being unified onto a shared, target-parametric
`merlin.benchharness` package (WS2) — the cloned `_common.py`/`_pbcommon.py`/`run_muon_*` collapse to
one harness + thin per-target config; the sandbox capsule view will be materialized from
`merlin/contract/capsules` instead of the committed `full_public_capsules/` copy.
