# AGENT.md — merlin/experiments

## Purpose

In-repo experiments and benchmark harnesses for the workstreams. Two flavors live here:

- **Small workstream experiments** (configs, notes, small drivers): `targetgen_toy/`,
  `kernel_policy/`, `semantic_memory/`, `gemmini_cert/`.
- **Benchmark harnesses** (relocated off the repo root): `agent_bench/`,
  `gemmini_capsule_bench_v0/`, `gemmini_perf_bench/`, `muon_perf_bench_v0/`,
  `targetgen_evals/`. These *consume* merlin (add `merlin/python` to `sys.path`); nothing in
  the library depends on them.

Each dir has a `README.md`/`AGENT.md` (and often a `reports/METHODOLOGY.md`) describing its
question and how to reproduce it.

## What belongs here

- Experiment configs, task specs, input bundles, kernel corpora, methodology writeups, and the
  harness drivers that run them.

## What does not belong here

- **Reusable library code** — lift it into `merlin/python/merlin/`.
- **Generated output** — runs go to `runs/<target>/<suite>/`; other products to `artifacts/`
  (three-root convention, see CLAUDE.md). Never `output/` (deprecated). Per-experiment `runs/`
  dirs are gitignored.

## Invariants

- Resolve the repo root by discovery (walk up to `merlin/python`, or `git rev-parse
  --show-toplevel`), never a hardcoded `parents[N]` (harnesses live several levels deep).
- `targetgen_evals/` is import-isolated by design (zero `merlin.*` imports — it evaluates merlin
  as an external subject); keep it that way.

## Notes for future agents

Keep experiments self-describing so other sessions can rerun them; lift any logic that becomes
reusable into the library.
