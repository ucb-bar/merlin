# AGENT.md — merlin/python/merlin/rvvgen

## Purpose

RVV target-package machinery: fork an iteration of the RVV codegen (a transform-dialect

## Modules

- `apply.py` — Apply an RVV package's codegen knobs to a workload build, via the existing build_app seam.
- `autotune.py` — ``merlin-rvv-autotune`` — AUTOMATICALLY enumerate, build, benchmark and rank impr forks.
- `beam.py` — Beam-search of forks — the engine that replaces hand-hunting levers. Each generation: expand
- `fork.py` — Versioned fork minting + lineage — forks ACCUMULATE under artifacts/targets/<target>/, never
- `from_strategy.py` — Render a transform-dialect RVV schedule FROM knobs, and mint a versioned fork package.
- `k1.py` — SpacemiT K1 board adapter — real RVV silicon (VLEN=256, Bianbu Linux/glibc).
- `mine.py` — ``merlin-rvv-mine`` — the deterministic mining driver that mints a versioned run.
- `registry.py` — Loader for ISOLATED, per-run RVV codegen packages.
- `report.py` — ``merlin-rvv-report`` — the auditable evidence chain for the kernel-mining -> compiler
- `runner.py` — certify_rvv — isolated, measured K-ladder for one (RVV package x workload), coupled across
- `sweep.py` — Parallel isolated certification of many (package, workload) cells, + ranking aggregation.
- `tuning_agent.py` — LLM-based ForkProposal proposer — the judgment alternative to the deterministic gap-router.
- `workloads.py` — Kernel-sized workload generator — single compiler-emitted ops at curated-kernel shapes.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
