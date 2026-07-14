---
title: merlin/ layout — what goes where
kind: reference
status: current
owner: core
last_verified: 2026-07-14
related: [repo_structure, architecture]
code_refs: [merlin/benchmarks, merlin/contract, merlin/experiments, merlin/prompts, merlin/runtime, merlin/schemas, merlin/targets]
---

# merlin/ layout — what goes where

`merlin/` holds **reusable code + schemas + curated INPUTS that cannot be regenerated**. Anything
generated/derivable lives under the single `out/` root's three subdirs (`out/runs/`, `out/artifacts/`,
`out/build/`) — see [Repository structure](repo_structure.md) and CLAUDE.md. Each folder's `AGENT.md`
is the local contract; this table is the map.

| Folder | Holds (curated input / code) | Generated equivalent lives in | Read by |
|---|---|---|---|
| `python/merlin/` | the importable library (all reusable code) | — | everything |
| `schemas/` | cross-workstream YAML data-model (`*.schema.yaml`) | — | `merlin.common.schemas`; pinned by `REQUIRED_SCHEMAS` |
| `contract/` | experiment-ABI **data** + capsule corpus (gemmini reference) | results → `out/artifacts/` | `merlin.targetgen.contract` (`contract_dir()`), capsule-bench |
| `benchmarks/` | workload corpus + measured data (DSE inputs) | oversized captures → `out/artifacts/recaptures/`; results → `out/artifacts/dse-guidance/` | `merlin.dse_guidance`, `merlin.kernels.validate` |
| `runtime/` | target-independent C runtime substrate + one spike backend | compiled objects/ELFs → `out/build/` | `runtime.backends.*`, `rvvgen.k1`, `baselines.buddy` |
| `targets/` | hand-authored reference target defs (toy_npu/gemmini/saturn) | codegen packages → `out/artifacts/targets/`; RTL scratch gitignored | `xdsl_dialects.targets.*`, lowering, `targetgen` |
| `prompts/` | versioned RVV agent-instruction templates | — | `kernels.agent_mine`, `rvvgen.tuning_agent` |
| `experiments/` | task specs, input bundles, harness drivers (consume merlin) | runs → `out/runs/`; reports → `out/artifacts/` (enforced) | nothing in the library (one-way) |
| `tests/` | the suite + fixtures/data (golden inputs) | — | pytest |

## Rules of thumb
- **Generated or specific?** → not in `merlin/`. Put it in `out/artifacts/` (products), `out/runs/` (runs), or
  `out/build/` (compiled). The `check_artifact_layout` gate enforces this (forbids `experiments/*/reports/`,
  `benchmarks/*/case_study/`, etc.).
- **Curated input the library reads?** → `benchmarks/` (workload corpus) or `tests/data/` (fixtures).
- **A reference target instance** (gemmini/saturn) is fine under `targets/`; keep its specifics out of
  general machinery (resolve via params/specs, e.g. `$MERLIN_RTL_FACTS`, not hardcoded defaults).
- **Provenance**: if a corpus is regenerable, say so (e.g. `contract/capsules/MANIFEST.yaml`,
  `benchmarks/dse_guidance/recaptures_levels/REGEN.md`) so an agent knows what's frozen vs derivable.
