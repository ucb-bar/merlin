---
title: Target generation
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-07
related: [adding_a_target, experiment_abi, generated_target_repos]
code_refs: [merlin/python/merlin/targetgen]
---

# TargetGen (Workstream 1)

Pipeline:

```
local docs / examples / Scala-Chisel source
  -> source manifest (ingest)        what we were pointed at; URLs recorded, not crawled
  -> evidence_report.md + index      files discovered + keyword-detected concepts (cited)
  -> the five plans (synthesize)     target_contract / dialect / runtime_adapter / zephyr / llvm
  -> merlin-target-<name>/ skeleton  (generate: xdsl, mlir, runtime adapter, zephyr, llvm)
  -> validation report               (validate: schemas + structural checks)
```

The realistic goal is **structured, human-reviewable scaffold generation with validation
gates**, not "RTL to correct dialect." TargetGen is deterministic and contains no LLM calls.

## Modules

`merlin/python/merlin/targetgen/`:

- `cli.py`, `pipeline.py` — entry points (`build`, `inspect`).
- `ingest/` — `SourceManifest`; `docs.py`/`examples.py`/`scala_chisel.py` discover files.
- `evidence/` — `report.py` (build + render) and `store.py` (data structures).
- `synthesize/` — one module per plan, each returning a schema-shaped dict.
- `generate/` — `target_repo.py` (skeleton/AGENT.md/contracts/docs/backfill), `xdsl.py`,
  `mlir_scaffold.py`, `runtime_adapter.py`, `zephyr_module.py`, `llvm_plan.py`.
- `validate/` — `schemas.py`, `generated_repo.py`, `report.py`.

## Reference target

`merlin/targets/toy_npu/` — concrete `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`. Real targets (gemmini/saturn/radiance) synthesize
conservative skeletons flagged `requires_human_review: true`. See `docs/adding_a_target.md`
and `docs/generated_target_repos.md`.

## Commands

```bash
python -m merlin.targetgen.cli build --target-name toy_npu \
  --source-dir merlin/targets/toy_npu/docs \
  --examples-dir merlin/targets/toy_npu/examples \
  --out build/generated/merlin-target-toy-npu \
  --emit xdsl,mlir,zephyr,llvm-plan,runtime
python -m merlin.targetgen.cli inspect --target build/generated/merlin-target-toy-npu
```

`--emit` is a comma list of `xdsl,mlir,zephyr,runtime,llvm-plan`, or `contract-only`. The
console script `merlin-targetgen` is equivalent. Generated repos are written under the
gitignored `build/generated/`.

## Must not

Build a generator with no target example; parse arbitrary RTL as if semantics are obvious;
derive dialect ops directly from instruction names; modify kernel-mining or DSE logic except
through shared schemas; claim synthesized artifacts are correct.
