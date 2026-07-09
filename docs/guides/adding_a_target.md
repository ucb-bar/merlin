---
title: Adding a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-07
related: [targetgen, generated_target_repos]
code_refs: [merlin/python/merlin/targetgen]
---

# Adding a target

Toy/reference targets live in-tree under `merlin/targets/`. Serious targets should become external
repos or MLIR plugins.

## Steps

1. Create `merlin/targets/<name>/` following the canonical per-target shape: `contracts/` and
   `generated/` are **required** (each with an `AGENT.md`; `generated/` is gitignored output). Add
   `docs/` / `examples/` only when you have real content — no empty stub dirs. The shared
   `merlin_iface` dialect spec is NOT per-target (it lives in `merlin/contract/`). RTL-grounded
   targets get a promoted `contracts/rtl_facts/facts.json` pin (see
   `merlin.targetgen.rtl.circt_introspect --promote`); scratch stays in `artifacts/cache/`.
2. Write `contracts/target_contract.yaml` (validate against
   `merlin/schemas/target_contract.schema.yaml`).
3. Run TargetGen to produce `contracts/dialect_plan.yaml` and a dialect scaffold
   (the `merlin-targetgen` CLI, writing the codegen package to `artifacts/targets/<name>/`).
4. Prototype the dialect in xDSL first (`merlin/python/merlin/xdsl_dialects/` /
   `targetgen/backends/xdsl_backend.py`); promote to MLIR/C++ only when stable.
5. Add conformance tests under `merlin/tests/conformance/` and per-target `tests/`.

## Reference

`merlin/targets/toy_npu/` is the canonical example: `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`.
