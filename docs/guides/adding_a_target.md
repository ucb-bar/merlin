---
title: Adding a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-22
related: [getting_started, targetgen, generated_target_repos]
code_refs: [merlin/python/merlin/targetgen]
---

# Adding a target

Toy/reference targets live in-tree under `merlin/targets/`. Serious targets should become external
repos or MLIR plugins.

## Prerequisites

**Shared base + the `targetgen` extra.** Complete the base install in
[Getting started](getting_started.md) (`uv sync --all-extras`, or `pip install -e '.[targetgen]'`). The
scaffold-generation steps below need **no** external toolchain. An RTL-grounded target additionally
needs the `circt_firtool` capability (`firtool`/`FileCheck` on PATH) to promote
`contracts/rtl_facts/facts.json`; confirm it with `check_repro_env.py`.

## Steps

1. Create `merlin/targets/<name>/` following the canonical per-target shape: `contracts/` and
   `generated/` are **required** (each with an `AGENT.md`; `generated/` is gitignored output). Add
   `docs/` / `examples/` only when you have real content — no empty stub dirs. The shared
   `merlin_iface` dialect spec is NOT per-target (it lives in `merlin/contract/`). RTL-grounded
   targets get a promoted `contracts/rtl_facts/facts.json` pin (see
   `merlin.targetgen.rtl.circt_introspect --promote`); scratch stays in `out/artifacts/cache/`.
2. Write `contracts/target_contract.yaml` (validate against
   `merlin/schemas/target_contract.schema.yaml`). If it advertises the tensor-resident interface
   (`features: [resident_packed_tensor|accumulator_commit, command_buffer]`, a `matmul` capability,
   and `ops`/`types`), the rest of the dialect is **data-driven**.
3. **Dialect is generated, not hand-written.** `synthesize_dialect_plan` derives
   `contracts/dialect_plan.yaml` from the contract's `ops`/`types` (the interface→target lowering is
   canonical), and `xdsl_dialects.targets.factory.build_dialect(name, plan=...)` synthesizes the IRDL
   op/type classes from that plan — no per-target dialect module. The target registry
   (`merlin.targetgen.target_registry`) resolves name → contract/plan/facts/backend. Run the
   `merlin-targetgen` CLI to write the codegen package to `out/artifacts/targets/<name>/`.
4. **What you still hand-write: the hardware backend.** The dialect + plan + lowering are generated,
   but the runtime **backend** (`merlin/python/merlin/runtime/backends/<name>*.py` — the real
   C/ISA codegen + execution) is not mechanizable and must be authored per target. This is the
   remaining gap to a fully one-command target.
5. Add conformance tests under `merlin/tests/conformance/` and per-target `tests/`.

## Reference

`merlin/targets/toy_npu/` is the canonical example: `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`.
