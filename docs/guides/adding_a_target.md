---
title: Adding a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-07-22
related: [getting_started, targetgen, generated_target_repos]
code_refs: [src/merlin/targetgen]
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
4. **Supply target support out of tree.** Declare the hardware backend through
   `plugin.backend` in the provider's contract and select it with `MERLIN_TARGET_PATH`.
   Target-specific codegen, ABI interpretation and execution do not belong in Merlin's
   shared runtime. Support code is separate from the compiler candidate generated and
   evaluated by the phase workflow; a discovered backend is not a certified compiler.
5. Add target-specific conformance tests to the provider's `tests/`; shared interface
   regressions belong in the relevant `merlin/tests/<subsystem>/` bucket.

For a RoCC target, the backend exposes a `rocc_semantics` object with three methods:

- `isa_constants(target)` returns current declared/derived facts, including
  `CUSTOM_OPCODE`, `FUNCT3`, and `FUNCT_CLASS`.
- `decode_instruction(funct, rs1, rs2, isa)` returns `(class_name, decoded_fields)`.
  Each operand is a mapping containing `raw`, `kind`, `arg_index`, and `offset`;
  unresolved values stay unknown, never guessed.
- `instruction_funct(name, rs1, isa)` resolves an instruction class to its funct code
  and raises `ValueError` for invalid operand selectors or unsupported classes.

Merlin owns transport/SSA parsing and assembler round-trip checks, not accelerator
operand layouts. Missing semantics refuse execution. The local Gemmini companion
implements this interface; select its `merlin-support` root explicitly and supply
the provider's required RTL facts. Legacy in-tree support is not a fallback.

Capability manifest loading returns the declared `encoding` mapping unchanged: an
`addr_len` does not imply readout flags or an accumulator layout. A RoCC provider may
expose `encoding_fields(declared)` to complete target-specific derived fields for ISA
emission; errors propagate rather than silently emitting an incomplete result. This
hook must not mutate the declaration. Gemmini owns its `derived_readout_bits` helper
in OOT support; the former core import is retired.

## Reference

`examples/toy_npu/target/` is the canonical example: `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`.
