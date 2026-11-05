---
title: Adding a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-09
related: [getting_started, targetgen, generated_target_repos]
code_refs: [src/merlin/targetgen, src/merlin/runtime/backends/chipyard_rocc.py, src/merlin/targetgen/plugins.py]
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
   Use the shared [host and device compilation boundary](../reference/host_device_compilation.md)
   for the host recipe and execution plan. Put the actual launch protocol and
   platform driver in the target package.
5. Add target-specific conformance tests to the provider's `tests/`; shared interface
   regressions belong in the relevant `merlin/tests/<subsystem>/` bucket.

**A data-only provider for a chipyard RoCC target.** When the target runs on chipyard's
toolchain (spike extension, Verilator harness, Merlin's GSIM harness), the provider need not ship
code. Its contract's `plugin` block names generic core modules (`plugin.backend:
merlin.runtime.backends.chipyard_rocc`, plus `plugin.rocc_semantics` / `plugin.rtl_checks`) and
an `isa_headers` spec (schema `merlin.isa_headers.v1`, `merlin.targetgen.isa_headers_spec`) that
locates the upstream bare-metal runtime by variable, pin and per-file sha256. The backend reads
`runner.toolchain`, `execution_capabilities`, `readout_semantics` and `counter_semantics` from the
contract, and generates a minimal facts header (`merlin.targetgen.isa_header_gen`) as the only
accelerator header on the include path; headers the spec lists under
`excluded_from_include_path` (a vendor kernel library) are refused. `examples/gemmini/support` is
the worked example; `build_tools/scripts/sync_support_contract.py <example>` regenerates its
contract copies from the reviewed ones.

For a RoCC target, the backend exposes a `rocc_semantics` object with three methods:

- `isa_constants(target)` returns current declared/derived facts, including
  `CUSTOM_OPCODE`, `FUNCT3`, and `FUNCT_CLASS`.
- `decode_instruction(funct, rs1, rs2, isa)` returns `(class_name, decoded_fields)`.
  Each operand is a mapping containing `raw`, `kind`, `arg_index`, and `offset`;
  unresolved values stay unknown, never guessed.
- `instruction_funct(name, rs1, isa)` resolves an instruction class to its funct code
  and raises `ValueError` for invalid operand selectors or unsupported classes.

Merlin owns transport/SSA parsing and assembler round-trip checks. Accelerator
operand layouts belong in independently derived external target support. Missing
semantics refuse execution. From-scratch experiments require issued RTL and runtime
authorities; a handwritten reference backend cannot supply this support.

Capability manifest loading returns the declared `encoding` mapping unchanged: an
`addr_len` does not imply readout flags or an accumulator layout. A RoCC provider may
expose `encoding_fields(declared)` to complete target-specific derived fields for ISA
emission; errors propagate rather than silently emitting an incomplete result. This
hook must not mutate the declaration. Gemmini owns its `derived_readout_bits` helper
in OOT support; the former core import is retired.

## Reference

`examples/toy_npu/target/` is the canonical example: `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`.
