# AGENT.md — merlin/python/merlin/targetgen/rocc

## Purpose

The RoCC coprocessor interface: decoding a trace of it, and emitting one.

## Modules

- `asm.py` — Derived RoCC assembler for ``inline_asm_insn`` targets.
- `decode.py` — Decode a target's RoCC instruction trace from a package's emitted ``lowered.llvm.mlir``.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

**`target` is required and has no default.** The selected backend must expose `rocc_semantics`:
`isa_constants(target)`, `decode_instruction(funct, rs1, rs2, isa)`, and
`instruction_funct(name, rs1, isa)`. Operands are plain mappings with `raw`, `kind`,
`arg_index`, and `offset`. Accelerator ABI interpretation belongs in that OOT support
capability, not in shared transport parsing. Missing capabilities refuse; there is no
fallback to another target's decoder. The provider must declare or derive its ISA facts.

**Fail closed, never drop.** An instruction form the decoder does not understand is recorded as class
`UNKNOWN` so `trace_check` can reject it. Silently skipping unparsed input is how a conformant backend
gets mis-measured: a too-narrow match once dropped every `.insn` in a trace and produced an empty one,
which read as "this backend emitted nothing" rather than "the decoder could not read it".

**Decode is runner-owned.** The trace is a measurement of what the package actually emitted, not an
artifact the package hands over — that is what makes it parity-clean between a baseline and an
assisted submission.

## Shared parsing, external semantics

Core retains MLIR/SSA resolution, R-type transport parsing, fences, UNKNOWN records and trace
construction. CONFIG subtypes, packed addresses and transfer/compute meaning are not common
RoCC semantics. Gemmini's implementation lives in its companion support package.
Only parsed syntax is cached; every decode resolves support and reads facts again. Returned
traces never alias cache-owned dictionaries. Loaded-provider ownership checks still apply;
this is not an attestation of arbitrary in-process mutation or complete source bytes.
Assembly resolves an omitted kernel symbol through the selected contract's `harness_abi`.

Structural checking is a separate `rocc_semantics.rtl_checks` capability. It exposes
`load_default_facts(target)`, `project_facts(facts_rec)`,
`screen(trace, capsule, rtl_facts, *, target, command_buffer)`,
`compile_trace_checks(facts_rec, capsule, prefix)` and `render_trace(trace, facts_rec)`.
Reports use Merlin's shared `Check`/`CheckReport` types. Specialized geometry and
instruction ordering remain support-owned; missing coverage is not a passing check.
