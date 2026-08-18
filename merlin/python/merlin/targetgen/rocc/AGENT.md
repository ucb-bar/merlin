# AGENT.md — merlin/python/merlin/targetgen/rocc

## Purpose

The RoCC coprocessor interface: decoding a trace of it, and emitting one.

## Modules

- `asm.py` — Derived RoCC assembler for ``inline_asm_insn`` targets.
- `decode.py` — Decode a target's RoCC instruction trace from a package's emitted ``lowered.llvm.mlir``.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

**`target` is required and has no default.** Every encoding fact — custom opcode, funct3, the
funct-to-class table, operand field positions — is read from that target's RTL-derived facts. This is
the one rule worth defending here: a default would let a decoder read one accelerator's trace against
another's table and report a clean result, which is worse than an error because it is quotable.

**Fail closed, never drop.** An instruction form the decoder does not understand is recorded as class
`UNKNOWN` so `trace_check` can reject it. Silently skipping unparsed input is how a conformant backend
gets mis-measured: a too-narrow match once dropped every `.insn` in a trace and produced an empty one,
which read as "this backend emitted nothing" rather than "the decoder could not read it".

**Decode is runner-owned.** The trace is a measurement of what the package actually emitted, not an
artifact the package hands over — that is what makes it parity-clean between a baseline and an
assisted submission.

## Why these stay in-tree

RoCC is a RISC-V standard interface, not one accelerator's ABI, so this is not a reference-target
module awaiting eviction. Four generic core modules consume it (the capsule runner, the CIRCT gate and
introspection, and the assembler); evicting it would break the core rather than de-couple it.

Known residue: `asm.assemble_program` still defaults `kernel_symbol` to one target's entry symbol.
It cannot simply become required — every caller relies on it — and the contract key it should read
does not exist yet. It is the same key the contract-compile harness weld needs, so the two are one
piece of work (see `merlin/contract/overfit_register.yaml`).
