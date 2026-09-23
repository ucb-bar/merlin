# AGENT.md — merlin/python/merlin/sched/isa

## Purpose

The shape of a target's schedule instruction set: `Operand`, `InstrDef`, `InstructionSet`, and
`instruction_set(target)`, which asks the target's backend hook `sched_instruction_set()` for it.

## Invariants

- **Derived by the target, defined here.** This package holds no instruction. A target builds its
  instruction set from its own sources (header macros, RTL facts) and records their digests in
  `provenance`.
- **Checks see concrete values.** `check(values, state)` and `footprint(values)` receive one dynamic
  instance's evaluated operands; `state` is shared across one kernel's instances in program order, for
  rules that span instructions.
- **Identity.** `InstructionSet.digest()` covers operand lists, facts and provenance; a measurement
  keyed on a schedule also keys on this.
