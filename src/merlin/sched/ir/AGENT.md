# AGENT.md — merlin/python/merlin/sched/ir

## Purpose

The kernel IR `mk`: one fused group's schedule as counted loops over a target's schedule instructions.
A kernel names its DRAM tensor arguments and calls instructions whose operands are index expressions
over loop variables, byte pointers into those tensors, `NULL`, or floats.

## Files

- `expr.py` — index expressions: constants, loop variables, sums, products by a constant, and a
  `select` on one loop variable (last-tile remainders). Evaluate, render to IR text or C.
- `kernel.py` — `TensorArg`, `Ptr`, `NULL`, `Call`, `Loop`, `Kernel`; the canonical text form and its
  digest; target-free structural checks; enumeration of dynamic instances; per-instance concrete values.

## Invariants

- **The text form is the identity.** `Kernel.digest()` hashes `Kernel.text()`; floats print as exact hex.
  Two schedules are the same exactly when they print the same.
- **Target-neutral.** Nothing here knows an instruction. Instruction names are resolved against a
  `merlin.sched.isa.InstructionSet` by the checker and the emitter.
- **Operands stay small.** Anything a `select` on one variable cannot say is a loop transformation
  (split, peel), not a richer operand form.
