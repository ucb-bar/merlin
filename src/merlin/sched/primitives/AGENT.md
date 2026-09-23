# AGENT.md — merlin/python/merlin/sched/primitives

## Purpose

The scheduling language: the moves a schedule is written in. `merlin.sched.ir` says what a kernel does
and `merlin.sched.mach` says what the hardware is; this is the vocabulary that turns one kernel into
another without changing what it computes.

## Files

- `__init__.py` — `Cursor`, the two failure kinds, `proof_of`, and the primitives themselves
  (`divide_loop`, `unroll`, `reorder`, `fuse`), plus the shared obligation every rewrite discharges.

## Invariants

- **A failure has two kinds.** `NotApplicable` means this move does not apply here, so a search may try
  the next one; `PrimitiveError` means the schedule is wrong, so it must stop. It is a subclass, so a
  caller that does not know the difference still fails safe.
- **A cursor is a path, re-resolved at apply time.** A kernel's identity is its text, so a live handle
  into one means nothing after the first rewrite. A cursor that resolves to a different instruction than
  it asserted is a DEFINITE failure: the caller reasoned about another statement.
- **Every primitive declares its obligation, and the declaration is read, not authored.** The coverage
  register takes each axis' obligation from the primitive that carries it, so a row cannot claim a
  stronger one than the code discharges.
- **The shared obligation is the instance multiset, never the order.** A rewrite must preserve WHICH
  dynamic instances run with which operand values; changing when they run is the whole point. That is
  what catches the dropped, duplicated or mis-indexed instance a hand-written index rewrite produces.
- **Refuse rather than invent.** A split that would need a tail, an imperfect nest, two loops of
  differing extents: each refuses. Inventing the missing piece is how a schedule silently stops
  computing what it did.
- **Target-neutral.** Nothing here names a target. What a machine allows comes from `merlin.sched.mach`,
  which is derived from the target's own data.
