# AGENT.md — merlin/python/merlin/sched/codegen

## Purpose

Emit a `mk` kernel as one C function: counted loops around each call's statement, which the target's
`InstrDef.render_c` produces. Pointers render as byte offsets from their tensor argument, floats as exact
float32 hex literals, and the body ends with the instruction set's `drain_c`.

## Invariants

- **No instruction knowledge.** Every statement text comes from the target's instruction set.
- **Refuse, never guess.** An operand list that differs from the instruction's, or a dtype with no C
  type, raises `IsaError`.
- **Version.** `EMITTER_VERSION` changes whenever the text emitted for an unchanged kernel changes;
  measurements key on it.
