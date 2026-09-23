# AGENT.md — merlin/python/merlin/perf/deps

## Purpose

Dependence primitives: what an instruction defines and uses, what is live, and what that costs.

## Modules

- `liveness.py` — What each instruction defines and uses, what stays live, and how much register file that spends.
- `measured.py` — Confront the dependence graph with a per-cycle trace: which separations were real.
- `rocc.py` — Turn a decoded accelerator command trace into the def-use footprint a dependence graph needs.

## Gotchas

- `liveness.effects_of` reads a measured operand-direction model keyed on the ARCHITECTURAL register
  operands. On a command-driven accelerator those are host GPRs that every command reuses, so a graph
  built from them makes everything depend on everything. Use `rocc.py` there: the real dependences
  live in the on-chip addresses the command payload encodes, not in the registers carrying it.
- An address field carrying mode bits must have them stripped with a DERIVED mask, or the same tile
  addressed two ways looks like two tiles and the edge between them disappears silently. `rocc.py`
  refuses such a field rather than using it raw.
- A command that writes a destination an earlier command staged for it defines nothing of its own.
  Left unmodelled, a readout depends on the STAGER and may legally hoist above the writer -- an
  illegal order scored as legal and fast. `rocc.INHERITS_DESTINATION` declares those pairings.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
