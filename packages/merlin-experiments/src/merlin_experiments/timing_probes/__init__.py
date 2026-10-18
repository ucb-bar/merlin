"""Operator timing-calibration probes, derived from a target's own timing space and data.

This package is the target-neutral half of the calibration stage that runs after Phase 1 and before
Phase 2. It reads, as data, the parameter space of the selected timing provider (its profile and the
target's decode of instructions into price classes), the labels already measured (timing archives),
and the Phase 2 operation domain; it selects the smallest set of probe requests that makes every fit
parameter identifiable over that domain, explains each choice, and runs the realized programs on the
selected cycle-accurate engine. A per-target probe backend, loaded from the target's data directory,
turns a request into a program. Nothing here names a target, an opcode or a field position.

The probes are operator reference programs. They are never granted to an author, never part of a
corpus or release payload, and never graded.
"""
