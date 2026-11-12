"""Kernel IR and the scheduling language: what a kernel does, and the moves that reschedule it.

Target-neutral throughout: every fact about a target (instructions, memories, readout arithmetic,
capacities) arrives as data derived from that target's own sources and is passed in by the caller.

- :mod:`.ir` -- the kernel IR (index expressions, kernels, stages, completion events, instructions).
- :mod:`.parse` -- read a kernel back from its canonical text.
- :mod:`.primitives` and :mod:`.loops` -- the scheduling language: semantics-preserving rewrites.
- :mod:`.machine` -- the machine a schedule is written against, derived from target declarations.
- :mod:`.checks` -- simulator-free correctness checks, numerics contracts and cost kinds.
- :mod:`.codegen` -- emit a kernel as C; reroll encoded command runs into counted loops.
- :mod:`.expressiveness` -- compare an expert corpus and our schedules on capability axes.
"""
