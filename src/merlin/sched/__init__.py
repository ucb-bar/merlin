"""Schedule-level compiler core: numerics contracts, kernel schedules, and the gates that check them.

The package is target-neutral. Every fact about a target (readout arithmetic, capacities, instruction
semantics) arrives as data derived from that target's own sources and is passed in by the caller; no
module here names a target. Subpackages:

- :mod:`merlin.sched.contract` -- numerics contracts: the declared arithmetic a golden is computed
  under (scale granularity, rounding, saturation, activation), built from derived readout facts.
- :mod:`merlin.sched.check` -- correctness gates that do not need a simulator (e.g. exhaustive
  epilogue enumeration).
"""
