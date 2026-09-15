# AGENT.md — merlin/python/merlin/compile/scheduling

## Purpose

Scheduling passes that sit between "which tile fits" (`compile/capacity.py`) and a backend's
instruction packing: the ORDER block moves and computes are issued in, which operand stays resident,
and where each operand's region sits on chip.

## Modules

- `block_schedule.py` — Order the block moves and computes of a contraction over a target's on-chip
  stores.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- **Knobs, not constants.** The four choices this pass exposes (load on index change, lookahead depth,
  operand order/grouping, region placement) were each measured worth 1-24% of cycles on one target,
  and none has a value that wins everywhere: the best setting depends on the SHAPE and on the
  SUBSTRATE (one capsule reverses between a fast-memory simulator and the FPGA). Defaults are
  documented in `Knobs`; a target picks values by measurement. Never bake one into a backend.
- **Geometry is derived, never passed as a literal.** `Geometry.from_address_space` reads block edge,
  store rows, per-bank depth and accumulator rows from `targetgen.address_space`, i.e. from that
  target's own RTL facts. A quantity the facts cannot answer is a `BlockScheduleError`.
- **Every schedule is checked before it is returned.** `check_residency` refuses a schedule whose load
  overwrites a block a later compute still reads. This is not defensive: when two operand regions
  overlap (a deep reduction), one emission order is correct and another silently computes on the wrong
  bytes, and an in-order executor cannot tell them apart. Two hand-written variants this pass
  reproduces do emit that fault at a shape they were never run on.
- **Not part of the `merlin-compile` facade.** `merlin.compile_cli` re-exports everything the flat
  `compile/` modules define; this package's vocabulary is deliberately generic (`Load`, `Store`,
  `Geometry`), so it stays a subpackage and callers import `merlin.compile.scheduling`.

## Testing expectations

- `merlin/tests/infra/test_block_schedule.py` — knob semantics over TWO synthetic geometries (different
  block edge, store size, bank depth), the refusals, and mutation controls on the residency check.
- `merlin/tests/gemmini/test_block_schedule_matches_packages.py` — equivalence: for each measured
  package variant under `out/artifacts/targets/`, the pass with the matching knob values reproduces
  that package's own instruction stream op for op (42 shape/variant cells), and the cells it refuses
  are shown to be real hazards in the variant's own stream.
