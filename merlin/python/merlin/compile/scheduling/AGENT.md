# AGENT.md — merlin/python/merlin/compile/scheduling

## Purpose

Scheduling passes that sit between "which tile fits" (`compile/capacity.py`) and a backend's
instruction packing: the ORDER block moves and computes are issued in, which operand stays resident,
and where each operand's region sits on chip.

## Modules

- `block_schedule.py` — Order the block moves and computes of a contraction over a target's on-chip
  stores.
- `derive.py` — Derive a block-schedule Geometry from a target's facts.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- **Knobs, not constants.** Load on index change, lookahead depth, operand order/grouping, region
  placement and loop order were each measured or shown to move cycles, and none has a value that wins
  everywhere: the best setting depends on the SHAPE and on the SUBSTRATE. Defaults are documented in
  `Knobs`; a target picks values by measurement. Never bake one into a backend by hand -- mint it
  (`merlin/experiments/scheduled_backend_v0`).
- **Geometry is derived, never passed as a literal.** `derive.geometry_from_address_space` reads block
  edge, store rows, per-bank depth and accumulator rows from `targetgen.address_space`. Stores are
  resolved to roles by ROW WIDTH (`address_space.operand_store`), and an accumulator is scheduled into
  only when `address_space.accumulator_kind` says it is ADDRESSABLE -- an in-datapath accumulator has no
  rows, and its depth is not in the facts. A quantity the facts cannot answer is a `BlockScheduleError`.
- **`block_schedule.py` imports nothing from merlin.** Generated backends may not depend on merlin at run
  time, so they vendor this module byte for byte; the package integrity scan rejects a merlin import.
  Anything that needs merlin goes in `derive.py`. Pinned by `test_block_schedule.py`.
- **One nest for every contraction.** Matmul and convolution share `_schedule_nest`; a convolution is a
  contraction whose streamed blocks are GATHERED (per-row DRAM sources, `None` = the target's zero path).
  A load is due the first time a compute needs it (per nest with `load_on_index_change`, per emission
  group without) -- no loop is named as the one whose index change triggers it, so every `loop_order`
  is correct.
- **Every schedule is checked before it is returned.** `check_residency` refuses a schedule whose load
  overwrites a block a later compute still reads. The block identity includes its gather: the same block
  position with different gathered pixels is different bytes. This is not defensive: when two regions
  overlap, one emission order is correct and another silently computes on the wrong bytes.
- **Every schedule is executable.** `execute` runs the ops at the addresses they name on host arrays;
  the result must equal `lhs @ weight` (or the direct convolution). That is the pass's own oracle.
- **Pooling is not modelled.** It drains the accumulator N-major and defers every drain past the nest --
  a different accumulator layout, refused rather than approximated.
- **Not part of the `merlin-compile` facade.** This package's vocabulary is deliberately generic
  (`Load`, `Store`, `Geometry`); callers import `merlin.compile.scheduling`.

## Testing expectations

- `merlin/tests/infra/test_block_schedule.py` — knob semantics over two synthetic geometries with
  neutral store names, every loop order and conv shape executed exactly, the refusals, mutation controls
  on the residency check and the executor, and that the module stays vendorable.
- `merlin/tests/targetgen/test_store_roles.py` — role resolution, accumulator kind and declared counts,
  on synthetic facts and on the real targets.
- `merlin/tests/gemmini/test_block_schedule_matches_packages.py` — the pass reproduces each measured
  package variant's stream (matmul and the zero-path conv), held to a checked-in golden so it runs on any
  checkout.
- `merlin/tests/gemmini/test_scheduled_package.py` — packages MINTED from the pass emit the hand-written
  packages' raw instruction streams, op for op.
