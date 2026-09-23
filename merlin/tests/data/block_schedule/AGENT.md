# AGENT.md — merlin/tests/data/block_schedule

## Purpose

Recorded instruction streams that hold `merlin.compile.scheduling` to the hand-written Gemmini
package variants it was lifted from, on a checkout where those packages are absent.

## What belongs here

- `golden_streams.json` — per measured variant × corpus shape: the package stream's decoded block
  moves as an op count and sha256 digest, the pass's own digest (or `refused` plus the hazard indices
  that make the refusal real), the geometry it was recorded against, each package's baked geometry
  constants, and the sha256 of the `lowering/isa.py` each stream came from.

- `atlas_declared_elaboration_facts.json` — Atlas facts as the generator-scoped FIRRTL census reads them
  from its declared elaboration (`AtlasRocketConfig`): arrays, SRAM memories (VMEM, the two accumulation
  buffers, the matrix register file, the instruction memory) and the compute cell's datapaths. Lets the
  address-space derivation be tested on real RTL structure without an RTL checkout. It is NOT the shared
  facts cache; regenerate it from `circt_introspect.dump_facts(target="atlas")` and keep only `facts`.

## What does not belong here

- The packages themselves (they live, untracked, under `out/artifacts/targets/gemmini/`).
- A golden produced from the pass alone. It is a record of AGREEMENT between pass and package.

## Invariants

- Regenerate only with
  `.venv/bin/python merlin/tests/gemmini/test_block_schedule_matches_packages.py --regen`, which refuses
  when any variant is missing or any non-refused cell disagrees with its package.
- Never hand-edit a digest to make a cell pass: a changed digest means the pass or a package moved.
