# AGENT.md — merlin/targets/gemmini_universal/generated

## Purpose

Generated scaffolds for the `gemmini_universal` target (Jack's "Universal" ResNet-50 Gemmini). Empty by
design and kept out of git except this file and `.gitkeep`: nothing has been generated for this device
yet, and its contract deliberately declares no `plugin.backend` because writing a backend for it is the
work under test.

## What belongs here

- Tool-generated per-target scaffolds, if and when a generator emits any.

## What does not belong here

- The target's reviewed contract, RTL-facts pin or ABI header — those are hand-reviewed inputs and live
  in `../contracts/`.
- `gemmini`'s backend, or anything copied from it. That codegen assumes a full-width accumulator
  readout, which this device does not have.
- Generated outputs (write under `out/runs/` / `out/artifacts/`; compiled trees to `out/build/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
