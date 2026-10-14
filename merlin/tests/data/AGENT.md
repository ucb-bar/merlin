# AGENT.md — merlin/tests/data

## Purpose

Small, checked-in INPUTS that tests read: trimmed kernel snippets for the kernel-mining extractors,
lit MLIR, recorded RTL/certification vectors, and tiny whole-model bundles (`mlp_bundles/`). They are
inputs, not results -- what a test or tool produces when it runs goes under the `out/` root.

## What belongs here

- Tiny trimmed kernel snippets and MLIR used as deterministic test inputs.
- Small recorded vectors / model bundles a test cannot cheaply produce itself (e.g. one that needs
  another repo's toolchain). Say in the subdirectory's AGENT.md where it came from and how to
  regenerate it.

## What does not belong here

- Full external kernels or large corpora (those are passed by path/env at runtime).
- Generated outputs (write them under the `out/` root via `merlin.common.paths`).

## Invariants

- Inputs are minimal and self-explanatory; they exist to pin behavior.
- Anything produced by a tool is regenerated with that tool, never hand-edited; where the source can
  drift (another repo's output), a test should check the snapshot still matches it.
