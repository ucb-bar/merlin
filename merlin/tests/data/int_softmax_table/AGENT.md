# AGENT.md — merlin/tests/data/int_softmax_table

## Purpose

Captured integer-softmax modules that `merlin/tests/ir/test_int_softmax_table.py` rewrites and
executes before and after the `int_softmax_table` lowering rewrite.

## What belongs here

- Modules exactly as the capture worker emits them (`PROVENANCE.json` names the capture sources and
  the model2MLIR pin). Regenerate them with `merlin/tests/fixtures/capture_integer_softmax_ir.py`,
  never by hand: the rewrite matches the captured spelling, and an edited fixture tests nothing.

## Invariants

- Small enough to execute in a test; the shapes are chosen for what they exercise (every grid index,
  a masked bf16 attention, flat 1024-key rows whose quantization scale clamps to its eps).
