# AGENT.md — merlin/python/tests/data

## Purpose

Small, checked-in fixture inputs for kernel-mining unit tests.

## What belongs here

- Tiny (≤~60 line) trimmed kernel snippets used as deterministic test inputs.

## What does not belong here

- Full external kernels or large corpora (those are passed by path/env at runtime).
- Generated outputs (use gitignored `output/`).

## Invariants

- Fixtures are minimal and self-explanatory; they exist to pin extractor behavior.
- Every subdirectory must also contain an AGENT.md.
