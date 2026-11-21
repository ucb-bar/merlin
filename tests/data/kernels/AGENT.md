# AGENT.md — merlin/python/tests/data/kernels

## Purpose

Trimmed kernel fixtures (one per source/ISA) used to pin ingest + feature-extraction behavior.

## What belongs here

- `*.c` / `*.py` snippets that exercise specific motif markers (packed RHS, accumulator,
  epilogue, vector-length, double buffering, weight-stationary).

## What does not belong here

- Full kernels or vendored repos.

## Invariants

- Each fixture is a faithful but minimal excerpt of a real kernel; keep markers intact so the
  extractors are tested against realistic syntax.
