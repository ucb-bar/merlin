# AGENT.md — merlin/benchmarks/rvv_paper

## Purpose

Curated, versioned inputs for the frozen-compiler K1 paper study. This directory declares the
holdout boundary, model/session semantics, comparison backends, precision policy, quality gates,
and freeze requirements. It contains no generated measurements.

## Invariants

- Paper models never enter the development corpus or search objective.
- Freeze hashes and capture hashes must be resolved before a live paper run.
- Same-buffer repetition is diagnostic only; reported throughput uses the declared stateful session.
- The primary table times every stage in a continuous session. Decode-only and denoise-only artifacts
  are diagnostic stage subsets and cannot be compared as end-to-end measurements.
- Results belong under `out/runs/` and `out/artifacts/`, never here.
