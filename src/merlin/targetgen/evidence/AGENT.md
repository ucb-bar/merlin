# AGENT.md — merlin/python/merlin/targetgen/evidence

## Purpose

Deterministically discover source files and detect concepts (with citations) for a target.
Produces `evidence_report.md` and `evidence_index.yaml`.

## What belongs here

- File discovery + short summaries (filename / first-lines based).
- The concept-detection logic that cites supporting files. The per-target keyword *vocabulary* is
  DATA, not a core literal: it lives in `<target-dir>/evidence_concepts.yaml` (loaded via
  `rtl.facts.target_base`), so `report.py` carries no per-target names.

## What does not belong here

- Any claim to "understand" RTL. This layer records what was *found*, not a parsed model.
- Synthesis decisions (those belong in `synthesize/`).

## Interfaces

- Consumes a `SourceManifest` from `ingest/`.
- Emits an `Evidence` object that serializes to the `evidence_report` schema.

## Invariants

- Read-only, deterministic, conservative.

## Testing expectations

- Exercised by `test_targetgen_toy.py` (toy_npu evidence must list files + concepts).

## Notes for future agents

- To teach a target new concepts, add rows to its `evidence_concepts.yaml` rather than adding
  bespoke parsing or a per-target branch in `report.py`.
