# AGENT.md — merlin/python/merlin/targetgen/evidence

## Purpose

Deterministically discover source files and detect concepts (with citations) for a target.
Produces `evidence_report.md` and `evidence_index.yaml`.

## What belongs here

- File discovery + short summaries (filename / first-lines based).
- Per-target concept keyword tables and the detection that cites supporting files.

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

- Extend `CONCEPT_KEYWORDS` per target rather than adding bespoke parsing.
