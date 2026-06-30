# AGENT.md — artifacts/recaptures

## Purpose

Model captures and golden references consumed by every target backend and by DSE:
`<model>_<dtype>_<variant>/` (each with `model.mlir` + weights/io) plus golden reference outputs
(`*_reference.npz`, `*_ref.npz`, `*_result.txt`). This is the real store (the legacy `output/`
tree was drained here and deleted). Resolve via `merlin.common.artifacts.recaptures_dir()`.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- **PURGEABLE**: model captures regenerate via the m2m exporter; large `*_spike`/`*_cgen` dirs are
  regenerable scratch. The golden `*_ref.npz` references are the bit worth keeping.
- Read via `recaptures_dir()`, never a hard-coded `output/...` or `artifacts/recaptures/...` path.
- Axis: **model + dtype + variant**.
