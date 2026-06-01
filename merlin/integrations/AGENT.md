# AGENT.md — merlin/integrations

## Purpose

Lightweight **adapters** to external projects (XNNPACK, Autocomp, Exo, Triton, xDSL, IREE, CUDA Tile, Hexagon-MLIR). Adapters parse, index, normalize, or call external tools and emit merlin schema artifacts. They are NOT the external projects themselves.

## What belongs here

- One subdirectory per external project, each with `README.md`, `AGENT.md`, `manifest.yaml`.
- Later: adapter modules (discover/parse/extract/normalize) that emit merlin schemas.

## What does not belong here

- **Vendored external repositories.** Never clone XNNPACK/Autocomp/etc. here.
- Hard build dependencies (those go in `third_party/`).

## Interfaces

External repos are passed by path or env var, e.g. `export MERLIN_XNNPACK_REPO=/path`. Adapters output normalized `kernel_record` / `abstraction_candidate` / `policy_rule` artifacts per `merlin/schemas/`. Note: this is distinct from `merlin/python/merlin/xdsl_dialects/` — the `integrations/xdsl/` adapter wraps xDSL tooling/import-export, while `xdsl_dialects/` holds merlin's own prototype dialects.

## Invariants

- Integrations are adapters only. Do not vendor external repos here.
- Adapter output must conform to a schema in `merlin/schemas/`.
- An adapter must degrade gracefully when its external repo is not configured.

## Testing expectations

Adapter tests should use small fixtures, not a full external checkout. Gate live-repo tests behind the `MERLIN_<NAME>_REPO` env var.

## Notes for future agents

Decision rule: if merlin can run without it, it is an integration; if merlin cannot build without it, it is `third_party/`.
