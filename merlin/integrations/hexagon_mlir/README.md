# Integration: Hexagon-MLIR

**Adapter only.** This directory does NOT vendor Hexagon-MLIR. Point merlin at an external
checkout instead:

```bash
export MERLIN_HEXAGON_MLIR_REPO=/path/to/hexagon_mlir
```

## Purpose

Hexagon-MLIR target integration edge (future).

## Outputs

Normalized merlin artifacts (schema: `n/a`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
