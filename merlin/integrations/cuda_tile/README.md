# Integration: CUDA Tile

**Adapter only.** This directory does NOT vendor CUDA Tile. Point merlin at an external
checkout instead:

```bash
export MERLIN_CUDA_TILE_REPO=/path/to/cuda_tile
```

## Purpose

CUDA Tile programming-model integration (future).

## Outputs

Normalized merlin artifacts (schema: `kernel_record`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
