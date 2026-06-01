# Integration: IREE

**Adapter only.** This directory does NOT vendor IREE. Point merlin at an external
checkout instead:

```bash
export MERLIN_IREE_REPO=/path/to/iree
```

## Purpose

IREE compiler/runtime integration edge (future).

## Outputs

Normalized merlin artifacts (schema: `n/a`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
