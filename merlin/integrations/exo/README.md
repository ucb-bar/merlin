# Integration: Exo

**Adapter only.** This directory does NOT vendor Exo. Point merlin at an external
checkout instead:

```bash
export MERLIN_EXO_REPO=/path/to/exo
```

## Purpose

Exo schedules and generated C kernels.

## Outputs

Normalized merlin artifacts (schema: `kernel_record`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
