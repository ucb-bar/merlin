# Integration: Autocomp

**Adapter only.** This directory does NOT vendor Autocomp. Point merlin at an external
checkout instead:

```bash
export MERLIN_AUTOCOMP_REPO=/path/to/autocomp
```

## Purpose

Autocomp-generated kernels for Gemmini and Radiance.

## Outputs

Normalized merlin artifacts (schema: `kernel_record`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
