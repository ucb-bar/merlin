# Integration: Triton

**Adapter only.** This directory does NOT vendor Triton. Point merlin at an external
checkout instead:

```bash
export MERLIN_TRITON_REPO=/path/to/triton
```

## Purpose

Triton kernels (larger corpus for feature extraction).

## Outputs

Normalized merlin artifacts (schema: `kernel_record`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
