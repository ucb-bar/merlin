# Integration: XNNPACK

**Adapter only.** This directory does NOT vendor XNNPACK. Point merlin at an external
checkout instead:

```bash
export MERLIN_XNNPACK_REPO=/path/to/xnnpack
```

## Purpose

RVV / RISC-V quantized GEMM and conv microkernels.

## Outputs

Normalized merlin artifacts (schema: `kernel_record`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
