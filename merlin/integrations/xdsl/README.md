# Integration: xDSL

**Adapter only.** This directory does NOT vendor xDSL. Point merlin at an external
checkout instead:

```bash
export MERLIN_XDSL_REPO=/path/to/xdsl
```

## Purpose

Adapter to xDSL tooling: import/export and CLI helpers. Distinct from merlin's own prototype dialects in merlin/python/merlin/xdsl_dialects/.

## Outputs

Normalized merlin artifacts (schema: `n/a`) written under `output/`. See
`merlin/schemas/` for the artifact formats.

## Status

Scaffold. Adapter modules (discover / parse / extract / normalize) are added by the
kernel-mining workstream. Keep this an adapter — never clone the external repo here.
