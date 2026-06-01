# Kernel abstraction mining (Workstream 2)

Pipeline:

```
XNNPACK RVV / Autocomp (Gemmini, Radiance) / Triton / Exo
  -> kernel_record
  -> abstraction_candidate
  -> policy_rule
```

Extract optimization lessons from existing kernels and turn them into compiler-consumable
policies.

## Modules

`merlin/python/merlin/kernels/{ingest,features,emit}/`. Sources are reached through adapters in
`merlin/integrations/` (external repos passed by env var, never vendored).

## Features to extract

packing, packed RHS, vector-length strategy, tail strategy, epilogue fusion, accumulator usage,
tiling/blocking, target-specific configuration.

## Tools

`tools/merlin-kernel-index/`, `tools/merlin-kernel-extract/` (write to `output/kernels/`).

## Must not

Build a large classifier before a small validated corpus; claim automatic abstraction discovery;
hard-code XNNPACK-only assumptions into the generic kernel schema.
