---
title: Bounded integer observation packet scheduling
kind: reference
status: current
owner: ir
last_verified: 2026-11-09
related: [lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/bounded_rne_maps.py
---

# Bounded integer observation packet scheduling

The explicit `schedule_bounded_rne_maps` API analyzes the complete current
ties-even integer observation and static parallel tensor maps. It derives input
coordinate permutations through supported transposes, preserves source
arithmetic in private scalar helpers, and stripmines a contiguous input axis.
Unknown operations, arithmetic attributes, maps, effects and bounds decline.

## Adjacent output coordinates

`output_minor_batch` is an optional bounded emission choice, with default one.
For distinct input-contiguous and output-minor axes, it groups independent
output coordinates within each input packet iteration. The selected width is
bounded by the actual output-minor extent. Coincident axes and unit minor
extents preserve the original schedule and report an explicit fallback.

Every coordinate invokes the original scalar packet body. Full packets and
exact static tails on both dimensions remain separate. Original tensor
destination values and live aliases survive for upstream bufferization to
resolve; the transform adds no physical no-alias or alignment assumption.
It changes neither numerical permission nor the observer's finishing operations.

The API validates a literal integer width from one through eight before mutation.
This bounds emitted code, rather than describing a target register or resource.
Default one adds no report fields and preserves original scheduling. An explicit
request records selected width, axes, tails, fallback and unknown profitability.

## Composition and complete costs

This scheduling choice belongs to the shared compiler and can be selected
independently of the accelerator. Target instruction legalization and resource
facts are separate provider obligations. Neither a workload identity nor a
captured output selects a width.

Legal coordinate reuse does not imply cheaper execution. Compare complete
quantization, coordinate arithmetic, initialization, allocation, padding and
callbacks, binding actual runtime support objects. Include independent shapes,
minor extents, both tails and live destination aliases. Retired instructions,
native elapsed time and hardware cycle measurements are distinct quantities.
Preserve default behavior until a complete cost policy selects a candidate.
