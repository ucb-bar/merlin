---
title: "Exact integer reconstruction in the source attention frontier"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/llvmlower/radix_integer_reconstruct.py
  - src/merlin/llvmlower/radix_product_groups.py
---

# Exact integer reconstruction in the source attention frontier

`emit_source_attention_frontier(..., integer_reconstruction=True)` reuses the
canonical signed-radix reconstruction proof and helper. The default is false.
Each signed-i32 degree is fully written by the existing source-bound product
callback. Its legal signed-seven-bit digit range, reduction extent and weighted
prefix bound prove that all terms and prefixes fit i64 and are exactly
representable as binary64. The original positive-zero, RNE reconstruction is
therefore reproduced by integer accumulation and one conversion. External
scales, source-ordered floating arithmetic, certificates and fallbacks do not
change.

With the default `fuse_integer_reconstruction=False`, a separate private
`int64_t` array is part of the compiled workspace. The first
complete degree initializes every active element before later accumulation.
The array is disjoint from the readout and double center; it is not a retyped
view of either. Its complete allocation and traffic belong in measured costs.
The actual compiled workspace query grows by eight bytes per maximum output
cell. Normal callers must update their workspace contract; a previous smaller
pool refuses safely and does not qualify the selected route.

The separate explicit `fuse_integer_reconstruction=True` option retains all
completed signed-i32 degree readouts until one local integer sum and conversion.
It requires integer reconstruction and replaces the intermediate i64 array with
the additional readout planes. Its workspace delta is four bytes per additional
degree per output cell, rather than the unfused eight-byte scratch delta. Always
use the selected compiled workspace query; the two layouts are distinct contracts.

This option does not certify an arbitrary product callback. Existing complete
source/producer/range and private-storage contracts remain mandatory. No
performance or routing decision follows from enabling it. Independent masked,
strided, dirty-workspace and refusal executions preserve source consumer
observations. Actual complete-group and whole-model qualification remains a
separate release gate.
