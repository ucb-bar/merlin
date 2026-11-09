---
title: Finite point observations at the BF16 quantization frontier
kind: design
status: current
owner: compiler
last_verified: 2026-11-09
related: []
code_refs:
  - src/merlin/llvmlower/frontier_point_cells.py
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/runtime/native/c/bf16_quant_frontier.h
---

# Finite point observations at the BF16 quantization frontier

`emit_source_attention_frontier(..., frontier_point_cells=contract)` is an
explicit, target independent host specialization. Ordinary emitter and runtime
header bytes remain unchanged. It introduces no approximation, workload match,
accelerator instruction or automatic selection policy.

## Proof and scope

The original row implementation first checks every endpoint and candidate is
finite BF16, ordered and enclosed. It computes the original minimum/maximum
possible row peaks, BF16 division, source epsilon maximum and BF16 reciprocal.
Unsupported scales or rounding modes refuse before writing output metadata.
The specialization retains this complete preparation and refusal order.

When both endpoint scales are equal, each endpoint uses that same quantizer.
For a finite point (`low == high`), the two final clamped integer observations
are identical. Signed zeros also compare equal; their source multiplication and
nearest-even result can retain different zero signs, but the source addition of
positive zero and final integer conversion make the observations identical.
This argument includes clamp ranges that exclude zero. Subnormal values, source
epsilon, reciprocal refusal and saturation use the unchanged source scale DAG.

Only the comparison between the two endpoint quantizations is replaced by false.
Nonpoint endpoints still execute both checks. An unstable scale still marks
every possible extremum using the original expression, including exact points.
Status, pending bytes, count and unique scale retain their original values.
Candidate and input arrays are never modified.

## Explicit obligations

`FrontierPointCellsContract` requires stable nearest-even execution, nontrapping
arithmetic, unobserved exception flags, pure returned integer quantizer values,
no quantizer library effects, immutable endpoints and disjoint private outputs.
These are separate source/component obligations. A typed pure MLIR
`math.roundeven` consumer can establish value operation semantics; an arbitrary
interposed `nearbyintf` C function cannot. A source observation witness must be
validated against the actual compilation input. This specialization grants no
new errno or library interposition permission to other operations.

An approximate product policy does not supply this observation proof. Original
source fallback, complete consumer closure and independent whole output gates
remain necessary when composing the helper with an approximate attention
provider. A numerical match at sampled inputs is not an effect witness.

The helper copies the authoritative owned runtime row implementation and checks
its exact observation seam. Template drift refuses emission. The default row
and coordinator remain unchanged. A caller must explicitly price the branch and
the remaining certificate work before selecting the specialization.

## Verification

The native test compares all 65,280 finite BF16 bit patterns under eight source
scale/clamp plans against the original implementation, with undefined behavior
sanitization. It includes signed zero pairs, BF16 subnormal epsilon, inverse
overflow refusal, saturation, clamp ranges excluding zero, stable mixed
intervals, unstable scales, every supported rounding mode and output guards.
Test-only rounding counters prove two calls are removed per admitted point;
they do not provide accelerator or whole model cycle estimates.

Whole model and target capsule receipts belong to their qualification owner.
No performance or production admission follows from these unit tests alone.

`frontier_point_cells.c_header` checks and specializes the owned row helper.
The regular frontier emitter controls selection and composition; the shared
runtime header remains the source of the scale/quantizer arithmetic. OOT
providers still supply the actual product and physical ABI.
