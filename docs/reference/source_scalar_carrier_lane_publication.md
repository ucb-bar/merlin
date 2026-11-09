---
title: Scalar observations in tensor lanes
kind: reference
status: current
owner: ir
last_verified: 2026-11-09
related: [lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/closed_tensor_insert.py
  - src/merlin/llvmlower/source_expression_interval.py
  - src/merlin/llvmlower/source_scalar_carrier_binding.py
  - src/merlin/llvmlower/source_stage_transport.py
---

# Current scalar observations in tensor lanes

This opt-in analysis composes the existing scalar carrier rewrite with
immutable `tensor.insert` publications. It leaves the selected pipeline stage
immediately before bufferization, after ordinary fusion and generalization.
Packet scheduling can remain selected. No earlier observation proof is reused.

## Admission

- Analyze the current complete source expression and original two rounded
  finishing multiplies, literal factor and saturated ties-even i8 observation.
- Require one use of the integer leaf at operand zero of a current insertion.
- Require a static positive i8 tensor, one destination carrier and one original
  tensor return. Every intermediate tensor carrier has one use in this chain.
- Accept static index constants and one bounded static SCF induction value plus
  constants. Prove all destination indices in bounds and every lane pair
  disjoint. Unknown bounds, coordinates and control effects decline.
- Preserve original input extractions, tensor insertions, SCF operations,
  destination operands, indices, resource handles and enclosing source context.
  Erase only the already authenticated private pure arithmetic.
- Complete every member's typed checks before insertion or use mutation. Bind
  the one-shot packet to the exact current native source digest; preserve all
  existing provenance joins and locations on the new scalar call.
- Keep the explicit numerical budget, original source fallback and actual
  provider-owned incoming-RNE predicate at every scalar point. This topic grants
  no permission to hoist that predicate across points or rounding changes.

Static `tensor.empty` construction stays in place and is recognized as unable
to observe the floating environment. Tensor-only bounded SCF control is admitted
recursively; unknown calls or effects remain refused. Source blocks with other
effectful work may therefore decline even when a tensor insertion is present.

## Qualification and cost scope

Independent lane widths two/four, nested shapes, tails, ownership/index/effect
refusals and native four-rounding-mode cases cover the analysis and normal
upstream insertion seam. Production applicability follows current typed IR;
no workload name or binding ordinal selects the transformation.

Selected workloads still require their original whole-output gate, a linked
runtime predicate, physical placement and complete producer/consumer timing.
Earlier coefficient/helper/provider results do not transfer to newly derived
code without revalidation. Retaining packet scheduling can generate additional
helper members; code size, per-point predicates and fallback service need pricing.

`closed_tensor_insert.prove_closed_tensor_insert` owns lane/publication
legality. `source_scalar_carrier_binding` retains the explicit host numeric
admission and incoming-RNE capability; `source_stage_transport` applies the
invocation-local native edit. These owners preserve the distinction between a
proved source expression, a runtime guard and an emitted implementation.
