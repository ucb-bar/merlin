# Current joint campaign capsule roster

The v14 release exposes **250 public capsules**: 173 core and77 scalability.
Eight hidden capsules remain outside author inputs. Admission does not establish
execution success. [Exact250 IDs and grouping](joint_capsule_roster_v14.json).

## Core173

| Group | Count | Coverage |
| --- | ---: | --- |
| ISA and geometry |21|Integer contractions, GEMV, tails, geometry, movement, reductions, accumulator boundary|
| Layers |125|66 SY,34 G,25 MF: layer/source-derived cases, convolution, dense, epilogues and numerical regimes|
| Batched slice |1|Ranked batched contraction|
| Model-category capsules |6|Micro model, causal decoder/prefill and decode-step, coverage MLP, multimodal policy, residual CNN; final-stage gated|
| Calibration candidates |20|Reduction depth, tile extent, layouts/reuse, capacity, queues and convolution controls|

The first153 are the original core categories; the20 calibration candidates were
added to that endpoint. The six model-category capsules are already in173.
Separate complete ResNet50/TinyLlama/SmolVLA program evaluation is not another
set to add to the capsule total.

## Scalability77

| Group | Count | Coverage |
| --- | ---: | --- |
| Remaining selected Phase2 cases |39|12 tile extents,8 queue cases,9 capacity/spill regimes,6 fusion/unfused/bias controls,4 CPU-island chains|
| Allowed archive-derived tests |16|GEMM, convolution and schedule/reuse shapes|
| Model-derived contractions |5|ResNet50, TinyLlama prefill/decode, SmolVLA prefix/flow|
| Explicit user shapes |2|512³ GEMM and batch4 56×56 convolution|
| Connected source regions |3|ResNet50, TinyLlama, SmolVLA|
| Host seam controls |2|ResNet bias/scale/ReLU and dequantization/scale|
| Wide TinyLlama contractions |10|Decode and prefill; FFN6096/5728, down projections and vocabulary32000|

All59 cases in the selected explicit Phase2 source roster are present:20 in
calibration plus39 in scalability. The larger historical deferred specification
ledger contains additional recipes; it is not the same roster and is not evidence
that all historical proposed performance tests are executable here.

## Engines and gates

Use affected-subset Spike checks for development correctness, then larger cases
and provisional/qualified model feedback as available. Large cases are not routed
to gSIM by default. The20 calibration candidates are a selection pool, not20
mandatory simulations; each physical witness needs an unresolved model question,
existing-evidence review and the original budget. Independent source groups matter
for fitting, not raw capsule count. Final physical validation remains separate.
