---
title: Closed scalar observations from current fused source
kind: design
status: current
owner: core
last_verified: 2026-11-09
related: [lowering_pipeline, architecture, source_expression_interval]
code_refs:
  - src/merlin/llvmlower/source_observation_stage.py
  - src/merlin/llvmlower/source_expression_interval.py
  - src/merlin/llvmlower/pipeline.py
---

# Closed scalar observations from current fused source

The default-off `source_observation_closed_scalar_i8` feature retains the exact
generic tensor source after ordinary elementwise fusion and generalization,
before bufferization or pointwise scheduling. The native worker then continues
the original pipeline. The installed compiler parses and verifies that source
and applies the existing typed closed-scalar-observer theorem. Earlier model
preparation may see disconnected scalar expressions; this boundary sees their
actual fused producer and complete integer consumer together.

Callers select the feature and supply the existing explicit
`IntervalEffectContract` as `source_observation_effects`. The ordinary text/file
lowering APIs and bare-metal and Zephyr model builders forward the contract.
All five permissions remain required: RNE, gradual underflow, nontrapping
arithmetic, unobserved flags and unobserved signed zero. An accuracy gate or a
target choice cannot infer those permissions. Custom pipelines must contain
exactly one discovery marker at the supported fused tensor boundary; ambiguous,
missing and late stages refuse.

`source_observation.mlir` preserves the current source. The native worker reports
its digest, the lowering recipe binds its exact bytes, and
`source_observation_report.json` records typed observer identities, explicit
effects and refusals. `load_source_observations` rechecks source identity and
effects and rederives the proofs before returning them. Operation ordinals
identify bindings in this source; neither ordinals nor model names choose a
transformation. A new invocation invalidates the prior checkpoint and report,
including when its feature or effect selection refuses.

Discovery does not rewrite arithmetic, grant storage ownership, choose device
instructions or prove profitability. Helper reification and any consumer
transformation need their own numerical, effect and ownership contracts.
Independent shapes and tails, unchanged complete output gates, final object
identity and complete hardware costs remain necessary for promotion. With no
feature selected, the normal pipeline and generated worker remain unchanged.

## Reifying the proved scalar helpers

`reify_closed_scalar_observer_helpers` accepts a revalidated current typed
`ClosedScalarObserver` and explicit effects and creates two fresh functions:
the original `f32 -> f32` expression and the complete `f32 -> i8` quantizer.
It clones original operation order, precision, attributes and literal bits.
The two finishing rounded multiplies remain at the consumer; the quantizer
accepts their original scaled result. The source operations and their uses
remain unchanged. Enclosing provenance is retained for traceability.

Stale source/use/context proofs, missing dependencies, invalid ABI names and
conflicting trace records refuse. Helper lowering and linkage use ordinary
compiler interfaces. Reification supplies neither a placement policy nor
consumer coordinates, ownership, floating-mode guards or performance routing.

`source_observation_stage.validate_pipeline` owns the supported native seam.
`source_expression_interval` owns the typed observer theorem and revalidation.
The retained checkpoint is an analysis input, not an automatically selected OOT
region or a numerical relaxation.
