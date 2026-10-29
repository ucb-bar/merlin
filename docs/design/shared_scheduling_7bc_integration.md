---
title: Shared scheduling source-choice integration
kind: design
status: current
owner: core
last_verified: 2026-10-19
related: [../guides/shared_pass_phase2, ../guides/shared_source_lowering, ../guides/shared_offload_selection]
code_refs:
  - src/merlin/llvmlower/source_choices.py
  - src/merlin/llvmlower/source_model.py
  - src/merlin/llvmlower/shared_source_routing.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/source_choices_workflow.py
---

# Shared scheduling source-choice integration

The selected upstream revision is
`7bc073ee03d93352144b0ce4e6ee865cc3fd8272`. Its latest five commits are
not a standalone change: the source-choice owner depends on earlier ordinary
source discovery, provider/ABI linking, complete-source placement, layout,
packing, readout and custody interfaces since `ade3489f`. This integration
retains that compiler dependency closure, together with its bounded tests.
It does not import the branch's separate Phase 1 compiler-call/selfcheck or
supported-substrate common-check changes. Existing formal authoring and
qualification gates keep their original behavior.

The merge preserves the existing candidate-owned pointwise and staged rewrite
interfaces, explicit nested `schedule_pass_order`, source capability registry
selection and additional-library comparison route. Both `read_dense_source`
and upstream `read_source_dense` remain available. The old numeric SSA-token
helper remains for existing weight-panel and quantization callers; the new
rank-N weight-layout implementation does not remove their import boundary.

## What executes

`SourceDenseChoices` freezes one original source and entry, one original
`SourceDenseLowering`, its implementation/effect/resource selections, and a
finite roster of legal controls. `SourceChoicesPolicy` exposes only
`compile-source-choice` through the existing broker. A request names the exact
source digest and a predeclared choice. It cannot replace source, callbacks,
effects, resources, output paths, numerical policy or a verifier.

The action calls ordinary `lower_model_file`, retains the actual prepared
source, compilation recipe, object/image and provider custody, and returns
bounded artifact identities. Host-native recurrent tests independently compare
all outputs, carried state, preserved inputs and actual provider effects under
two declared resource models. These tests establish the shared mechanism and
broker transport. They establish no Gemmini implementation, hardware timing,
measured speedup or formal campaign readiness.

The existing explicit source-routing interfaces also connect their original
emitter's optional `FreshTensorWriterContract.borrowed_effects` to the stronger
borrowed-buffer lifetime checker. Only the original typed route/build owner or
its exact reviewed-registry wrapper supplies this transport. The original
source and selected bytes, writer-effect snapshot, implementation files and
registry selection are reopened. Numeric equality, a ranked signature and a
successful inverse do not imply noescape, no-release or synchronous completion.
Missing effects grant nothing and can therefore refuse a loop containing an
unknown fresh-buffer call. Neutral fixtures select separately reviewed effects
for their exact C wrappers; no corresponding target permission is inferred.

## Authoring and release boundary

The source-choice workflow is opt-in original-host orchestration. The normal
Phase 2 authoring launcher still rejects it as a replacement workflow. Its
fixed provider does not become the retained generated compiler merely because
both compile the same source. Normal retained-revision compilation, independent
numeric/effect checks, timing admission and final decisions remain separate.

The smallest useful normal integration is an explicitly selected auxiliary
broker action alongside the existing corpus-feedback workflow. It requires
original frozen public-source membership and a generated-revision adapter that
binds the actual retained compiler, its provider/ABI, selected library and
emitted products. It must not substitute a neutral fixture or a protected
reference implementation. The existing formal gate and selected diagnostic
outcome rules remain in force; no allowlist flip supplies these prerequisites.

Finite source choices complement candidate source edits. Existing rewrite
callbacks and pass-order interfaces continue to permit new implementations and
shape/resource heuristics within the original frozen edit authority. The new
choice request is not an automatic discovery or arbitrary-pass service.

Updated exported modules require a fresh explicit shared-library release for
both treatments. No existing snapshot, including the frozen 66-leaf SDK, is
overwritten or silently upgraded. The original Phase 1 package remains
immutable. Selecting an additional reviewed library does not by itself require
new Phase 1 authoring; its host-owned access/transport and independent gates
must still be selected consistently before the comparison.

## Verification scope

Focused tests cover finite requests, private-path and callback custody, changed
source/resources, rebound compiled products, exact provider linking, current
selected contracts, candidate rewrites and pass ordering. Source-choice broker
models use complete independent recurrent references and two neutral native
resource geometries. Public saved-model checks additionally exercise ordinary
functional Spike readback, with explicit rejection of a foreign selected target
contract. These functional counts are not accelerator latency.

Earlier test failures are retained as evidence: an unscoped `os.open` test
patch affected pytest teardown; an ambient foreign target selection correctly
refused neutral execution; and the stronger lifetime checker exposed missing
explicit ABI-effect transport in the older source route. The fixes isolate the
test patch/selection and transport the original effects; they do not weaken
backend, numerical, lifetime or qualification checks.
