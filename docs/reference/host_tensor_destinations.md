---
title: Automatic host tensor destination reuse
kind: reference
status: current
owner: core
last_verified: 2026-10-18
related: [shared_compiler_passes, optional_passes, lowering_pipeline, whole_model_op_profile]
code_refs:
  - src/merlin/llvmlower/automatic_destinations.py
  - src/merlin/llvmlower/insert_slice_destination.py
  - src/merlin/llvmlower/destination_check.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/optional_passes.py
---

# Automatic host tensor destination reuse

Ordinary serial host lowering can write a pointwise tensor producer directly
into a fresh padded destination. It initializes the exterior only and retains
the complete original scalar body in the interior. A sole-use intermediate
therefore no longer needs a separate logical materialization and insertion.
The original padding tensor remains separate when another user observes it.

For example, inserting a `5x7` pointwise result at offsets `(1,2)` into a `7x11`
constant tensor can become exterior fills followed by direct interior writes.
The rule derives those regions from the current types and slice geometry; it
contains no model, target, captured-region or benchmark selection.

## Inputs and ordinary use

The default-on optional pass is `automatic-host-destinations`. It uses the
computation's typed tensor semantics and the selected compiler's index width
and data layout. It requires no additional accelerator dimensions or cycle
estimates. RTL extraction alone grants neither numerical equivalence nor
physical buffer ownership. Device-local banks, DMA and completion are outside
this host transformation.

Disable the pass through the existing compiler pass selector:

```sh
merlin-compile --no-pass automatic-host-destinations ...
# Or for an existing ordinary build invocation:
export MERLIN_PASSES=-automatic-host-destinations
```

`MERLIN_AUTOMATIC_HOST_DESTINATIONS=0` also deselects the default. An explicit
optional-pass selection takes precedence over that environment default.
The source helper `prepare_automatic_destinations` accepts explicit
`max_added_operations` and `max_metadata_items` budgets, defaulting to 4096 and
65536. These bound generated operations and rank-sized region metadata; they
are not hardware facts or a performance model. Addressed tensor bytes must fit
the selected signed index range. Unknown byte storage retains the original route.

## Checks and pass ordering

The pass runs after automatic source row scheduling and before ordinary host
preprocessing and bufferization. Its existing `PipelineSession` declares source
legality, invalidates prior IR analyses on rewriting, then requires fresh
emission for destination conformance. Context records bind source numerical
semantics, compiler layout/index selection, budgets and direct helper sources.
Source changes invalidate the ordinary full-model build cache.

The conformance reader reconstructs actual emitted view/fill/insert operations.
It verifies bounds, pairwise disjointness and total written volume without
enumerating elements or calling the emitter's region generator. It checks the
complete scalar body, inputs, original interfaces, unchanged effects and all
outer dataflow. Missing or overlapping writes, changed fill words (including
signed zero), arithmetic mutations and escaping temporary tensors refuse.
Reports describe those observations; a selected plan is not their proof.

When a rewrite succeeds, lowering selects the existing upstream destination
forwarding and exact self-copy cleanup features. Upstream tensor alias and
ownership analysis decide physical backing. Source-level conformance does not
establish a final machine allocation or traffic count. Inspect the actual
lowered program and execute complete outputs to qualify a deployment.

`automatic-host-destinations.json` and
`LowerResult.stats["automatic_host_destinations"]` record source/output and
helper identities, context, checks, realized regions and fallback diagnostics.
IR audit also records the `host-destination-reused` stage after an actual rewrite.

## Fallback and scope

The first domain is a static, unencoded, unit-stride rectangular insertion from
a sole-use, full-writing, all-parallel pointwise producer into a splat. Its
input maps must be projected permutations or broadcasts with statically
matching extents; dynamic and encoded inputs provide no full-write proof.
Full overwrites without an exterior retain their existing canonical route.
Output initialization reads, reductions, source indices, captured scalar values,
unknown attributes/effects and intervening calls retain the source. Only
constants and tensor storage constructors may lie between producer and insert.
Nested or multi-block control, encoded layouts, dynamic regions, insufficient
index/growth budgets and unsupported lowering families retain their legal route.
Automatic admission currently shares the 64-bit serial source domain with row
scheduling; neutral index-bound tests do not qualify other native ABIs.
The stock CLI transport retains its original path because it lacks the selected
runner cleanup. Explicit unsupported feature requests retain their existing
diagnostics; automatic selection supplies no alternate compiler driver.

No scalar operation, rounding, saturation, conversion or reduction order is
changed. There is no cross-invocation cache or hoisting of weights or carried
state. Caller output ownership and upstream physical alias rules remain required.
Native checks use private outputs and exercise aliased inputs, live padding,
carried state, boundary guards and several element widths and rectangular ranks.
These cases establish no new accelerator support.

Operation-profile markers are execution barriers. An instrumented build can
retain the original insertion even when its uninstrumented control rewrites it.
Compare actual pass reports and paired complete executions before using a
profile to prioritize later work. Runtime copy counters cover the wrapped copy
routine only; inlined transfers and accelerator traffic require separate evidence.
