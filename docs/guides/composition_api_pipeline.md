---
title: Compose shared compiler transformations in an authoring loop
kind: guide
status: current
owner: compile
last_verified: 2026-10-26
code_refs:
  - src/merlin/compile/composition_catalog.py
  - src/merlin/compile/scheduling/api_catalog.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/prepare_author_library.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/full_compiler_workspace.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/author_skills/heavy-pass-recipes/scripts/composition_routes.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/composition_next.py
---

# Compose shared compiler transformations in an authoring loop

Offline construction and the automatic authoring loop use the same Python/xDSL
APIs. Select the sources before authoring, then connect applicable transformations
to the ordinary compiler commands. An import, catalog entry or selected source
file establishes availability; execution requires the transformed product to reach
the target emitter and the original independent checks.

## Select and discover sources

Use `python -m merlin_experiments.phase1.prepare_author_library` with
`--library-version 22`, an exact source revision, the previous reviewed library
selection and a fresh output directory. Additional generic leaves require explicit
`--reviewed-module` selections. The normal preparation retains existing access
policy and checks the source export and contract. Inventory V20 corresponds to
`merlin.common_compiler_substrate.v13` (shared-library schema
`merlin.shared_scheduling_library.v14`), V21 to `merlin.common_compiler_substrate.v14`
(schema `v15`) and V22 to `merlin.common_compiler_substrate.v15` (schema `v16`).
Select matching profile, SDK and additional library identities together. Existing
frozen selections keep their versions.

V22 keeps the V21 scheduling roster and catalog. Its SDK release stages no
author-inaccessible module (`merlin.common.access.author_inaccessible`: a package in
`AUTHOR_WITHHELD_PREFIXES` or an access-policy grader/oracle module), even where an
arm tool grant names one: the CCA spine stays operator-side behind its broker
clients, and the bundle records those grants as denials. It stages the generic
`merlin.compile.analysis` modules the remaining tools import instead. Inventories
V19 to V21 remain unchanged for frozen runs and sealed bundles.

Inspect `scheduling_capabilities(22)` from
`merlin.compile.scheduling.api_catalog`. For composition, pass the exact available
module roster to the separate catalog:

```python
from merlin.compile.composition_catalog import composition_capabilities
from merlin.compile.scheduling.library_sources import library_modules

small_sdk = composition_capabilities(selected_modules=library_modules(22))
# For an additional reviewed library, use its selected contract instead:
additional = composition_capabilities(
    selected_modules=tuple(selected_contract.public_modules),
)
```

`source_availability` reports missing exact modules. An omitted selection remains
unknown; package names grant no descendant modules. Module membership does not
establish symbol-version compatibility, transitive import access, target support
or original host authority. Some composition tools require an explicitly selected
additional library or full compiler source export.

For a fresh full compiler source snapshot, the ordinary operator preparation is:

```python
from merlin_experiments.phase1.full_compiler_workspace import prepare

selection = prepare(repo, fresh_output, review_id=review_identity)
selection.verify()
record = selection.record()
```

Persist that record and supply its root, manifest and digest through the ordinary
`--full-compiler-root`, `--full-compiler-manifest` and
`--full-compiler-manifest-sha256` selections. Preparation copies the allowed generic
compiler source closure and reports unavailable dependencies. It includes no
workload inputs, model parameters or reference implementations. Readable host
source does not authorize replacing an original host/checker owner.

In an author workspace, the heavy-pass-recipes skill's `composition_routes.py --case NAME`
applies this discovery to one public case: it reads the case's declared structure (and, for a model
or program, its interface IR) and lists the matching routes with their catalog rows, roster status,
one `module:symbol`, whether `submission/` imports it and a recipe or contract command, plus the
`source_composition.json` options described below. Self-check rows carry the same pointer as
`optimization_next.composition` and `optimization_next.source_composition_request`
(`merlin_experiments.phase1.feedback.composition_next`). These are pointers, not selections.

## Build a layered pipeline

| Layer | Reusable entry points | Product that must reach the next layer |
| --- | --- | --- |
| Source graph and views | `fusion_workflow`, `connected_regions`, `source_region`, `tensor_views`, checked `global_plan` emission | Selected original SSA, explicit inputs, all live outputs and actual transformed components |
| Grouped work | `shared_input_groups`, `shared_input_schedule`, `projection_views`, `multioutput_fusion` | Checked grouping and physical member views retaining original pointer, pitch and output identities |
| Preparation and storage | `temporary_storage`, `fresh_tensor_writer`, `private_workspace_pool`, `weight_prepack`, `compiler_observations.install_prepared_observed` | Bounded owned placements and actual exact preparation bytes with alias, lifetime and completion obligations |
| Spatial composition | `spatial_producer`, `rolling_spatial`, `rolling_window`, `spatial_readout_cache` | Actual counted packing/cache transfers and source-exact ordered readouts |
| Schedule and order | `pass_exploration`, `loop_schedule`, typed `schedule_ir`, `scf_commands` | The selected transformed Kernel/SCF, revalidated and consumed by target callbacks |
| Target emission | Selected public facts, adapters, readout and execution contracts | Final commands checked for complete work, physical accesses, state and completion |

Keep original quantization order, coefficients, rounding, saturation, activation
and output types across every layer. `prove_spatial_readout` and
`propose_exact_affine_pair` provide bounded source-exact proposals; they do not
make varying per-column coefficients scalar. Unsupported matches retain a valid
original route or report the missing prerequisite.

Caller-provided loop factors, cuts and pass orders express alternatives, without
requiring a particular optimization policy. Emit the actual transformed Kernel,
not a checking-only projection. Final wide transfers need all-band mapping and
hazard checks. Persistent scalar SCF state is explicitly opted in and must match
actual configuration callbacks and selected visibility/clobber laws. Loop-carried
event ownership is not supported by that scalar-state extension. Asynchronous
issue or removed fences require separate selected execution and completion laws.

## Preserve original host ownership

`ConnectedSpatialRoute` is an explicit original-caller selection for
`OwnedHostModelCompiler.compile(connected_spatial=...)`. It binds the original SSA
region, ordered quantizers, observers, private workspace and source-owned host
build to ordinary device object/link checks. A candidate flag or a readable route
module cannot activate it. `GraphOptimizationContract` likewise requires original
source and layout permissions for host graph changes.

`prepare_sibling_source_host` can outline sibling integer roots while preserving
original host epilogues and live outputs; `emit_ranked_source_entry` supplies its
explicit entry transport. An original caller can
select `SiblingSourceRoute` through
`OwnedHostModelCompiler.compile(sibling_source=...)`. This binds exact source and
separately qualified synchronous borrowed effects, emits one closed raw integer
core and keeps original host readouts. It is mutually exclusive with the spatial
route and opaque source/ABI rewrites. Independent many-to-one source membership
and actual-object audits preserve the original required-source denominator.
The original owner can select `SiblingSourceRoute(..., partial=True)` to outline
independently checked shared-input components, retaining unmatched accelerator
groups on their ordinary device path. The automatic preference below requests
this fallback when a single complete cluster is unavailable. Component membership
comes from actual shared SSA inputs and shapes; original host quantizers and
observers remain. Immutable weight prepacking and exact preparation sharing may
compose with this route. Dense input views apply only at proven compatible
boundaries, including source-owned borrowed wrappers. The checked bufferization
consumer retains the source shape, element type, readonly access and original
C-bridge stride/alias guards while allowing a dense nonzero element offset.
Noncontiguous inputs retain copies. Selection records distinguish applied layouts
from retained wrapper inputs.

The owned automatic caller accepts an optional candidate
`source_composition.json` with these required fields:

```json
{
  "schema": "merlin.source-composition-preferences.v1",
  "shared_input_siblings": true,
  "prepack_weights": true
}
```

The optional Boolean `share_exact_tensor_preparations` selects the existing
source-exact immutable view/constant and exact integer-to-float preparation pass.
It is also available as an explicit `OwnedHostModelCompiler.compile` argument.
Omitting it preserves the prior request identity and default lowering. Merged
provenance and all original numerical/target checks remain required.

The optional Boolean `dense_input_views` exposes only the original generated
logical shim’s readonly rank-two dense-stride/nonzero-offset ABI. The owner
reopens actual shim compilation and interfaces before binding it; noncontiguous
inputs retain layout-conversion copies, and guards on outputs and aliases remain intact.
This opt-in can compose with exact preparation sharing and ordinary immutable
weight prepacking on the supported group route. A callee argument gains the dense
layout only when every original call proves it; an unproved or strided call keeps
the original layout for that position across all calls.
It is separately available on `OwnedHostModelCompiler.compile`. Both options
retain the ordinary numerical and target checks; neither predicts a speedup.

The optional Boolean `independent_heads` requests exact source head refolding
through the existing ordinary batch-slice ABI. The original owner requires
complete independent operation admission for every composed head; unmatched
groups keep their ordinary accelerator route and original readouts remain.
With both this and `shared_input_siblings`, checked head refolding runs first,
then checked sibling components use disjoint remaining original products.
If no legal sibling component remains, the head route records that request as
deferred. The typed API is `HeadBatchSourceRoute(..., shared_input_siblings=True)`;
spatial composition still needs its separate owner. Verified receipts distinguish batch ABI
calls from unchanged matrix invocations and actual target commands. See
[`independent_heads.md`](../../examples/compiler/independent_heads.md).

These are optimization requests. The original owner rederives source membership
and retains the baseline when the complete sibling form is inapplicable. The
candidate cannot select source IDs, change numerical boundaries or grant effects.
Independent per-member device work, host-call, object and execution checks remain
mandatory. Prepacking uses the existing bit-preserving immutable-weight pass for
both lowering and runtime; the original source identity remains authoritative.
An old frozen framework does not acquire this adapter by importing a new library.
New selections must also include the corrected canonical `interface_emit` owner
and `merlin.compile.scheduling.projection_provenance`, or explicitly attributed
ordinary authored adaptations. Immutable preparation and measured profitability
remain separate choices.

In the automatic loop, retain source and selected-library identities, transformed
IR, emitted products and ordinary qualification results. Keep correctness,
structural admission and measured performance separate. Discovery, reduced command
counts or storage estimates alone establish no numerical qualification or speedup.

## Diagnose complete model execution

The original model owner and the automatic owned-model caller expose the same
explicit preparation choices: `prepack_weights`,
`share_exact_tensor_preparations` and `dense_input_views`. Check the completed
build's selected inputs and transformed artifacts: availability alone does not
mean a choice was applied. Immutable weight preparation requires original
storage ownership; read-only views require the actual callee ABI. These choices
preserve the captured numerical recipe and do not grant arbitrary host changes.

`OwnedHostModelCompiler.compile(op_profile=True)` selects the existing typed
operation profiler. The automatic caller accepts the optional Boolean
`op_profile` in `source_composition.json`. It retains the actual operation table
and summarizes completed observations by source category. Missing observations
are diagnostic unavailability, independently of numerical results; changed table
identities refuse. Original complete-output checks still run.

Use a separate uninstrumented control. Markers can change fusion, allocation and
scheduling. The reported raw counter ticks are not automatically elapsed cycles:
functional Spike counters do not establish target performance. Bind the selected
measurement engine before interpreting timing, and retain complete-call timing
alongside the diagnostic breakdown. Host packing, copies and synchronization
inside an operation interval are not independently resolved by its category.

Build larger regions from shared operation, layout and effect contracts. Keep
independent batch axes and quantization-plane products explicit until scheduling;
combining their dispatch must preserve all products and ordered readouts. Choose
legal schedule alternatives from shape and resource features, then compare whole
regions. Model names, source IDs and private reference scores are not schedule
selection features.

### Emit checked dense completion epochs

`schedule_dense_panel_pipeline` consumes the ordinary typed dense schedule plan.
Its result keeps actual operand effects, private row assignments and explicit
full-completion epochs. The caller supplies independently established issue,
configuration snapshot, queue backpressure and completion premises. The separate
`batch_dense_panel_readouts` pass may group pure readouts when their numerical
configuration and selected snapshot contract permit it; it crosses no compute
or movement effect. Neither service grants target execution semantics.

After checking the original source and final actual command projection, use the
result's distinct effect and completion callbacks:

```python
receipt = checked_schedule.emit_index_program(
    index_control,
    lower_effect=emit_actual_target_instruction,
    complete_epoch=emit_real_full_device_completion,
)
```

The completion callback receives every explicit drain in the checked program.
It must emit the original full device completion mechanism. An issue event or
an estimated delay cannot replace it. The effect callback receives the actual
lowered scalar operands and must preserve the selected decoding and numerical
configuration. Source, plan, admission and shared-library identities are checked
again after emission. This interface is available to ordinary schedule proposals;
it contains no shape choice, target instruction recipe or performance claim.
