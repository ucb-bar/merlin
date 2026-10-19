---
title: Shared passes in ordinary compiler authoring
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-19
related: [../reference/shared_compiler_passes, phase2_corpus_search]
code_refs: [src/merlin/targetgen/tool_profile.py, src/merlin/targetgen/tool_profile_shared.py, src/merlin/targetgen/common_transforms.py, src/merlin/targetgen/compiler_library.py, packages/merlin-experiments/src/merlin_experiments/phase2/compiler_libraries.py, packages/merlin-experiments/src/merlin_experiments/phase2/compiler_library_state.py, packages/merlin-experiments/src/merlin_experiments/phase2/compiler_library_guidance.py, packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_stage.py]
---

# Shared passes in ordinary compiler authoring

The **V6 common compiler substrate** makes the reviewed shared scheduling
library available to ordinary compiler candidates. The implemented ordinary
inheritance route freezes one release in Phase 1 and preserves its grants and
source identities in Phase 2. Versions 1–5 retain their original public APIs
and proof policies.

This is an access-comparison contract, not a technical requirement to rerun
Phase 1 authoring before optimizing an older compiler. A separate Phase 2
experiment may select additional reviewed compiler libraries explicitly. It
must preserve the Phase 1 compiler and evidence as its origin, copy the compiler
into Phase 2, record the new library release separately, and revalidate baseline
and candidate under the selected Phase 2 environment. It cannot attribute the
new grants to the original Phase 1 run or claim the original same-grant ablation.
All compared Phase 2 treatments must receive the same selected shared algorithms.
The current source adds bit-preserving dense-float IO to the reviewed SDK.
Existing V6 profiles retain their original source roster and hashes; validation
also recognizes that preceding roster without adding the new helper to its
grants. Selecting this updated release is explicit and never rewrites Phase 1.

The explicit `CompilerLibraryContract` and component-view APIs review and
isolate such a library. A host-owned `CompilerLibrarySelection(contract, root)`
now connects additional imports to the normal package sandbox and ordinary
retained feedback API. Pass it as `compiler_library_selection` to
`prepare_ordinary_search_policy`. The fixed search binding and artifact context
include the selection; broker and queued member threads retain it for both arms.
The sandbox mounts exact reviewed leaves, reopens their hashes, reserves its
library namespace against overlays, and preserves original public leaves outside
the selected replacement set. Candidate manifests cannot select a release.

The selected-release prompt and `STAGE_CONTEXT.json` expose a source-derived
public API inventory: reviewed contract identity, exact guest source paths and
digests, declaration lines, public class methods and bounded parameter/docstring
previews. Omitted declarations are counted. Imported names are descriptive;
dynamic exports and value aliases remain unresolved. Reading the inventory
executes no selected library code and grants no target support or timing credit.
The shared algorithms are reviewed authored code; software effects and RTL
facts constrain legal policy inputs and target-adapter applicability.

New compiler preparation may explicitly copy and select
`targetgen.selected_library_imports.activate_selected_sources_v2` before imports.
When the recorded host root is absent inside the sandbox, V2 opens only the
existing `/perf-compiler-library/<contract-sha256>/python` projection and rechecks
every selected source before and after loading. Missing original dependency
roots add no imports; the guest tree already contains the exact admitted public
dependencies. Existing invalid roots and preloaded foreign leaves still refuse.
This is a startup path connection, with no host-directory grant, SDK algorithm
change or qualification authority. V1 and previously prepared packages retain
their original startup; both treatments must select V2 before source freeze.

When normal queued search freezes its edit authority, the host projects the
actual validated contract into `STAGE_CONTEXT.json` as
`frozen_compiler_edit_authority`, before policy or worker creation. It retains
initial-source and contract digests, exact existing paths/symbols, scoped helper
extensions and explicitly approved v2 new Python components. Private seed paths
are excluded. Editing the projection, automatic inventory or candidate manifest
cannot change the original authority. Candidate-owned policies and transformation
bodies may change within that authority; the selected library remains read-only.
The prompt connects hypotheses to owned entrypoints, expected emitted-program
changes, affected public shapes, retained revisions, independent correctness
and paired measurements or qualified predictions. UNKNOWN feedback stays visible.

The host `authoring.run_stage` API accepts `compiler_library_selection`
before any authoring preflight. The outer author receives exact read-only source
leaves; the broker's compiler execution receives the selected import environment.
Outer Codex Python does not inherit the broker's `PYTHONPATH`. Original public
leaves are reconstructed from the unchanged frozen Phase 1 profile. Broker
threads retain the bound owner, and policy verification reopens the selected
sources and rejects mount overlays or an ambient replacement library.

The final functional guard uses the same selection for both arms and runs the
original complete numerical/certification grade on the baseline and candidate.
A historical unchanged-dependency result cannot qualify a new library release.
Without an original profile executor, the guard uses the existing narrowed
package sandbox with exact library leaves. An external hard resource boundary
retains all its original checks; selecting a library alone supplies no hard
aggregate resource guarantee.

The authoring and paired-measurement CLIs accept an explicit host selection:
`--compiler-library-root`, `--compiler-library-contract` and
`--compiler-library-contract-sha256` must be supplied together. The last value is
the canonical reviewed `merlin.compiler_library.v1` contract identity, not the
raw JSON file hash. Candidate manifests and saved records cannot authorize a
selection. Omitting the flags retains the original source grants and schemas.

An admitted additional-library stage emits candidate record v4 and a typed
`VerifiedCandidateHandoff` v2. After the original same-release functional guards,
the host copies only reviewed source leaves into an exact readonly snapshot;
private siblings are excluded. The record binds the original host selection,
snapshot contract, complete tree and binding digests. Final consumption requires
the independently supplied original host selection and reopens both source
trees. Paired plan v4 carries the same snapshot owner into both compilers and
each serial or threaded worker, with checks before and after execution. Source
drift, extra snapshot leaves, writable snapshots and dropped or foreign owners
refuse. Diagnostic opt-in records retain their previous non-consumable v3 meaning.

Serialized model-budget children use sealed original-parent transport v2 only
when an additional library is selected. It binds the reviewed source closure,
environment and original executor alongside the unchanged Phase 1 profile.
The fresh interpreter reconstructs and rechecks that exact selection; completion
and parent consumption recheck it again. Missing or stale library data and
attempted v1 upgrades refuse. Original v1 transport and receipts retain their
interpretation. The descriptor remains private to the trusted child grader.

These connections supply no target adapter, dependency installation, runtime or
timing qualification. Original Phase 1 profile equality and functional admission
remain unchanged. A new Phase 1 run is one available same-grant baseline, not a
prerequisite for this Phase 2 mode. Tests exercise policy construction, actual
local sandbox source access, thread transport, mocked final-grade routing,
fresh-interpreter restoration and paired worker ownership; they are not
accelerator qualification or an author campaign.

The library provides checked mechanisms for tiling/operand reuse, temporary
storage, capability-driven fusion and copy elimination, persistent configuration
cleanup, explicitly supported device spills, complete row reductions and final
program limits. Availability is independent of target support. A backend must
supply typed computation, current resources, exact instruction effects, realized
encoding/decoding and independently qualified execution semantics where required.
Opaque backend output and missing capabilities remain unavailable.

Library availability does not automatically enroll every pass in saved-model
compilation. A generic ordinary preparation owner now connects the supported
integer pointwise source family through explicit backend and qualifier selection,
as described below. Dense preparation also has the explicit checked source route
below; typed-row complete-model tests still supply manual test adapters.
Qualified real-device adapters and automatic discovery from an
original target capability selection remain unfinished. Importing the SDK alone
changes no schedule.

### Ordinary pointwise source routing

`llvmlower.shared_pointwise_routing.PointwiseSourceRouting` consumes an explicit
host-selected `PointwiseSourceBackend` and a separate original independent
execution qualifier. Its `build_arguments()` supplies the existing
`prepared_model_transform` and `host_provider_builder` keywords to ordinary
`spike_model.build`. The backend supplies instruction proposals, target ABI/glue,
provider compilation and original capability revalidation. There is no
target-name lookup or default backend, and no diagnostic target hook is enabled.
Missing backend or qualifier returns no hooks and reports the disabled selection.

The owner scans actual prepared typed source and calls `compile_source_pointwise`
for supported semantics, layouts and resources. It checks actual final encoded
programs and selected software/RTL inputs, then independently inverts all newly
introduced calls and fresh result allocations to the complete original module.
The inverse retains unrelated operations, interfaces and loop-carried SSA. The
original independent qualifier runs on the emitted implementation before the
selected model is published. The existing fresh-writer, provider object/ABI,
compilation pin and final linked-image checks remain mandatory. Source, selected
contracts, callback source and original verification ceilings are reopened;
policies can change supported scheduling choices within those frozen limits.

Unsupported islands retain their original source with exact refusal reasons.
Incorrect programs fail rather than becoming fallback. A completely empty route
may skip provider construction only through this exact typed owner after fresh
source/selection checks; it emits no synthetic provider or witness. Arbitrary
callbacks retain the original provider admission rules.

The small recurrent regression in `test_shared_pointwise_routing.py` uses this
normal owner instead of test-specific source matching. Both arms retain the same
documents, numerics and library; independent native and functional Spike checks
cover complete outputs/state, linked instruction bytes and packing costs. Neutral
primitive implementations remain private test support. This establishes the
production connection, not real-device support, absolute model accuracy,
calibrated timing, transferable policy or automatic enrollment of every pass.
The routing owner is host infrastructure, not a new public SDK/checker grant.
The original caller still owns module loading, sandbox execution and transitive
dependency qualification; direct source/code freshness checks do not supply
those authorities or a calibrated hardware model.

### Ordinary dense and shared source routing

`llvmlower.shared_source_routing.SharedSourceRouting` accepts an explicit
`DenseSourceBackend`, `PointwiseSourceBackend` or `SharedSourceBackend` plus the
original independent execution qualifier. It uses the same ordinary build hooks,
complete-module inverse and provider/link checks above. The older pointwise
entry remains available. This host routing owner is outside the public SDK
grant and creates no new candidate authority or target support.

For dense islands, `read_dense_source` checks the current signed rank-two
multiply/add recurrence, default tensor layout and exact zero initialization
before invoking target policy. `schedule_source_dense` repeats that admission
and checks the actual compact loop program against selected software/RTL,
assigned storage, tails, continuation and address bounds. Policies may select
blocking resources, loop order and smaller static/repair budgets; they cannot
replace the current operation, selected contracts, original runtime buffer
ownership, solver limit or source ceiling. The backend emits real ABI glue;
the original qualifier must independently check that emitted implementation.
The provider's exact source, compiled objects, typed ABI and final linked image
remain separately checked. A schedule report is not an executable certificate.

Shared selection can route supported integer readouts through the existing
pointwise checker while leaving other arithmetic and layout work in its
original source. It does not itself fuse contractions/readouts, establish
device residency or transfer ownership across calls. Dense and readout tests
exercise independent native implementations and complete ordinary recurrent
programs with unchanged outputs/state. Their byte-machine command counts and
functional Spike instruction counts are not accelerator latency.

Direct batched contractions, arbitrary GQA layouts, KV ownership and mixed
attention floating numerical contracts are not inferred. Only currently
matched supported rank-two islands may route within an unchanged surrounding
program. If an adapter's preload may remain pending when its fence returns,
that fence does not establish the dense-v1 standalone synchronous completion
premise. Missing exact target effects/qualification remain disabled;
the original Phase 1 compiler/profile and numerical contracts are untouched.

The ordinary native model executor also accepts a compiled output name that
differs from its frozen source reference name only through checked source-return
position, shape, type, complete plan and ABI ownership. It reopens the exact
runtime source and capsule, compares every output under the original source
numerical policy, and keeps the actual compiled output names in physical
readback evidence. Candidate alias lists cannot supply this mapping. This
connection can exercise retained Phase 1 emissions against a separately selected
public development experiment without rewriting the compiler or references.
An explicitly selected functional Spike L2 numerical diagnostic leaves required
deeper tiers, source sealing, public-program admission and timing qualification
pending; a numerical match does not supply those missing authorities.

### Explicit reviewed capability selection

The original-host `targetgen.compiler_pass_selection.CompilerPassSelection`
binds an exact live registry, binding and mechanism roster. Its
`prepare_compiler` creates a disjoint compiler copy before retained-source and
edit-authority freezing; it never swaps a package inside the original executor.
Missing capability refuses preparation. Both compared arms must receive the
same selected library and adapter code; candidate policy remains a separately
editable source component. Preparation supplies no execution qualification.

The ordinary `run_stage` host API accepts `compiler_pass_selection`; queued
policy contexts and their binding retain that same owner. Original sealed
model-child transport v3 optionally carries it, alongside the unchanged library
selection. Fresh children execute a source-pinned original factory which reruns
independent admission and must reconstruct the exact registry and resolutions.
A copied successful owner record cannot construct this selection. Original
v1/v2 child schemas and absent-selection defaults retain their interpretation.
This host infrastructure grants no additional candidate imports or target support.

The final functional guard rechecks both arms and requires the selected owner's
original emitted-program checker. A missing checker makes the guard offending;
preparation, a policy request or earlier paired evidence cannot replace it.
These APIs connect custody and execution ownership, not automatic target-name
discovery or formal admission of the provisional Phase 1 checkpoint.

The ordinary authoring CLI can reopen an explicitly reviewed pass selection
using `--compiler-pass-selection`, `--compiler-pass-selection-sha256` and
`--compiler-pass-admission-root` together with the selected library flags. The
SHA-256 pins the exact selection file bytes. The fresh admission directory must
be disjoint from the author stage and outside the selected library. The CLI
first checks the selected search resource boundary, then reruns the original
source-pinned factory within that resource context and compares the live
registry. A saved successful verdict supplies no capability. Original callbacks
retain their independent resource admission; selecting no hard boundary adds
no hard limits. The CLI forwards the resulting owner through the existing
ordinary stage and final checks. Compiler preparation remains a separate host
step before source/edit freezing; these flags do not prepare or enable a pass.

The host-only `llvmlower.reviewed_pass_capabilities` registry transports live
original owners. `PassCapabilityBinding` requires exact hardware, software,
numerical policy, runtime and library identities. An original host may construct
`ReviewedPassCapability(mechanism, binding, owner)` only with the existing owner
whose stable `identity` and source-pinned `revalidate()` reopen its evidence.
`ReviewedPassCapabilities((capability, ...)).resolve(mechanism, binding)` rechecks
that original owner and returns it only for the exact mechanism and binding.
Missing selection returns `None`; duplicate ownership refuses. No target-name
lookup, nearby configuration, copied JSON verdict or label enables a mechanism.

The consumer must still invoke the owner's checked API and original independent
qualification. For example, `timing_model_correspondence` may carry structural
model/facts/source correspondence without supplying native event timing or
hardware qualification. `configuration_cleanup` is a separate owner from dense
source scheduling and cannot qualify its different completion/arithmetic domain.
The registry publishes no owner factory to author sandboxes, changes no original
profile/library release and supplies no target implementation on its own.

For ordinary source builds, the host may wrap its original `SharedSourceRouting`
owner with `reviewed_source_routing.SelectedSourceRouting(registry, binding)`.
`source_capability_binding(owner, runtime_identity=...)` binds its current
software/RTL, numerical policy and checked library sources; the runtime identity
must come from the same original reviewed selection, whose backend verifier
still owns runtime applicability. Register that live owner under
`source_scheduling`, then pass `**selection.build_arguments()` to the existing
ordinary build API. This supplies its existing preparation and provider hooks,
with the exact registry/owner/source binding reopened at preparation, provider
resolution and final provider consumption. An absent capability supplies no
hooks. Only this checked original wrapper may preserve an empty preparation's
ordinary no-provider route; a foreign callable gains no such exception.

Native tests exercise a registry-selected dense implementation and an ordinary
recurrent program through source preparation, independent complete-output
qualification, provider compilation and final linking. These tests qualify the
connection, not a real target adapter. This host API does not select a capability
from saved JSON, add a normal-stage CLI selector, expose owner factories to the
author, or certify unsupported batching, attention effects or target completion.

## Prepare the release before Phase 1

Select explicit canonical source and output paths. Temporary examples below use
the current managed task's `TMPDIR`; retain a release needed by an active run.

```sh
python -m merlin.targetgen.tool_profile_shared \
  --python-source-root "$MERLIN_SOURCE_ROOT" \
  --license-directory "$MERLIN_LICENSE_ROOT" \
  --output-dir "$TMPDIR/common-v6"
```

This stages the existing shared SDK plus exact preceding common facilities and
host checker sources in **one Python source root**. It retains the exporter's
three fixed namespace replacements, excluding backend discovery from the public
namespace. Profile checks bind both the original host namespace hashes and the
published namespace hashes. Algorithm files remain byte-identical to the selected
source release. The manifest, LICENSE and NOTICE receive exact resource grants.
External package versions are recorded; this does not qualify their implementation.

For assisted treatments, also repeat `--tool NAME` for the union of the existing
`tool_registry.ARM_TOOLS` entries being compared. The preparer stages only those
registry-declared tool files, with fixed file/byte bounds; staging grants nothing.
The original bundle still selects each arm's grants and answer masks. A directory
containing staged host checker files is never itself a candidate grant.

Use the normal bundle generator with:

```text
--common-substrate merlin.common_compiler_substrate.v6
--python-source-root <the prepared common-v6 root>
```

Keep the usual descriptor, public corpus, numerical policy, RTL/software input
selection, toolchain and qualification requirements. The ordinary Phase 1
snapshot and authoring admission run unchanged. Its profile must be inherited
exactly by ordinary Phase 2, including queued source-revision search through
`authoring.run_stage` with the existing search configuration and compiler edit
contract. No separate tuner or pass manager is required.

## What Phase 2 changes

The compiler's editable policy can choose supported loop orders, blocking,
operand retention, fusion candidates, copy cleanup or configuration cleanup from
current operation semantics, shapes, layouts and resource descriptions. Compile
those choices through the same shared checks and independent final evaluation.
A policy is ordinary candidate source: the existing immutable revision store
pins its bytes before evaluation, so later author edits cannot change a queued
compiler. There is no new serialized callback or candidate-provided evaluator.

Pass order is also a useful policy choice, within declared prerequisites, IR
levels and invalidations. For example, fusion before tiling may expose reuse;
tiling before fusion may make a composition fit. Phase 1 retains a deterministic
default. A Phase 2 search over alternative orders must use the existing
`PassSpec`/`PipelineSession` checks and bounded repair, rebuilding invalidated
analyses. It cannot move checks before the operations they verify or retain
allocation after spills. The ordinary candidate-source search can change the
order that its compiler expresses; there is no separate automatic enumerator of
all complete-model pass sequences. Some current checked lowering stages have
only one legal order. Their ordering parameter does not create alternatives or
make a different IR level supported.

### Policy, rewrite order and lowering order

Hardware facts constrain legality, rather than prescribe one profitable schedule
for all shapes. Candidate policy is ordinary editable Python. It can introduce
new thresholds or piecewise rules using current dimensions, layouts, jointly
live storage, supported effects and compilation costs. A larger shape can select
a different rule when a panel, bank, accumulator, queue or program-size boundary
is crossed. Such a threshold is a hypothesis until independently evaluated; it
must not be inferred from model identity or protected workload answers.

The candidate chooses which supported optional transformations to request and
the ordered `PointwiseScheduleRewrite` tuple to apply. New bodies can live in
the explicitly reserved candidate `passes/` scope described below. Each rewrite
receives the current schedule's reconstructed storage/configuration effects and
selected resource context, and its result is checked before the next rewrite.
The original source and independent final-byte checks remain required. This
route currently supports the documented bounded integer pointwise family;
it is not a generic dense, attention or cross-IR transformation interface.

Source APIs distinguish two ordering inputs:

| Input | Executed consumer |
| --- | --- |
| `pass_order` | Source admission, optional candidate rewrites, lowering and source conformance in `schedule_source_dense` or `compile_source_pointwise` |
| `schedule_pass_order` | The inner `schedule_compact_dense` or `compile_schedule` pipeline, forwarded by those source APIs and bound into the source analysis context |
| `rewrites` tuple | Actual sequential candidate pointwise rewrite bodies, checked against current effects after every step |

The ordinary selected backend's `propose` result carries these supported inputs
through `SharedSourceRouting`. `PassSpec` supplies prerequisite analyses, IR
level, stage, produced facts and invalidations; `PipelineSession` rejects stale
facts after changes. Mandatory admission, completion, realization and final
checks cannot be omitted or reordered ahead of their inputs. A new pass at a
different representation needs an explicit checked producer/consumer connection,
and a return to an earlier planning stage needs bounded repair and fresh analyses.

The nested ordering connection is a new shared-library release. Select that
explicit release equally for both compared arms; an older frozen SDK retains
its original API and identity. It does not require rewriting or regenerating the
inherited Phase 1 compiler. The retained-revision regression also executes an
editable shape-dependent transformation threshold through the isolated library
and independently checks emitted bytes and complete native outputs on both sides
of that threshold. This tests expressive access, not a learned or profitable rule.

Keep physical capacities, numerical policy, qualification identities, completion
rules and verification gates fixed. Most `ScheduleBudgets.max_*` fields bound
checker work; increasing them is not a performance optimization. The broker's
existing frozen common-transform request schema is unchanged. An agent can also
propose new compiler transformations, but library availability is not evidence
that a reusable rule has been learned.

## Comparison and verified scope

Every V6 treatment receives the same shared algorithms and release identity.
V6 includes generic ISA parsing/generation algorithms as common infrastructure;
historical ISA-tool ablations therefore need a new interpretation. Target facts,
software declarations, runtime tools and independent qualification keep their
original separate grants. Do not compare a library-assisted arm against an arm
silently denied the same library and attribute the difference solely to tuning.

The ordinary retained-queue regression exercises two candidate policies through
public shared APIs, actual encoded/decoded instructions and an independent
neutral native executor. It checks complete output bytes, untouched guards,
persistent settings, requested traffic and a live-edit mutation. The policy and shared compiler run in an isolated `-I -S` process from the
actual frozen source snapshot; every loaded Merlin owner and SDK hash is checked.
The host reopens the resulting bytes with a separate decoder/native executor.
The test substitutes the normal measurement executor, and its external
functional/timing roles are explicitly synthetic. This establishes executable
routing, not a qualified accelerator, a learned heuristic or hardware speedup.
Source, namespace, manifest and frozen-resource mutations must refuse admission;
all-treatment checks require identical common source rosters and SDK identities.

Actual Phase 2 optimization studies still require admitted target hooks, original
numerical/functional gates, calibrated timing, bounded equal search budgets and
frozen policies evaluated on unseen workloads. Whole-model measurements must
include host arithmetic, packing, transfers and dispatch. Command reductions
remain command results until paired timing establishes an execution improvement.

## Pass and abstraction inventory

The exact public module roster is `compile.scheduling.export.PUBLIC_MODULES`.
These are executable shared mechanisms with explicit admission limits, rather
than a promise that every target's ordinary compiler already calls them:

| Mechanism | Shared interface | Present scope |
| --- | --- | --- |
| Blocking and reuse | `schedule_compact_dense`, `schedule_source_dense` | Counted dense loops, M/N order, K panels and tails; synchronous completion and exact accumulator continuation |
| Block scheduling | `schedule_contraction`, `schedule_convolution` | Load order, bounded lookahead and placement; explicit geometry and residency checks |
| Temporary allocation and copy removal | `allocate_temporary_storage`, `compile_schedule` | Invocation-private lifetimes, reservations and exact private copies; finite schedules |
| Physical fusion | `plan_fusion`, `compile_source_pointwise` | Declared integer scalar stages/readouts with checked layouts and effects |
| Configuration cleanup | `eliminate_redundant_configuration` | Pure ordered persistent writes; unknown effects remain barriers |
| Device spilling | `allocate_with_spills` | Explicit exact synchronous save/reload capabilities |
| Complete-row reductions | `compile_row_softmax` | Selected maximum/exponential/sum/normalization algorithm and complete-row workspace |
| Final program checks | `finalize_program` | Actual decoded size, addresses and branches; segmentation needs qualified continuation |

Typed computations and numerical boundaries describe what must execute.
`DenseResources`, `Geometry`, `StorageBank` and reservations describe available
physical resources. `StorageEffects` and `ConfigurationEffects` describe
ownership, reads/writes, persistence and completion. `ScheduleEncoding` and
target adapters connect selected calls to actual encoded/decoded instructions.
`PassSpec`, `PipelineContext`, `AnalysisStamp` and `PipelineSession` enforce
ordering and freshness. `read_scheduling_contracts` resolves selected RTL facts
and reviewed software/ISA effects; structural geometry cannot supply missing
arithmetic or synchronization semantics.

The xDSL schedule dialect additionally represents layout preservation, pack
hoisting, placement, accumulator lifetime, dispatch grouping and interface
selection. A represented decision or stream analysis is not an executed
transformation; verify its consumer and emitted program before claiming a gain.

Remaining generic work includes automatic asynchronous overlap/double buffering,
loop-carried and cross-operation residency, and broader source composition for
contraction epilogues and batched attention/KV layouts. Existing asynchronous
effect records support dependency checks but do not supply those scheduling
transformations. Target work supplies qualified state/completion effects,
addressing, encoders/decoders and actual primitive support. Neither layer should
select behavior by model name or benchmark identity.

Phase 2 may add or modify compiler transformations as well as tune policy.
The host-frozen edit contract selects existing pass symbols. Explicit
`compiler_edit_contract_v2.new_python_components` can also reserve absent
directories for new Python pass modules, each owned by existing surfaces:

```yaml
schema: compiler_edit_contract_v2
new_python_components:
  - directory: passes
    surface_ids: [scheduling]
    reason: checked compiler transformations
```

This fragment supplements the original required decisions and symbol authority;
it cannot grant itself permissions from a candidate manifest. V1 keeps its
existing interpretation. New modules are retained with each candidate revision,
and restoring an earlier revision removes its later pass modules. The ordinary
isolated source-revision regression exercises this route with actual encoded
programs and independently checked native outputs.

The host can construct this authority directly from its verified source inventory:

```python
contract = build_compiler_edit_contract(
    inventory,
    new_python_components=[
        {"directory": "passes", "surface_ids": ["scheduling"], "reason": "checked compiler transformations"}
    ],
)
```

Omitting `new_python_components` preserves the V1 document and digest. Explicitly
providing it selects V2, including an empty list. Each directory must have an
existing inventoried compiler parent (or the package root), own frozen surface
IDs, and a host explanation. The builder checks scoped intent; the freeze gate
independently checks actual directory absence, real parents and source identity.
The source inventory alone cannot prove absence of unindexed files. Python-only
edits under the reserved scope are captured in retained revisions; the original
seed, unrelated compiler controls and evaluation remain protected. This API does
not automatically enable a scope for existing experiment profiles: the host must
explicitly select and freeze it before authoring.

Editable transformation implementations remain separate from frozen numerical
contracts, independent checkers, qualification and evaluators. Modifying a shared
library mechanism requires an explicitly reviewed Phase 2 release selected
equally for compared treatments; the additional-library runtime connection
described above remains necessary. New candidate pass modules do not silently
make a frozen shared SDK writable.

## Complete layer execution artifacts

Ordinary layer execution explicitly retains executed-program V2 evidence: the
effective command buffer, lowered LLVM, linked kernel, complete console, build
receipt and original completed execution invocation. The original ELF and
harness remain content addressed per simulator. V1 keeps its original meaning.
Changed or missing evidence refuses additional pass realization; retention does
not create a numerical or certification verdict. Final target consumers must
also reopen the original graded source, selected runtime and compiler snapshot,
and independently decode the actual issued program and all output values.

Artifact retention is bounded verification work. Its operator walltime is
separate from target execution cycles. Packing, dispatch and transfers inside
the selected execution window remain part of that window. Retaining these
artifacts neither qualifies a timing model nor enables a target capability.

## Continue implementation from another workspace

The `shared-scheduling` branch is an incomplete implementation checkpoint.
Continue the production-routing gap above before expanding policy search:

1. Inspect `compile/scheduling/{dense_source,pointwise_source,row_softmax}.py`,
   the selected contracts, `compile/route_before_build.py` and
   `llvmlower/prepared_model_transform.py`. Connect one supported source family
   through ordinary production preparation using explicit qualified backend
   capabilities, preserving the source fallback for unsupported cases. The
   generic integer pointwise owner above supplies the first connection; keep
   missing real-device capabilities disabled and continue dense/typed-row routing.
2. Replace the test-specific selection for that family with a test of the normal
   entry. Preserve independent checks of complete outputs and carried state, actual emitted
   effects, stale-input and numerical mutations, packing and compilation metrics.
   The test providers are neutral execution fixtures, never target support or
   production compiler implementations to copy into the library.
3. Expose legal policy alternatives to Phase 2 through the same frozen library.
   Keep facts, software semantics, numerical admission, qualification and checker
   limits fixed. Demonstrate held-out transfer separately from helper availability.

The source-spill and source-row ordinary-model regressions live in
`merlin/tests/infra/test_spilled_pointwise_model.py` and
`merlin/tests/infra/test_source_row_model.py`; the shared export and frozen V6
profile tests cover source closure and identical authoring access. Use the
reproduction commands in [the pass reference](../reference/shared_compiler_passes.md).
Local temporary reports and model binaries are not part of a Git checkout;
recreate them with the existing toolchains and task-local `TMPDIR`. Do not infer
hardware qualification or calibrated timing from these neutral regressions.
