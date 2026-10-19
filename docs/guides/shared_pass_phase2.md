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
compilation. Current dense, pointwise and typed-row full-model regressions supply
explicit test adapters through the ordinary preparation/build hooks. They verify
execution under that selection; automatic production routing from declared
capabilities into these source entries remains unfinished. A generated compiler
can call the exported entries, but importing the SDK alone changes no schedule.

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
allocation after spills. Current entry points validate explicit orders; an
automated search over alternative complete-model pass sequences is future work.

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

## Continue implementation from another workspace

The `shared-scheduling` branch is an incomplete implementation checkpoint.
Start with the production-routing gap above before expanding policy search:

1. Inspect `compile/scheduling/{dense_source,pointwise_source,row_softmax}.py`,
   the selected contracts, `compile/route_before_build.py` and
   `llvmlower/prepared_model_transform.py`. Connect one supported source family
   through ordinary production preparation using explicit qualified backend
   capabilities, preserving the source fallback for unsupported cases.
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
