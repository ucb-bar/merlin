---
title: Shared passes in ordinary compiler authoring
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-18
related: [../reference/shared_compiler_passes, phase2_corpus_search]
code_refs: [src/merlin/targetgen/tool_profile.py, src/merlin/targetgen/tool_profile_shared.py, src/merlin/targetgen/common_transforms.py, packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_stage.py]
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

The existing explicit `CompilerLibraryContract` and component-view APIs can
review and isolate such a library. The ordinary retained-source authoring route
currently binds compiler execution to the original Phase 1 profile owner; an
additional-library selection is **not yet connected through that route**.
Connecting it requires host-owned source/import and sandbox selection, original
numerical/functional gates, queued-job custody and final evaluation using the
same effective Phase 2 release. Changing a profile equality check alone cannot
establish those connections. A new Phase 1 run is one available same-grant
baseline, not the prerequisite for implementing this Phase 2 mode.

The library provides checked mechanisms for tiling/operand reuse, temporary
storage, capability-driven fusion and copy elimination, persistent configuration
cleanup, explicitly supported device spills, complete row reductions and final
program limits. Availability is independent of target support. A backend must
supply typed computation, current resources, exact instruction effects, realized
encoding/decoding and independently qualified execution semantics where required.
Opaque backend output and missing capabilities remain unavailable.

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
