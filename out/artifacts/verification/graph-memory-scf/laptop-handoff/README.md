# PAUSED BY USER — laptop handoff: exact reuse→segment memory/SCF composition

**Implementation stopped on the user's explicit request. Do not resume from an
old continuous-work prompt. A NEW user instruction is required before further
proofs, builds, tests, qualifications or integration.** Packaging this handoff
performed no new proof/build/test. No owned jobs remain running. The last owned
build finished with returncode1, with errors preserved below.

Workspace at stop: `/home/jeremy/Projects/merlin`, branch `feat/pass-verification`,
HEAD `229f42c627854c49476a101cb393d0b15044fab8`. All paths below are REPO-RELATIVE.
The unfinished work is uncommitted additions; HEAD alone does not contain it.

## Files to take and ownership

- `SOURCE_MANIFEST.json`: machine-readable ALL54 owned canonical source files
  (6current +48previous), transitive project source dependencies, actual pinned
  reference-provider sources, ancillary reproduction sources and source identities.
- `unfinished-owned.patch`: exact ADDITIVE patch for the6 unfinished new Lean files.
- `all-owned-source.patch`: the4 preserved prior source patches followed by the
  unfinished6-file patch. This is source delivery, not integration authorization.
- `source-snapshot/primary/`: byte-identical canonical owned sources and selected
  project dependencies at their repository-relative paths.
  The dependency closure is conservative. Non-owned dependency snapshots are
  REFERENCE ONLY: do not overwrite another worker or treat them as an owned patch.
  The coordinator should prefer each canonical owner when assembling streams.
- `source-snapshot/reference4499/`: observed selected producer source closure for
  review; use the full pinned read-only Git checkout for actual API execution.
- `prior-patches/`: byte-identical old sealed patches, never rewritten.
- `reports/` and `REPORTS.json`: key actual captures, old gap/countermodel facts,
  current build records and raw FIRST failures. Raw receipt JSON retains its
  original Linux absolute paths; these are HISTORICAL receipts, not laptop proof.
- `HANDOFF_STATE.json`: observed done/unfinished/job/tool/incident statuses.

Own ONLY new `GraphLoopsMemorySCF*.lean`, `lean_graph_loops_memory_scf*.py`, matching
new tests and `out/{artifacts/verification,build}/graph-memory-scf`. There are
currently6 new Lean source files and NO new projection/checker/test Python leaves.
Old48/shared/production/registries/Main/CLI/frozen semantics/specs/locks/foreign
sources/seals are READ-ONLY. No stage/commit/push/PR/agents/downloads/numerical
work/toolchain repair/whole4499 merge. These holds remain on the laptop unless a
NEW user instruction explicitly changes them. Central coordinator assembles the
cross-stream bundle; do not copy another worker's implementation into this track.

## Current assignment and exact gap

Full ACTIVE assignment (already read in full):
`out/artifacts/verification/merlin-passes/progress/planner/assignments/graph-memory-scf-composition/README.md`.
It is copied into `reports/` with its original relative path.

Close the EXACT retained original→reused→segmented chain, not a toy replacement:

| Stage | Repository-relative actual file | SHA256 |
|---|---|---|
| original | `out/artifacts/verification/graph-buffer-reuse/actual-cases-v3/two_epochs/before.mlir` | `bb5d985ac001a8c9e3b04ff889003965c798796f2b3bc351f0eabf5610190110` |
| exact middle | `out/artifacts/verification/graph-buffer-reuse/actual-cases-v3/two_epochs/after.mlir` | `9e2a5e8315002c1ad00e57508729112d4a5e01871e41ced4ecca1a10bf259ff5` |
| segmented final | `out/artifacts/verification/graph-buffer-reuse/actual-neighbor-segment-v1/after.mlir` | `a98bd7576316e4ad9b4ba873beb9dfe5440e9db1f37460cf52628833f3134773` |

Real pinned producer: `4499eebb959c0f92cefd92cc6d80646ba66474ed`, Linux checkout
`/home/jeremy/Projects/merlin-graph-proof-intake`, read-only. Actual APIs are
`transforms.planning.plan_storage/reuse_storage` then `transforms.loops.segment_loops`.
TWO rank2 rectangular initializers change. The preserved old refusal is
`graph-buffer-reuse/neighbor-composition-refusal-v1/GAP.json`: nested source body
regions need separate semantics. That refusal is still correct for its frozen
old checker. The NEW memory/SCF bridge has NOT discharged it yet.

## What actually compiles / what does not

Five current modules compiled successfully in the verified fixed Linux context:

| Current source stem | Successful build record | Meaning proved so far |
|---|---|---|
| GraphLoopsMemorySCF | build-v2 | tagged scalar/index environment, explicit partial pure operations, SSA definition frame, scoped leaf/store event structure |
| GraphLoopsMemorySCFScope | build-v4 | read/frame agreement, IV binding and no-shadow address selection, one-store event and log monotonicity |
| GraphLoopsMemorySCFLoops | build-v6 | rank1/rank2 lexical captures and nested partial old SCF execution; source↔segmented EXACT ordered logs under ONE body meaning |
| GraphLoopsMemorySCFCoverage | build-v8 | parameterized counted/SCF event coverage; rank1 cell event and rank2 full row, preserving correlated outer-prefix choices |
| GraphLoopsMemorySCFRect | build-v9 | decidable rank1/rank2 shape/no-shadow conditions; every successful initializer log covers every valid cell; positive volume |

These are kernel build successes, **not new qualification/audit results**. No new
transitive axiom audit, actual composed checker, mutation suite, qualification or
fresh replay exists. No full new bridge/pipeline/native claim is justified.
The old audits/receipts do not automatically cover these new modules.

`GraphLoopsMemorySCFEffects.lean` is an UNCOMPILED draft. Last build-v10 ended
returncode1. Preserve its current source and raw diagnostics before fixing:

1. `applyHeapWrites_at` line79: over-expansion during simplification leaves
   `writeCell (heap object) event` mismatched with its explicit lambda under
   `List.foldl`. Prefer a small pointwise stored-object equality lemma and use
   the induction hypothesis without globally unfolding both fold functions.
2. `Run.materializes` line101: `valid.positive_volume` cannot resolve on the
   reducible conjunction `WellFormed`. The existing lemma is
   `Initializer.positive_volume valid`; use explicit namespace application.
3. line107: deprecated `dif_pos`; current tool advises `dite_eq_left`.

No fixes were made after the stop. `build-v10/{RUN.json,stdout.log,stderr.log}`
and the exact failed source snapshot are copied in `reports/`. No Effects object
is qualified/importable. All other failed build attempts (v1,v3,v5,v7) are retained;
these were concrete parser/type/elaboration failures, not unexplained HOST failures.

## Semantic design to continue after NEW authorization

The frozen old `Primitives.initialization` admits a complete result. Replace that
premise in the NEW common model with per-operation execution, then derive the old
atomic initializer relation. DO NOT add a whole-program equality or opaque
bridge-agreement premise that restates the desired theorem.

Current model uses lexical capture environments as loop inputs, not mutable loop
carries (actual initializers have no iter_args). `EvalPure` only sees explicit
operands and returns local SSA bindings. `Emit` reads its stored scalar and actual
index SSA values, then emits an ordered `(rowMajorOffset,value)` event. Outer
prefix choices execute ONCE per row and are shared across inner columns; do not
replace them with independent per-coordinate choices. Loop-local environments
cannot escape the body; global/public effects are framed. Partial scalar meanings
remain partial; complete outcomes do not imply totality or categorized failure/
divergence/error-code equivalence.

Next substantial steps:

1. Fix the UNCOMPILED Effects draft after preserving v10. Finish sequential
   store-log heap effects, framing/append laws, successful-run materialization
   values derived from event coverage and independence from prior/stale contents.
2. Give actual memory-state SCF leaf and nested bodies: pure bindings plus one
   `replaceCell` store; nested rank2 memory execution uses the old SCF relation.
   Prove writer-log↔memory execution by per-operation effects and induction,
   including reverse reconstruction. Do not merely DEFINE whole-run conformance.
3. Define the old atomic initializer primitive from the same operation-level
   loop executions/logs. Atomic fill clears out-of-range cells; actual stores
   leave them alone. Prove SAME-WORLD IN-BOUNDS DOMAIN EQUIVALENCE, not whole heap
   equality. Compatible shapes for shared slots are necessary. Lift that relation
   through all old memory instructions to complete possible public outcomes.
4. Instantiate old reuse preservation and old segmentation with ONE operation/
   body/SSA interpretation. Prove whole instruction-list/registry congruence for
   ANY number of changed initializers, and original→exact-middle→final outcome
   composition. Local accepted receipts alone are insufficient.
5. Implement NEW full triple-module projection/checker. A useful host design is
   a new subclass/copy adapter around the immutable reuse decoder: source alloc/
   free handling stays explicit; each source initializer consumes stride-constant
   + generated outer loop, matching add/min/chunk-for/yields and ALL cloned body
   ops including the inner original loop. Use forward per-scope SSA bindings;
   NEVER an ambiguous global inverse for two logical buffers sharing a slot.
   Normalize both candidate bodies to the ORIGINAL source SSA roles. Match all
   outside observers/functions and exact same middle BUFFER, not just receipt hashes.
6. Build independent actual rank1/rank2, width/alignment/capture/public-observer
   and huge-shape examples; valid multiple-initializer mutations; wrong/stale/
   coherently rehashed associations/writer guards; nonzero common-model witnesses
   tied to the ACTUAL whole case; finite xDSL/reference evidence kept separate.
7. Freeze a NEW fixed source/profile and roster; exact public/transitive std3
   audits, originals, ONE retained fresh-process replay with
   `tmp_path_retention_policy=all`, identity postchecks, source-only delivery and
   paper row. Do not rebuild/relabel an old sealed capsule or retry an unknown
   crashing native context to promote it.

Only `propext`, `Classical.choice`, `Quot.sound`; no sorry/admit/new axioms,
native_decide/bv_decide/reflection/unsafe/extern/substituted implementations.
Keep local scalar primitive meanings, raw parser/SSA projection, allocator,
ABI/native/LLVM/hardware fidelity and policy review premises separate. The frozen
MerlinVerify evaluator has no general SCF route; do not invent or edit one.

## Future commands, NOT executed after pause

On this exact Linux context, once a NEW user resumes work, the first intended
command was (use a NEW build label):

```sh
.venv/bin/python out/artifacts/verification/graph-memory-scf/build_v1.py v11 Merlin.GraphLoopsMemorySCFEffects
```

First fix the known source errors; do not simply rerun the unchanged failed v10.
Existing dependency order for the new leaves is core→Scope→Loops→Coverage→Rect→Effects.
The builder is a historical Linux helper with absolute selected receipts; it is
NOT a portable laptop command as written. On the laptop, prepare a FRESH private
profile from source, matching Lean4.35, and inspect/pin its actual imports/sidecars/
host bindings before native/kernel checks. No implicit elan/Lean-MLIR download,
no binary copying across architectures, no Linux receipt path rewrites+rehash
claimed as qualification. Missing matching tool/context remains unavailable.

## Dependencies, incident and earlier completed evidence

- Lean version `leanprover/lean4:v4.35.0`. Linux executable
  `/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean`,9024bytes,
  SHA256`bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38`.
  That Linux executable hash is historical, not a claim about laptop binaries.
- Existing Python3.14 environment; observed xDSL0.73.0. Repository declares
  Python>=3.12 and xDSL>=0.68; current execution relied on the observed provider
  source identities. Exact runtime/version metadata is in HANDOFF_STATE/manifest.
- Fifteen old pure project objects were byte-copied for Linux: shared2,
  Segment4, Unroll core+Grouping2, Reuse7. SOURCE dependencies are included; no
  binaries are in this handoff. Source snapshots plus common intake4 are needed.
- Full Lean-MLIR selection remains unavailable; no download was attempted.
- Shared incident: `Std/Time/Format/Basic.olean` expected7a21... vs observedfbe7...,
  cause UNKNOWN. Preserve the old incident observations. For this NEW task,
  `fixed-context-v1/CONTEXT.json` records an ACTUAL import inventory of the15 old
  objects: suspect NOT_SELECTED, no suspect reread; all selected imports/sidecars
  and actual ldd host libraries matched ORIGINAL fixed pins. New builds checked
  those bindings pre/post and did not rebuild/write the toolchain. These Linux
  findings do not qualify a laptop or a newly changed context.
- Previous sealed reuse batch:214tests +ONE fresh214, zero skips,40root std3 audit;
  prior unroll batch151 twice/27roots; preceding storage/capacity/segmentation
  qualifications remain as originally sealed. Those are HISTORICAL records,
  not new current-task or laptop tests. See old packet READMEs and manifests.
- Preserve the ACTUAL source8-byte→candidate16-byte allocation-reservation
  countermodel, allocation failure/pointer identity/stale/overlap countermodels
  and the old refused join. TOTAL PRIVATE UNOBSERVABLE allocation remains a
  premise. No numerical gate/counterexample was removed. Old all-masked NaN,
  capture/review/native obligations and unobserved old numerical audit status
  remain untouched. The inherited structure debt is `zephyr_model.py`3557>3537;
  no fix was attempted in this track.

Publication and independent review/integration remain central and unauthorized
here. No files from old48 or earlier sealed evidence were edited while preparing
this handoff. No portability fix or newly completed compiler proof is claimed.
