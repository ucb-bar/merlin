# CURRENT USER RESUME — 3.5-day paper deadline

The user explicitly resumed implementation and asked for one or two workers to
fix Graham's review findings while the others continue the highest-value graph
work. This supersedes the end-of-day stop. No numerical work or broad new pass
families. The target is a strong, reproducible CONDITIONAL claim for enumerated
graph transformations, with an actual enforced worked example and human-reviewed
assumptions. Full all-graph/model/native verification is not predeclared.

Jeremy's [human review guide](HUMAN_REVIEW_GUIDE.md) gives the code-reading order,
per-family theorem/observation map, evidence checks and exact claim worksheet.
It is a review procedure, not a human sign-off or a declaration that the pending
fixes have been qualified.

Review input: `origin/feat/pass-verification-verf` at
`5a7f872db33d378d6b0c4a294e4099e80efd4956`, deliverables1–3 under
`out/artifacts/verification/graham-graph-review/`. Read with `git show`; do not
merge review-generated test results into a new qualification.

## Ownership and delivery

Primary: `/home/jeremy/Projects/merlin`, `feat/pass-verification`, current published
58dbc7c00. Preserve every other worker's dirty work. The original producer at
`/home/jeremy/Projects/merlin-graph-proof-intake` remains READONLY at4499eebb9.
A separate corrective producer checkout is available at
`out/build/graph-paper-deadline/producer-corrections`, branch
`fix/graph-paper-gate-capture`, based on that exact commit. Its two workers have
disjoint files. Coordinator alone stages/commits/integrates/pushes; all final
implementation lands on primary through reviewed targeted integration, never a
wholesale experimental-branch merge. Record base+patch/new revision honestly.

| Existing worker | Deadline assignment | Exclusive boundary |
| --- | --- | --- |
| Q / GraphState | [Compiler gate repair](gate.md), D1-G09/D3-G10 | Corrective producer lean_checker.py and pipeline enforcement/tests; new primary lean_graph_pipeline_gate* modules/tests. Pause scheduling feature expansion. |
| F / Frontend | [Input-admission fixes](capture.md), D2-G01/G02 | Primary shared admission helper and capture entry hooks; corrective producer certificates.py and matching helper/tests. No gate or W emitter/model edits. |
| W / GraphLoops | [Actual after-program binding](after-binding.md), D2-G03 | Shape/memory model and emitter wrapper, new pair-binding modules/tests; current composition and its evidence. No shared admission or corrective producer edits. |
| Coordinator | Runtime recovery investigation, immutable evidence delivery, claim text and integration | Original-runtime snapshot provenance; full original bundles/exact parser identity for Graham; daily status and paper table. |

After the admission regression batch, F also owns the new common
`lean_graph_original_image_context*.py` runner. Q/W consume that provider instead
of writing separate runtime adapters. It needs coordinator review before native
use; the old Pure runner, failed contexts and installed SDK remain unchanged.

F owns the shared admission interface. Q and W consume it after agreement; they
must not implement competing guards or edit F's entry hooks. Canonical Python
files named by these bugfixes may advance as NEW corrected versions. Old source
snapshots/seals/receipts/failures stay immutable; never update their expected
hashes or imply the fixed source was historically qualified. Core/Sem/Spec/locks
and frozen claims remain protected. No new axioms/sorry/admit/native_decide/
bv_decide/reflection/unsafe/extern or substituted implementation.

## Runtime and test boundary

The selected installed libleanshared binding incident remains UNKNOWN. User
resume does not make that native context valid. No worker may reread/rehash it,
retry held native profiles, repair the SDK or adopt new expected bytes. Existing
qualification capsules remain unchanged. Coordinator will separately review the
PRE-EXISTING independent runtime-copy metadata against ORIGINAL pins; no launch
or worker use until an explicitly justified new context is authorized.

Runtime update: coordinator verified/copied 4,527 original-pinned SDK images to
a new independent private context and reviewed one basic `Init` kernel probe.
That probe passed with zero axioms and the exact eleven-image runtime closure.
The old incident is still UNKNOWN. Graph qualification and worker native use
are NOT authorized by that basic probe; see `runtime-snapshot-review/README.md`.

SDK-free Python producer/parser/projection/data tests and mocked gate unit tests
are allowed after inspecting their call paths for native/SDK actions. Use explicit
isolated runners; no Lean/MLIR compiler invocation or dependency guard that opens
the held SDK. Such tests are not real Lean acceptance. Candidate Lean source may
be developed but stays UNCOMPILED until checking is authorized. Do not conceal a
native/environment failure as a domain refusal.

## Deadline sequence and acceptance

First12h: close unsafe/local-only gate acceptance, resource loss and indirect-call
admission; retain concrete regressions; unblock original evidence delivery and
investigate the runtime. Next24h: execute a genuinely changed supported compiler
boundary through the corrected gate, and demonstrate valid wrong-result, stale,
missing and forged-evidence rejection in a justified context. Next24h: Graham's
independent repro/counterexample checks and human review of the exact claim.
Final24h: freeze one selected source/profile/case population and replay once;
finish paper text/table/example. No late feature expansion or invented test sums.

Local arithmetic/identity lemmas must never grant whole-program proof authority.
The accepted route must bind actual complete before/after programs, domain,
relation, assumptions and checker/source identity. Preserve failures only where
the theorem actually does. Name trusted parser/projection/frame/allocator/control
premises; do not promote model correctness to native or whole-model correctness.

Continue each full assigned batch after compaction, report substantive milestones
briefly, and do not stop after a small lemma. A NEW genuine user stop wins.
