# CURRENT USER RESUME — graph transformations before numerics

Authorization received 2026-10-25, recorded 23:34 UTC. The user said:

> Please tell the agents to continue. Nothing has changed since last time,
> make sure that you stay on the goal of verifying all of the graph
> transformations before doing numerics. Please try to get this to a claimable
> position where we can claim correctness on graph transformations for the
> merlin shared passes.

This NEW resume supersedes the Oct24 stop in all three original conversations.
The laptop handoff is an immutable historical checkpoint, not a current stop.
Resume the existing three workers; do not spawn replacements or restart obsolete
W8A8/TorchAO/quant-ext numerical assignments. Read this file and the current task
after compaction, then continue the current batch. A NEW user stop still wins.

## Latest paper steering

Read [the two paper FIXMEs and required evidence](PAPER_TARGET.md). The deliverable
is a precise transformation/semantics/assumptions/IR-mapping table plus one actual
changed before/after example with retained checker output. Each worker supplies
its own paper-evidence packet while continuing the complete current graph batch.
Graham's independent review complements implementation; it does not replace it.

## Objective and claim discipline

Finish the Merlin-owned shared graph transformation proofs before numerical work.
Use existing sound translation validators and explicit composition interfaces.
The paper claim must enumerate the actual passes and variants it covers; no
unsupported pass may disappear from the denominator. If the full graph roster
is incomplete, say so and report the exact narrower relation proved. LLVM, the
OOT backend, final machine code and hardware are outside this graph claim.

Numerics remain deferred. Preserve arithmetic operation order and parameterize
unchanged primitive meanings explicitly. This does not discharge numerical
definedness, primitive fidelity or existing counterexamples. Do not replace
needed operational correspondence with a hypothesis of whole-program equality.
Generic graph proofs under precise contracts are useful; contracts, independent
specification review, actual IR binding, qualification and model coverage remain
separate obligations.

## Three substantial, disjoint implementation batches

All workers continue the existing primary checkout `/home/jeremy/Projects/merlin`,
`feat/pass-verification`, observed HEAD
`229f42c627854c49476a101cb393d0b15044fab8`. The index is empty; the same five
intentional tracked changes and existing untracked families remain preserved.
Worker branches are allowed, with final code/dependencies delivered to primary.
The production reference remains detached/read-only at
`4499eebb959c0f92cefd92cc6d80646ba66474ed`.

| Existing worker | Current assignment | First checkpoint and complete batch |
| --- | --- | --- |
| W8A8 / GraphLoops | [Memory-SCF composition](../assignments/graph-memory-scf-composition/README.md) | Preserve failed v10; fix the two known Effects proof errors in new work, then prove operational initializer/memory correspondence and validate the actual two-initializer reuse→segment triple under one interpretation. Finish checker, adversarial coverage, audits and frozen qualification. |
| Frontend / GraphRegions | [Declared-call correspondence](../assignments/graph-declared-call-correspondence/README.md) | Continue the three compiled declaration/matching/evaluation modules. Bind actual provider scalar IR and physical calls; derive complete local fusion preservation from per-instruction contracts, with surrounding observers. Finish checked artifact binding, mutations, audits and new qualification in an unaffected context. |
| Former Qinner / GraphState | [Scalar-loop pipeline](../assignments/graph-scalar-loop-pipeline/README.md) | Build on the compiled unchanged-evaluator failure representation. Derive concrete index/range/control and scoped SSA/carry bridges, then compose actual CSE/DCE→LICM→unroll→segment, preserving failures and terminal worlds. Finish checker, mutations, audits and frozen qualification. |

Read each worker's original `laptop-handoff/README.md` for exact source inventories,
failed commands and next lemmas. Do not restore an archived snapshot over live
sources. The previous current task files already describe full multi-hour batches;
this resume does not weaken their acceptance criteria. Update your OWN durable
active/resume record to ACTIVE with this new authorization before implementation.
Work through the complete batch; do not stop after a lemma or ask for routine
reassignment. If a genuine blocker is found, document it and continue useful
independent work within this batch. Report milestones concisely.

## Coordinator work and acceptance

The coordinator owns the pass/variant coverage and claim ledger, theorem/hypothesis
review, exact-boundary composition map and review/integration plan. Audit the
actual pinned source pipeline and existing reports to identify every remaining
graph pass, not just these three active gaps. Record an owner, evidence and open
obligation for each. Communicate assignments between workers only when an actual
shared dependency needs coordination; do not send repeated status prompts.

A paper-ready graph row needs:

1. Actual pass/variant and pinned implementation; complete source/result IR,
   public outputs/observers, untouched context and exact middle artifacts.
2. A Lean soundness theorem with explicit inhabited domain and precise relation;
   recomputed structural facts, checked correspondence and honest parser/host trust.
3. A justified link to existing graph roots and adjacent supported transformations,
   preserving effects, failures, memory/SSA/carry scopes and necessary contracts.
4. Valid actual changed examples, independent variants and compiler-valid mutation
   refusals; no identity-only or vacuous positive coverage.
5. Exact-root transitive standard-three axiom audits and a new frozen qualification
   with originals, one retained fresh-process replay and post-run identity checks.
6. Independent specification/contract review recorded separately from engineering
   evidence; supported variants and remaining model/native obligations explicit.

Only mark a row complete when its own obligations are met. Broad all-graph/full-pass
or pipeline claims stay false while any required row or composition is open.

## Latest publication authorization

The user subsequently said: "once you have the markdown file, please commit and
push everything and give me a path to teh markdown". The coordinator is now
explicitly authorized to review, stage, commit and push pending verification
source/documentation in coherent groups to feat/pass-verification. This overrides
the publication hold BELOW for this coordinated publication. Workers must leave
staging/commits/pushes to the coordinator and continue their assigned graph tasks.
No PR, force-push, protected semantic/spec/lock changes, ignored build/SDK/core
binary publication or blanket experimental-branch merge is authorized. Preserve
unfinished/failed status; publication does not qualify a proof or approve a policy.
Historical sealed records remain immutable and retain their original hold wording.

## Preserved holds and failed contexts

No staging, commits, pushes/PRs, broad merges, reset/switch/stash/prune, foreign
cleanup, production edits or protected shared/frozen/spec/lock changes. Common
interface edits require one coordinated owner. No new agents, downloads,
installed-toolchain writes or silent dependency reselection. Only propext,
Classical.choice and Quot.sound; no sorry/admit/new axioms/native_decide/bv_decide/
reflection/unsafe/extern or implementation substitutions.

Preserve the frontend 21-file bridge as PARTIAL/UNQUALIFIED after the unexplained
Std.Time.Format.Basic binding discrepancy. Preserve configuration elision as
UNQUALIFIED with zero qualified originals/replays after its Lean SIGSEGV. Never
rehash or retry those contexts to promote them. New work uses verified unaffected
fixed contexts, or remains honestly source-only. Preserve the allocator's explicit
total/private/unobservable premise and actual 8→16-byte storage counterexample.
Keep all earlier numerical, native, NaN/overflow and F11/F12/F13 counterexamples.
Final selected capture remains with the user/teammate; don't ask repeatedly.

No policy, primitive-model, native or full-model approval follows from this resume.
Historical qualified sources/seals and the laptop archive remain unchanged.
