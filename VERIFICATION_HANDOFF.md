> **CURRENT2026-10-28:** MACC gate/publisher code and candidate qualification reviewed for integration; human policy approval remains absent. See [review](out/artifacts/verification/merlin-passes/bf16-integration-20261028/COORDINATOR-REVIEW/README.md). Active agents are on scoped loops, then bounded buffer reuse, in `/home/jeremy/Projects/merlin-loop-verification` (`feat/loop-transform-verification`).

> **LATEST2026-10-28:** User directed all three agents to work strictly on MACC and stop version/toolchain investigation. Read [MACC_ONLY](out/artifacts/verification/merlin-passes/bf16-integration-20261028/MACC_ONLY.md).

> **CURRENT2026-10-28 USER RESUME: bf16/f16 f32 accumulation only.** Three original workers are resumed on `feat/bf16-f32acc-verification` in `/home/jeremy/Projects/merlin-bf16-verification`. Read [current task/ownership](out/artifacts/verification/merlin-passes/bf16-integration-20261028/README.md) and your role TASK.md. The main checkout and other workers remain untouched. This supersedes the historical stop below for this bounded assignment only; old runtime holds and human-review requirements remain.

# Merlin verification: START HERE

**2026-10-26: user requested STOP at 5% weekly allowance.** All five workers
were told to checkpoint, write portable handoffs and stop. Do not restart them
from historical prompts. Resume only after a new user instruction.

Primary delivery branch: **feat/pass-verification**. Pull it and record the
exact commit before working. Numerical work remains deferred; the goal is a
paper-defensible claim for Merlin-owned graph transformations in the shared
interval between model2MLIR and OOT. LLVM-owned passes are outside scope.

## Worker handoffs

Each short pointer below identifies the detailed handoff, exact owned source
manifest/patch, current checkpoint, open obligations and next task. Read your
pointer first, then its linked handoff. Preserve other workers' source cuts.

- [W: memory/shape/caller-context and SMT binding](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/W.md)
- [Q: graph gate, whole-entry schedule/unroll and integration](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/Q.md)
- [F: declared calls and sole runtime investigation](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/F.md)
- [Prepare: independent source/claim/lowering review](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/Prepare.md)
- [Assign: portable source DAGs, staging and reproduction](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/Assign.md)

## Graham and the next coordinator

[Graham's substantial parallel assignment](out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025/GRAHAM_HANDOFF.md):
independent clean-host Lean4.35 rebuild, human theorem/assumption/IR mapping
review, actual compiler gate audit, and the two paper FIXME deliverables.

[Planner onboarding](out/artifacts/verification/merlin-passes/progress/planner/README.md)
and [current stop/runtime state](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/CURRENT_WORK.json).
Read repository instructions and docs/design/merlin_lean_verification.md.

## Do not lose these distinctions

- Historical qualified proofs retain their original sources, domains, relations,
  runtime bindings and counts. They do not qualify later source candidates.
- Newly published graph extensions are **UNCOMPILED / UNQUALIFIED**. Source
  preparation tests, generated queries and mock host delivery are not Lean
  acceptance, independent review or ordinary compiler enforcement.
- The original/private runtime library hash incident is unresolved. Do not run,
  reselect, rehash or promote those contexts. A fresh official SDK/context and
  new compilation/audits/qualification remain necessary. The SDK watchdog and
  archive Context candidates are source-only; no real SDK download was launched.
- The successful2144-file original IR transport is DATA evidence only. It does
  not clear the runtime hold or fix the separate Omega/IR failure.
- Memory proofs can concern possible public successes under total/private/
  unobservable allocation; scalar/control proofs can preserve complete errors
  and terminal worlds. Never silently equate these relations.
- Preserve finite-allocation, numerical-overflow and other counterexamples;
  frozen Core/Sem/Spec/locks/capsules remain read-only. No new axioms/sorry/admit/
  native_decide/bv_decide/substituted evaluator.
- New host/service/consumer integration is incomplete and public native gate
  activation is held. A saved receipt JSON grants no live proof authority.

Finished source, patches, review documents and selected portable source/query
packets are published. SDKs, compiled objects, giant build trees and duplicate
scratch outputs stay outside Git. Exact source/packet manifests record the
transfer boundary; missing external evidence must be reported, never invented.
