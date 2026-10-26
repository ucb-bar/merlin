# Graham: independent graph-proof review and clean-host reproduction

Updated 2026-10-26. Pull **feat/pass-verification**, record the exact commit, and
keep your review in `out/artifacts/verification/graham-graph-review/`.
This is a sustained parallel assignment: complete packages 1–4 in order and
record unresolved findings. Package 5 is an optional additional review.
Do not edit the implementation workers' frozen source cuts.

## What we can claim today

Historical packets contain kernel-checked conditional results for selected
scalar/loop pipelines, shape normalization, memory reuse/segmentation and
declared-call cases. Their domains and relations differ. The newest graph
extensions and service/gate candidates are **UNCOMPILED / UNQUALIFIED**;
source-only tests and generated queries do not establish Lean correctness.
There is no claim that all graph transformations, ordinary compiler routes,
MLIR lowering, numerical behavior or hardware execution are verified.

The paper needs two deliverables: a precise, independently reviewed supported
claim, and an actual before/after IR example whose projection, checker result,
assumptions and compiler acceptance boundary are explained honestly.

## 1. Independently rebuild the fixed graph proof candidates

This is the most useful parallel work. Use your own clean machine and fresh
Lean **4.35.0**, separately verify the official archive identity, and use new
source/object/output directories. Record OS, architecture, toolchain origin,
digest and exact repository commit. Do not copy old `.olean`, libraries,
absolute receipt paths or runtime approval files as fresh authority.

Start with the smaller Q schedule/empty-inline/tail source profiles; then build
W's memory/caller-context profile. Preserve the first compilation failure and
its exact source instead of changing frozen semantics, deleting obligations or
relaxing hypotheses to make it compile. Report any proposed repair separately.

Inputs, relative to the repository:

- `out/artifacts/verification/graph-pipeline-gate/development/publication-cut-v1/SOURCES.json`
  identifies Q's fixed primary sources and separately held producer changes.
- In that same development directory, `schedule-profile-v1`,
  `unroll-inline-v1`, `unroll-tail-v1`, `larger-profiles-v1` and
  `chain-endpoint-v1` contain the actual buffers, source inventories, generated
  queries, exact requested roots and explicit source-only status.
- `out/artifacts/verification/graph-shape-normalization/after-binding-deadline-v1/final-memory-after-v1/publication-context-v1/MANIFEST.json`
  identifies W's fixed 61 owned files. `source-context-v1` has the complete
  118-Lean/51-Python source closure and 14 query/preparation cases;
  `context-source-review-v1/README.md` explains the intended theorem.
- `out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/independent-reproduction/fresh-graph-build-plan-v1/`
  contains `RUNBOOK.md`, `PROFILE_INDEX.json`, `FRESH_QUALIFICATION_ROSTER.json`
  and `BUILD_AND_AUDIT_ROSTER.md`. Its plans are source/data plans, not executed
  build results. A newer `fresh-graph-staging-v1` handoff is separate.

Review each selected dependency DAG; compile project modules in its topological
order with objects isolated per profile. Then compile each complete generated
query, inspect the exact final theorem types and transitive axiom output.
Allowed axioms are only `propext`, `Classical.choice`, `Quot.sound`.
No `sorry`, `admit`, added axioms, native reflection or substituted evaluator.
A successful support-module build does not prove a generated query.

Retain raw command/stdout/stderr/exit status and source/tool/object manifests.
Classify positive acceptance, semantic refusal, parser refusal, infrastructure
failure, timeout and unexecuted cases separately. Replay and adversarial
qualification follow compilation; do not reuse old counts as new results.

Deliver `deliverable-4/REPRODUCTION.md`, a machine-readable result roster,
raw evidence, and the first actionable blocker per profile. If everything
builds, return exact audited theorem names/types and source identities so the
coordinator can prepare qualification. Independent compilation alone does not
activate the live compiler gate or approve the model.

The shared Linux checkout's original/private runtimes remain held because of
an unexplained library hash mismatch. Do not run them. This does not prevent
independent reproduction using your own verified fresh environment.

## 2. Human review of assumptions, theorem strength and IR mapping

Read actual theorem statements, generated queries and source, rather than only
agent summaries. Prioritize these questions:

- Does the relation preserve all public returns, effects, observers, errors and
  terminal worlds, or only possible successful outcomes? State that relation
  per row; do not compose unlike relations without a bridge.
- Memory reuse relies on total/private/unobservable allocation. Preserve the
  counterexample where source peak is 8 bytes and reused reservation is 16.
  W's caller context allows different private clocks/cells: check its public
  frame and suffix correspondence, and its explicit single-call restrictions.
- Do literal/control interpretations and input domains have constructive
  nonzero witnesses? Is any whole-program equality hidden in a hypothesis?
- Trace every actual operation, type/width/property, SSA value, declaration,
  capture, region/carry/yield, untouched entry and public observer into Lean.
  Distinguish trusted Python parsing/projection/matching from Lean recomputation.
- Scheduling/unrolling must retain ordered partial calls, escaping carries,
  observer order and failure worlds. Empty, inline, bulk and tail variants
  require their own supported boundaries.

Start with `independent-proof-review/graph-successor-review-v1/` under the
paper-deadline directory: `HUMAN_WORKSHEET.md`, `HISTORICAL_BOUNDARY.md`,
`Q_REVIEW.md`, `W_REVIEW.md`, `CLAIM_EVIDENCE_MAP.json` and `FINDINGS.json`.
These are source reviews, not your independent approval.
Also read `HUMAN_REVIEW_GUIDE.md` and
`independent-proof-review/registered-coverage-v1/APPENDIX_TABLE.md`.

Deliver `deliverable-5/ASSUMPTIONS.md` and `IR_MAPPING.md`, with reviewer/date,
source hashes, concrete omissions/counterexamples and a verdict per claim:
approved conditional claim, needs narrower wording, or unsupported.
Keep numerical laws/policy review and native fidelity explicitly separate.

## 3. Audit the real compiler acceptance boundary

Your earlier finding that the old checker remained in use is still central.
The new gate/service source and mocked delivery controls are useful progress,
but public native gate activation remains held and ordinary broker routes
are not established merely by those tests.

Trace the real CommonTransformService / pipeline / selected compiler consumer.
Check that only a same-live typed completed proof over actual before/after
buffers can grant eligibility. Saved JSON, local lemmas, plan-only results,
identity flags or replayed reports must not substitute for live authority.
Check shared descriptor/control registries and actual middle-buffer joins
across three buffers: independent pairwise projection IDs can drift.

Review Q's `service-route-review-v1` and `chain-endpoint-v1`; also
`independent-reproduction/service-integration-v2/HANDOFF.md` and
`host-consumer-bridge-v1/PREREQUISITES_AND_PLACEMENT.md`. Producer fixes are
preserved as explicit patches; they are not a wholesale integration of the
experimental producer branch. Work with the coordinator before changing code.

Deliver `deliverable-6/ENFORCEMENT.md`: exact route/file/line map, remaining
bypass or integration obligations, and the strongest supported enforcement
sentence. A host mock demonstrates plumbing, not a fresh kernel proof.

## 4. Finish the two paper FIXME items

Use the retained actual four-stage example at
`out/artifacts/verification/graph-state-pipeline/review/sealed-v1/paper-evidence/PIPELINE_WORKED_EXAMPLE.md`.
It includes CSE/DCE, LICM, ordered unrolling and segmentation, partial external
calls and a post-loop observer. Trace complete original/middle/final buffers,
projection, exact theorem, original checker output and original qualification.
Label original evidence as historical; do not substitute fresh source candidates.

Produce `PAPER_READY.md` with:

1. A precise paragraph stating supported transformations/variants, modeled
   integer/loop/effect semantics, assumptions, IR mapping and trusted components.
2. A compact annotated before/after example plus exact checker outcome and
   independent review references.
3. A table separating kernel-proved components, source candidates, SDK-free
   tests, fresh qualification, human approval and actual-model coverage.
4. Explicit exclusions and remaining obligations.

There are eight registered common transformation kinds. Broader mechanism or
variant inventory rows are not additional proved passes. Never describe the
combined inventory as 91 verified passes or convert test counts to pass counts.
Do not connect this SCF example to the two-buffer panel-pipeline/gSIM figure
without an actual theorem covering that scheduling boundary.

## 5. Optional: independent lowering and SMT boundary audit

`independent-proof-review/lowering-example-review-v1/` contains a retained
16x16 linalg-to-Merlin-interface example, field map, evidence search and claim
worksheet. No matching retained SMT receipt was located. Test definitions are
not evidence that the actual retained pair was solver-checked.

Review shape/type/bitwidth binding, ordered outputs and imported recognizer
source identities. W owns a separate investigation/fix for dtype/digest gaps;
coordinate before changing those files. Findings are currently source-level,
not observed solver false acceptance. Keep linalg/interface SMT checks separate
from Lean SCF graph proofs and conditional quant-ext numerical contracts.

## Coordination and completion

Send findings and patches to Jeremy/coordinator; do not edit sealed evidence,
refresh locks, force-push, activate held contexts or merge experimental branches.
Keep your own source and review commit IDs exact. Flag blockers early, then
continue an independent package. A defensible narrower claim is useful progress.
