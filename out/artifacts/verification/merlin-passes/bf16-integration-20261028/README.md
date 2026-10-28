> **LATEST USER DIRECTION: MACC only; stop toolchain/version-mismatch investigation. Read [MACC_ONLY.md](MACC_ONLY.md) FIRST. It supersedes the historical custom environment-approval/forensics tasks below. Ordinary builds/checks may proceed using normal repository tools; actual failures remain failures.**

# CURRENT USER RESUME — bf16/f16 f32 accumulation integration

Date:2026-10-28. This is a NEW explicit user assignment and supersedes the Oct26 stop and historical W8A8/graph/quant-ext queues FOR THESE THREE WORKERS ONLY. Two lighter workers remain stopped. Continue this assignment after compaction by reading this file and your own ACTIVE.md; a genuinely newer user stop wins. Do not answer an outdated original prompt or return to numerics/graph work outside this scope.

Worktree: `/home/jeremy/Projects/merlin-bf16-verification`.
Branch: `feat/bf16-f32acc-verification`.
Base: `65e44afbf4dbb6c7270e8936c4ef76b579a78492` from updated feat/pass-verification.
Use this worktree explicitly for every command/edit. The main checkout `/home/jeremy/Projects/merlin` stays on feat/pass-verification and is READ-ONLY for this task. Do not switch/reset/stash/prune/stage/commit/push any checkout. Coordinator owns integration and Git index. Do not start extra agents or change models. Preserve other workers' ownership; communicate only dependency/interface changes and meaningful milestones.

Read root and relevant local AGENT.md, CLAUDE.md, docs/reference/architecture.md, docs/design/merlin_lean_verification.md, docs/guides/lean_pass_gate.md, merlin/lean/LEDGER.md, and the existing bf16 review packet. The latter lives at out/artifacts/verification/merlin-passes/frontend/bf16-f32acc/review/README.md. Most large historical evidence and candidate.json/after-approval.patch are present ONLY under the original main checkout's out/ tree: those may be read as data, never overwritten/promoted. No broad out/ symlink or reuse of historical .olean/native objects. Use a NEW output tree here with exact source/tool identities.

## Scope and claim

Deliver integration and fresh engineering qualification for the EXISTING narrow bf16/f16 -> f32-accumulation policy: checked two-input contractions, supported source maps, literal +0.0 splat/fill initialization, ordered accumulation, exact permitted flags and one final truncation. All actual changed surrounding IR and every eligible site must be accounted for. Initial target is the existing zero-init proof scope; no nonzero-destination expansion, arbitrary mixed fusion, native bit-exactness, whole-model definedness or all-pass claim. Keep F14/F18 and all other counterexamples.

The original half-accumulating source is NOT numerically equivalent to f32 accumulation. A policy-specific theorem/handler is required; do not substitute tensor_refinement or passing numerical tests. All bounds, initialization, arity, FloatModel/SSA/call/descriptor/lowering premises remain explicit. Preserve frozen Core/Sem/Spec/SPEC_LOCK and original qualification capsules. No axioms/sorry/admit/native_decide/bv_decide/unsafe/extern/replaced evaluator, no semantic weakening, no production pass modification.

Independent human approval of the candidate policy is still absent. Agents may audit/prepare questions and an UNAPPLIED pin patch; no agent may mark human_reviewed=true or install policy/locks. Implement all reviewable mechanics and tests so approval can be the final dependency. Real approved acceptance waits for an explicit independently attributable review record and matching exact specification/pins. Never treat abstracted/human_review_pending/rejected/unavailable as an accepted gate result. gate_worker currently accepts every normally returning handler: the new adapter must explicitly reject every non-admitted outcome.

## Shared interface / ownership

Canonical pass key: lower_bf16_matmul_f32acc.
Proposed built-in check name: bf16_f32acc_policy.
Q owns the handler/protocol and publishes INTERFACE.md early. F consumes that protocol and owns independent tests/build/qualification. W is read-only on implementation/proofs and owns semantic/claim review reports. Changes to this interface go through coordinator; no competing context runners or gate protocols.

Q (integration): src/merlin/verify/bf16_policy_gate.py (new), gate_checks.py, gate_worker.py, pass_validation.py; narrowly scoped actual hook in runtime/backends/zephyr_model.py; necessary adapter changes in verify/lean/bf16_f32acc.py and policy publication plumbing after a concrete dependency note. Q may add its own tiny handler-unit tests with a distinct name. No production bf16 algorithm, Lean proof, lock, F test or W review edits.
F (qualification/environment): new merlin/tests/ir/test_lean_bf16_policy_gate.py and dedicated bf16 qualification tests/fixtures/runner, plus own qualification output. Existing tests may be extended only within this scope; do not rewrite frozen evidence. F alone may own NEW SDK/build/native jobs after selecting a reviewed unaffected context. W/Q do not run SDK guards/Lean/native. Existing installed/private graph incident contexts remain held; no old-image retries or pin refresh. Send coordinator a small exact environment-selection request before any real image acquisition or native run; meanwhile run SDK-free tests and prepare captures/fixtures as data. Coordinator can review/authorize a concrete safe selection without requiring another user permission question.
W (review): own review/ under this assignment directory, independent read-only source/model/IR inventory, source-only probes and an UNAPPLIED proposed spec/pin patch. No implementation/proof/lock/production edits or SDK/native actions. Agent audit is not human approval.
Coordinator: ownership, shared interface decisions, explicit environment authorization, final source freeze, human-review intake, eventual approved spec/lock installation, integration/publication.

## Environment / preserved observations

Use the existing Python at /home/jeremy/Projects/merlin/.venv/bin/python only with explicit PYTHONPATH=<this-worktree>/src and MERLIN_REPO_ROOT=<this-worktree>; verify imported module paths before relying on tests. This Python is tooling, not a selected Lean runtime. Use merlin.common.paths for generated paths in library/runner code. No regex in core.
The bf16 source and old receipt hashes bind older shared semantics/locks. Origin changed Q/DQ semantics and SPEC_LOCK in the base: old receipts are HISTORICAL until newly exported/checked/replayed. Do not apply the old after-approval.patch blindly. The separate fix/bf16-f32acc-contractions production branch extends nonzero initialization; do not merge it into this initial scope.
All old runtime/hash/Omega/IR failures remain UNKNOWN; no automatic global clearance follows from Graham's different new checker work. Working SDK/build evidence must be explicitly bound to this task. Failed/timeout/all-skipped executions are unavailable, never successful refusals or qualification.

## Work discipline and end state

Work through your whole bounded queue, using large milestones and few messages. Keep ACTIVE.md with precise next action and current outputs to survive compaction. At a real dependency, finish independent work and document a concrete blocker with reproducible evidence; do not invent approval or duplicate another owner's work. Once your entire assigned batch is complete, send coordinator source paths, exact tests/outcomes, open obligations and no/live-job status.
Final engineering packet must distinguish proof compilation/audits, handler-unit/mocked tests, actual compiler changed positives, parser-valid negative transformations, real checker refusals vs infrastructure failures, replay/byte binding, human review and actual-model coverage. Historical 24 sites repeated in a segment are not48. Frozen artifacts stay untouched. Full W8A8/shared-pipeline verification remains false.
