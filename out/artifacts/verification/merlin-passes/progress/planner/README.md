**CURRENT: USER RESUMED ALL THREE GRAPH WORKERS — 2026-10-25.**
The new user instruction supersedes the historical stop below. Graph
transformations remain the priority before numerics. Read the current resume
and claim criteria in `graph-first-20261025/README.md` (planner-relative).
Continue each existing graph batch after compaction; no stale numerical tasks.
The latest coordinator publication authorization and current assignments in that
file supersede historical publication holds and completed task rows below.
Frozen-source, specification-review and failed-context holds remain in force.

**PAUSED BY NEW USER STOP — 2026-10-24 06:12 UTC (Oct23 US/Pacific).**
The user returned and explicitly requested ALL agents stop, document progress and
prepare laptop handoffs. Normal stop messages were accepted by all three current
turns. Only safe checkpoint/documentation/packaging remains authorized now.
Do NOT resume proofs, builds, qualification or task assignment after compaction.
Older continuous-work instructions below are superseded until a NEW user resume.
Stop record: planner/eight-day-paper/USER-STOP-LAPTOP-20261024T061200Z.json.

**CURRENT graph-first scope after user discussion with Agustin.** Numerical proof work
is deferred; all three existing agents now own graph transformation tasks. Worker branches
are allowed, with all implementation delivered to feat/pass-verification. Read the
[graph-first assignments and paper alignment](graph-first-20261023/README.md) and
[current coordination](eight-day-paper/active/README.md) before historical instructions below.
The user requested continuous work while at dinner until a NEW stop. No publication hold
or specification approval is released by this priority change.

# Onboarding for the new Merlin verification planner

**CURRENT: USER RESUMED,2026-10-23 20:22UTC.** Keep W8A8 and quant-ext progressing until
user interruption. Frontend is also resumed by explicit20:29UTC correction. Read the [active coordination record](eight-day-paper/active/README.md)
before historical paused states below. No publication/integration hold is released.

Updated 2026-10-22. Read this first, then the
[active eight-day paper goal](eight-day-paper/README.md) and
[consolidated progress report](../README.md).

**The user approved a scoped pass-correctness artifact as the active goal, targeting
2026-10-30.** Prioritize W8A8, finish TorchAO f32 and supported quant-ext qualification,
record reviewed assumptions and actual artifact coverage, and prepare one frozen combined
qualification. Complete shared-pipeline/model verification remains a longer-term obligation.
The deadline does not authorize weakening a theorem or lifting the publication/integration hold.

This conversation is the planner/coordinator. The existing W8A8, frontend and quant-ext
workers continue in their owned checkouts; the former Qinner worker now owns lower_quant_ext.
Exact deliverables, acceptance criteria, current worktrees, schedule and compaction-resume
rules are in the active goal. It supersedes the historical paused state and initial assignments
below. Existing worker active records retain technical resume commands and frozen identities.

## Objective and user expectations

The long-term objective is complete Lean verification integration and qualification for **Merlin-owned shared passes**
between model2MLIR and the OOT compiler, prioritizing the complete W8A8 pass
(`lower_contraction_int8`). Do not spend effort proving LLVM-owned optimization passes.
Model2MLIR itself, the OOT compiler, LLVM/backend execution and deployment are outside the
current correctness claim unless separately scoped. The user explicitly corrected earlier
scope drift into LLVM pass verification.

Use the verified-checker / translation-validation approach: untrusted Merlin transforms
produce actual before/after programs, and Lean-proved checkers validate the stated relation.
The user wants Axon-style proof discipline. Do not infer that matching an architectural idea
or passing tests makes our system or paper claim equivalent to Axon's.

The user wants sustained progress, clean general solutions and explicit discrepancies.
Existing staging/commit/integration/push holds remain in force; the goal update does not
authorize publication. Do not repeatedly seek permission for already authorized routine
work. Do not quietly substitute fixtures, an easier domain, an assumed policy or a repair
claim for the requested complete pass claim. Explain genuine remaining limits.

This coordinator maintains ownership, interfaces, review and coverage; the separate W8A8
worker retains implementation. Earlier integer/reshape/splat subagent names are historical;
do not assume those workers are active or restart their old LLVM rewrite assignments.
Notify other workers only about a goal/interface change or an actual dependency.

## Mandatory reading and precedence

Read root `AGENT.md`, `CLAUDE.md`, `docs/reference/architecture.md`, the applicable local
`AGENT.md` files, especially `merlin/lean/AGENT.md` and `src/merlin/verify/lean/AGENT.md`.
Read `docs/design/merlin_lean_verification.md`, especially specification independence,
refinement/definedness, policy checking, composition, qualification and the trust ledger.
Read the current `merlin/lean/LEDGER.md` and `findings.md`; they contain other workers' dirt.

The older `out/artifacts/verification/merlin-passes/plan/{README,RULES}.md` and worker briefs
are useful historical context. Their dated schedule, historical A0–A6 assignments and fallback
suggestions are not evidence of current ownership or completion. In particular, the old idea
of simply approving `pow(x,2)=x*x` must not erase the subsequently retained native
counterexamples. User instructions and current evidence take precedence.

## Historical intake checkout facts

The following records the original intake. Current worker locations and assignments are in
the active goal; later worker evidence takes precedence over the old progress/refusal numbers.

Checked on 2026-10-22 without fetching or rerunning qualifications:

- Repository: `/home/jeremy/Projects/merlin`; branch `feat/pass-verification`.
- HEAD: `229f42c627854c49476a101cb393d0b15044fab8`, Qinner concrete decoder.
- W8A8 last pushed checkpoint: `7304d8e501c090782c81e25a8471038fc8a542ac`.
- Frontend checkpoint: `pv/frontend` at `8d814330f`, implementation
  `1838049e4c76d3e9013f12f82f45a2ec0b759304`. Local/origin-tracking refs exist;
  `8d814330f` is not an ancestor of current HEAD. No frontend merge has been performed here.
- `/tmp/merlin-pv-frontend` is missing; Git lists its old worktree as prunable. Its source and
  handoff are committed and recoverable. Do not delete/prune other work as a setup step.
- Six untracked W8A8 source/test files still exactly match the hashes in their final
  qualification record. See the next section.
- Many other tracked/untracked files are present. They are intentional working material,
  not disposable build leftovers. No code cleanup, reset, stash or broad staging is authorized
  merely by this handoff.
- The consolidated report, W8A8 handoff and this planner report are local uncommitted files.
  They must be shared/committed deliberately; a teammate pulling a branch will not obtain them
  until published. No new commit/push was made during documentation-only handoff preparation.

Fetch before deciding integration or push actions. Origin-tracking refs above are observations,
not a claim that the remote has not advanced. Preserve published history; no force-push.

## Historical worker ownership and initial assignments

| Owner | Existing area | Initial task to assign | Boundaries |
|---|---|---|---|
| **This agent: W8A8 worker** | W8A8 rules/checkers, numerical definedness and bounded receipts | Diagnose the preserved TinyLlama public receipt error, checkpoint the qualified files, then continue source-definedness at result 1256 | Do not edit frontend, Qinner, shared Main/CLI or locks independently |
| **Existing frontend worker** | `pv/frontend`; bool repair, TorchAO affine, local-pass candidate/replay | Finish captured-call → independent tensor-reference execution, exposing concrete numerical premises | This work is already owned. Do not assign it to a new teammate as an independent task. |
| **Existing Qinner/other-pass worker** | Qinner plus scalar-square, overrank and bf16 candidates | Confirm exact owned dirt and next review/runtime/integration obligations | Preserve its counterexamples and provisional numerical claims; shared drivers/locks require coordination |
| **You: planner** | Assignment map, interface decisions, integration, review and claim ledger | Verify branch state/ownership; establish the critical path; coordinate one combined frozen qualification | Own final integration decisions; do not self-approve numerical intent or claim all proof gaps are human review |
| **Optional independent teammate** | Unassigned support task | Capture-artifact production, coverage audit or clean-environment reproduction | Capture generation supplies inputs; the frontend worker retains the binding proof. Numerical-law/exporter edits need ownership first. |

Use separate worktrees for new parallel edits. Keep shared interfaces small and agree one owner
for any common numerical layer. A coverage audit should derive the actual Merlin pass roster,
flags and changed boundaries from the pipeline and map them to theorem/checker coverage and
refusals. It must not count trusted LLVM passes as Merlin verification requirements.

## W8A8: exact worker recovery

Detailed source: [W8A8 handoff](../../w8a8/handoff/README.md).

Completed proof batches include bounded helper calls, function-domain-preserving suffixes,
exact total dead-producer deletion and source-derived projection contraction geometry with
exact signed-i32 accumulation bounds. Recorded broad regression: 1184 tests; latest receipt
regression: 239 tests, zero failures/errors/skips, including 53 new cases. Guard: 9037 theorem
audits, zero violations. The batches overlap and are not a unique-total count.

The six files still uncommitted are:

```text
merlin/lean/W8A8BoundedMain.lean
src/merlin/verify/lean/interpreted.py
src/merlin/verify/lean/bounded_numerics.py
src/merlin/verify/lean/bounded_policy.py
merlin/tests/ir/test_lean_w8a8_bounded_receipts.py
merlin/tests/ir/test_lean_w8a8_bounded_protocol.py
```

The bounded route uses its own standalone Lean entry, authoritative exports, exact per-function
contracts, suffix/handoff joins, runtime-artifact freshness and a fresh theorem-axiom audit.
It deliberately remains draft: successful conditional checks return `abstracted`, exit 1.
The receipt runtime digest binds the actual interpreted execution capsule, not an unrelated
public executable. Package/toolchain trust remains explicit in the existing ledger.

Qualification:
`out/artifacts/verification/merlin-passes/w8a8/bounded-receipts-qualification-v1.json`.
It records source hashes, 239 cases in 299.640 seconds, the unchanged 1396-record freeze and
752-file runtime capsule. The private Lean snapshot is
`out/build/lean-w8a8-source-integration/repo`. Its Lean sources are isolated but Python source/
tests link to the working tree. Do not copy all foreign Lean modules into it or rebuild it
while receipts are being qualified. The detailed handoff contains exact environment exports.

Two independent open issues:

1. **Public real-model receipt error:** retained TinyLlama attempt exits 2, `unmeasured`,
   `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. No bounded certificate exists.
   Structural preflight completed. The responsible subprocess has not been isolated. First
   retain raw auditor/interpreter stdout, stderr and exit status using already exported JSON;
   diagnose the underlying cause before adding a regression or changing error handling.
   Evidence: `w8a8/bounded-receipts-tinyllama-v1.json` and its same-named directory.
2. **Numerical definedness:** the earlier standalone checker reaches quantization result 1256,
   after 902 supported prefix operations, and refuses the RMSnorm/product output domain.
   Finite operands do not imply finite floating products. This needs justified numerical
   evidence, not an unconditional finite-output assumption.

Seven source and three dynamic primitive numerical laws remain hypotheses. No concrete IEEE
model proving them has been admitted. `whole_pass_verified`, `policy_reviewed` and
`primitive_model_laws_verified` remain false. Real-model execution failure is not a numerical
counterexample, and conditional proofs are not full pass verification.

## Frontend: exact remaining proof work

Recover the committed handoff without the missing temporary worktree:

```sh
git show 8d814330f:out/artifacts/verification/merlin-passes/local-passes/HANDOFF.md
git show 8d814330f:out/artifacts/verification/merlin-passes/local-passes/REMAINING.md
git show 8d814330f:out/artifacts/verification/merlin-passes/local-passes/TORCHAO-REVIEW.md
```

`BoolToFloatProgram.check_sound_ieee` proves the complete f32 repair intent: false/+0,
true/+1, poison preserved, including nested regions, calls and suffixes. Its two float cases
use Lean's existing IEEE model. The actual SmolVLA denoise boundary has two sites; the older
full SmolVLA boundary has zero. This is an intent repair, not original signed-cast equality.
Other production float formats refuse. Independent admission remains pending.

TorchAO affine has an exact scalar ordered-expression theorem and complete correspondence
to a source-driven materialization, **not execution refinement of captured opaque TorchAO
calls**. Receipts explicitly say `proved_materialization_provisional` and
`source_call_refinement_proved=false`. Complete the independent tensor-reference bridge for
row reductions/layout, poison, scale addressing, declarations and calls. Then discharge
`AbsReductionCorrect`, `ZeroPointElisionCorrect` and clamp/rounding/conversion fidelity for
the reviewed concrete domain. There is a countermodel to universal +zero elision for arbitrary
frozen FloatModel; a proof cannot assume that law implicitly.

Recorded qualification: 33 new/qualification + 39 regression tests, no skips; 10502 theorem
audits, zero violations. Three pre-existing lock/admission guard differences remain. A census
found 230 quantizers and 627 i64-zero-point dequantizers outside the narrow existing proof.
Extending these contracts is cross-cutting work to assign explicitly, not an automatically
independent fourth worker task.

## Qinner and other pass candidates

Read `merlin/lean/QINNER_REVIEW.md`. Qinner's concrete decoder handles supported static
little-endian, row-major tensors and canonical byte-per-bool i1. It proves typing,
initialization and exact bits and specializes complete-program correspondence.

Its source tagged empties mean named archive bindings; ordinary frozen `tensor.empty` means
poison. Do not relabel this intent-relative binding theorem as literal original-MLIR equality.
Archive authenticity, host/NumPy/C/compiler/runtime agreement and actual delivery remain
explicit obligations. Unchanged external calls must share interpretations. Initializing added
tensors does not prove whole-model success or poison freedom.

Recorded qualification: 161 fast tests and two slow tests; all 11 program + 16 decoder
receipts replay; 2587 theorem audits, zero violations. Nine synthetic positive boundaries
perform 18 lifts. Retained TinyLlama/SmolVLA are unchanged zero-lift boundaries.

Other same-worker packets under `out/artifacts/verification/merlin-passes/frontend/`:

- `scalar-square/typing/review/README.md`: operational typing of the captured TinyLlama source;
  complete-program composition work exists. Typing allows poison and preserves failed outcomes;
  it does not prove progress/success. Native pow-versus-multiply one-ULP counterexamples remain.
- `overrank-matmul/review/README.md`: source-derived batched-linear intent repair and a checked
  standalone kernel domain. Original overrank MLIR is invalid; native NaN payload equality is
  not established. Intent approval remains necessary.
- `bf16-f32acc/review/README.md`: modular complete-output equality, recognized-site policy
  qualification, coverage and local-domain nonvacuity. SmolVLA has 24 sites; TinyLlama has zero.
  Nonzero initialization and fused/unfused behavior require explicit policy treatment.
  Generated after-approval patches have not been approved/applied by this worker.

Substantial candidate files for these families are still untracked. Ask their existing owner
to identify the intended commit series before integration; do not discard or stage them en masse.

## External inputs, decisions and review

Agustin pointed to `examples/gemmini` on EL4. Reference intake pins
`0cc289de52a10bed256f32c90cfbb54a21b3c81f`. Historical environments use Torch 2.10 with
TorchAO 0.16 for TinyLlama/ResNet and 0.15 for SmolVLA. Do not infer a capture's actual version
from its model name or whichever Python environment is installed.

The selected static grouped-16 recipes, older development captures and synthetic dynamic
per-token tests are different artifacts. Need actual selected `model.mlir`, capture receipt,
recipe bytes/hash, scales and exact software identities. Models discussed are TinyLlama and
SmolVLA; the final paper demonstration configuration is not established by these reports.

The user approved explicitly including the all-zero-row W8A8 scale rule `0.00001`. They also
authorized provisional assumptions during implementation, but this is not permission to
admit new axioms or label provisional contracts reviewed. Agustin was named as independent
reviewer. The W8A8 review packet is `w8a8/policy-review/README.md`. Keep engineering moving on
independent tasks while concrete review decisions are pending; review is not the only blocker.

## Safety, proof and claim requirements

- Do not edit the production transform being verified except authorized byte-neutral recording.
  Report pass bugs and preserve reproductions. Separate any explicitly authorized fix.
- Treat Core, Sem, Check/Types and Typing, Spec, Claims, Assumptions, SPEC_LOCK and shared build/
  toolchain interfaces as protected pending the applicable review. Put additive lemmas in
  owned files; a new planner role does not itself authorize changing frozen meanings.
- No sorry/admit/new unreviewed axioms/native_decide/bv_decide/unsafe/partial/extern or
  implemented_by under MerlinVerify. Audit all imported theorem dependencies. Allowed ordinary
  axioms are propext, Classical.choice and Quot.sound plus separately reviewed assumptions.
- Write and review the intended theorem first. Checkers must recompute required facts; hints
  are untrusted. Unsupported input is refused/abstracted, never skipped or called verified.
- Separate exact equality, refinement, policy repair, operational typing, NotStuck, success and
  poison freedom. Preserve explicit input domains and prove they are inhabited where relevant.
- Keep native differentials, universal Lean results, concrete primitive laws, specification
  approval, compiled-checker trust and external execution assumptions as separate evidence.
- A positive fixture or an unchanged model boundary does not establish positive model coverage.
  Do not publish a percentage without a concrete denominator and evidence-based completion rule.
- Preserve W8A8 F11/F12/F13 and scalar-square, overrank, bf16 counterexamples. Do not replace
  reported native differences with an assumed universal equality.

## Integration procedure and definition of done

1. Establish ownership and inspect actual Git state. Recover frontend work in a separate
   checkout; do not switch/reset the dirty shared checkout. Preserve uncommitted work first.
2. Assign the W8A8 execution failure, frontend bridge and external capture/review tasks in
   parallel. Give one owner the common quantization/numerical interfaces.
3. Review theorem statements and all hypotheses. Track every gap as implementation, concrete
   semantics, domain, capture binding, independent review, runtime correspondence or integration.
4. Check actual pass outputs on the declared selected capture, including source-definedness
   and nontrivial changed sites. Require valid corrupted-output refusals and relevant native
   differential tests. A compiler refusal is not a successful Lean mutation test.
5. Freeze source, interpreter/binary, local imports, exporter, lock, contracts and capture bytes
   for qualification. Run suitable regressions, fresh dependency/axiom audits and receipt replay.
   Check identities again afterward. Regenerate receipts after changes; never relabel old hashes.
6. Review each unpublished commit and final range after fetching; verify the remote is an
   ancestor before pushing. Commit only owned paths, use coherent conventional commits, do not
   force-push or open a PR without explicit user authorization.
7. Claim completion only when every required changed boundary composes, concrete premises and
   input domains are justified, reviews/pins are admitted, actual artifacts are bound, and
   required checks pass or scope/assumptions are explicitly approved. Do not redefine full
   W8A8 completion around whichever narrower fragment already passes.

Lean is v4.35.0; use the preserved environment in the W8A8 handoff. Qualification sets
`MERLIN_LEAN_REQUIRED=1` and retains successful artifacts with
`-o tmp_path_retention_policy=all`. The default pytest retention otherwise deletes successful
directories. Avoid redundant huge builds and competing Lean jobs; freeze shared builds during
receipt runs. Existing unrelated structure failure is `zephyr_model.py` 3557 > 3537 lines.
That failure and frontend lock/admission differences must be reported, not counted as passes.

Maintain a compact per-worker status: owned files, commit(s), exact claim/hypotheses, actual
boundary coverage, tests/audit/replay evidence, open discrepancies and next task. Require a
reviewable progress report before reassigning another worker's unfinished task. Human review,
paper wording and arbitrary scheduling pressure must not hide remaining implementation work.
