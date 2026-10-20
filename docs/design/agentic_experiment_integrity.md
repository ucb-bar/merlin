---
title: "Design: rules for citable agentic-compiler experiments"
kind: design
status: current
owner: core
last_verified: 2026-10-08
related: [derived_capsule_axes, capsule_phase_split, radiance_staged_evaluation]
code_refs:
  - packages/merlin-experiments/src/merlin/targetgen/capsule_grade.py
  - packages/merlin-experiments/src/merlin/targetgen/capsule_runner.py
  - src/merlin/verify/epilogue_applicability.py
  - merlin/experiments/capsule_bench/harness/agg_agentic_results.py
  - merlin/experiments/capsule_bench/harness/regrade_run_snapshot.py
  - merlin/experiments/capsule_bench/harness/freeze_run.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/promotion.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/qa.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/source_inputs.py
  - merlin/experiments/capsule_bench/harness/conformance.py
  - merlin/experiments/capsule_bench/harness/readiness_check.py
  - merlin/experiments/capsule_bench/harness/verify_no_cheat.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/providers/codex_agent.py
  - build_tools/scripts/check_no_answer_keys.py
  - build_tools/scripts/check_provenance.py
  - merlin/tests/infra/test_continuous_is_the_default.py
---

# Rules for citable agentic-compiler experiments

A capsule-bench campaign compares compiler authoring treatments under the same
target, model and declared budget. Harness behavior, grading scope and accounting
must remain controlled so that observed differences can be attributed to the
treatment. The rules below state the required evidence and enforcement limits.

Current functional feedback implementations live in `merlin_experiments.phase1.feedback`:
`qa` owns grading and redaction, `promotion` owns per-capsule certification, and `snapshots`
owns enqueue-time candidate copies. Native launchers call these owners; relocating them does
not reattribute historical experiment evidence or qualify a complete installed controller.
The newer implementation-source inventory in `merlin_experiments.phase1.source_inputs` binds
native and outer Phase-1 owners for freshly frozen runs, including the historical target-access
policy, the selected target's capability contract, and the software specification named by the
run's numeric profile. It does not retroactively establish ownership for legacy freezes that
recorded only a repository SHA and submission bytes.
For a new installed run, the reviewed Phase 0 release, selected tool/provider bytes
and frozen Phase 1 implementation are separate identities. Reusing an immutable
historical capture does not relabel its producer as the current tool; launch
readiness and final grading still bind the current selections independently.

## 1. Cite the tier, never the bare score

**Rule.** A pass count is quoted together with `tier_reached` and `pass_evidence`
(`rtl_backed`, `cheap_tier_only`). A pass only at the loop tier (L2) is not a certified result.

**Gate.** `capsule_grade.py` writes `pass_evidence` on every score. `agg_agentic_results.py` reads
L3 evidence rather than the headline. A whole-model capsule that must accelerate is scored only
from the submitted program's own evidence (`candidate_native_model_check`): the runner-owned
host-dispatch graph is kept as a diagnostic, its tiers, numerics and cycles never enter the score,
and a candidate tier without an explicit `derived_from_rtl` flag is not counted as RTL-backed by
name. The agent-facing verdict carries the same redacted candidate facet and no execution digest
for such a row, so a certificate is never reused across the two programs. A capsule that inherits
a deeper tier from a sibling reads that sibling's certificate only from the same selected suite
run, never from ambient output roots. Every engine on the ladder answers memory in issue order, so
a program correct only under that order passes L2 and L3 and can still fail on a board. The opt-in
memory-order tier (`MERLIN_L3_ORDER_SEEDS` > 0) sweeps a passing cert tier under seeded
out-of-order responses when the decoded trace shows an unbarriered ordering hazard, and fails it on
`<tier>_order` with the reproducing seeds. Off, a grade is byte-identical, so a cited cert result
states whether that tier ran.

## 2. Compare arms on one cohort, pinned before launch

**Rule.** The capsule set is pinned in the descriptor (`grading.search_cohort`) before any arm
launches. Cross-arm numbers are computed on the intersection of the capsules each arm actually
measured. The descriptor of a live run is never edited.

**Gate.** `agg_agentic_results.py` (`common_cohort`, `--cohort`); the descriptor-digest check in the
grader; `n_not_measured` on the score.

## 3. Freeze the harness for the whole measurement window

**Rule.** A campaign runs against one harness revision. No grader plane, prompt, or library change
lands between the first and last arm. When a fix must land, **every** arm is re-graded under it with
`regrade_run_snapshot.py`, and the comparison is restated.

**Gate.** `freeze_run.py` records `repo_sha` and the submission digest. Enqueue-time promotion
snapshots bind candidate source bytes; the shared implementation inventory binds source ownership
for newly frozen Phase-1 runs. Model capsules are copied into a grade-owned, byte-frozen snapshot
before the package build, so a corpus generation collected mid-grade cannot change what is graded.
None of these makes a legacy campaign's in-process mid-batch working-tree
imports trustworthy. Such a campaign still needs a revision pin or a fresh freeze and regrade of
every arm before its comparison can be cited.

## 4. One stop rule for every arm, decided in advance

**Rule.** The rule that ends a run (plateau over N grades, wall budget, token budget) is written down
before launch and applied to every arm identically. Whole-run totals are reported only for arms
stopped by the same rule. Otherwise compare **cost to reach a milestone**, each arm on its own clock.

**Gate.** The plateau detector (`plateau.json`) is operator-side only and is never shown to the agent.
Applying it uniformly is still a process rule.

## 5. A check must be able to fail

**Rule.** Every gate is mutation-proved: break the thing it exists to catch and confirm that it
fails. A skipped tier is not a pass (`not_run_is_not_pass`).

**Gate.** Mutation tests next to each gate. Examples: the promotion-wiring tests, and the holdout
check that must fail when it is blind. See `merlin/tests/infra/`. `check_mutation_proofs.py` requires
every `check_*` gate to name a runnable mutation that makes it fail (`gate_mutations.yaml`), and CI
runs each recorded proof with `--prove`.

The same rule holds for the instruction policy a capsule is graded under. `capsule_runner` scans every
ELF a grade linked for the sealed prohibited roles: a prohibited instruction anywhere fails the capsule
(`PROHIBITED_INSTRUCTION`), and a grade that linked no ELF, a scan that could not run, or roles that
match no instruction leave it `PROHIBITION_NOT_MEASURED` — incomplete, never passed.

## 6. Prove the arm was served before blaming the model

**Rule.** Before a result is read as "the agent could not", confirm from **inside** the sandbox that
the arm's grants were visible, the feedback channel was healthy, and the conformance parser could see
the tool use.

**Gate.** `readiness_check.py` (visibility checked from inside the sandbox; its Chipyard timing probe
runs on an explicitly selected reference backend, recorded by path, package id and manifest digest,
and fails without one rather than timing whatever package is on disk); `conformance.py`;
feedback-channel health on every verdict. Under the outer sandbox, the Codex driver runs its
candidate commands in a second, deny-by-default permission profile and proves it with a no-model
probe through the same mounts (credential unreadable, workspace writable, tools runnable) before
the first paid turn; it refuses to launch rather than fall back to bypassing that profile.

## 7. Answer keys stay out of reach, and holdouts are real

**Rule.** Goldens and hidden capsules are untracked, excluded from the wheel, and masked by the
sandbox. Hidden sets are genuine holdouts along a declared axis, not renamed public capsules. Goldens
are never restored while a run is live.

**Gate.** `check_no_answer_keys.py` (pre-commit and Stop hook); `verify_no_cheat.py`;
`check_provenance.py` for hardware revisions.

## 8. Tokens and time come from the transcript, with the method stated

**Rule.** Cost is computed from the agent's own per-turn usage records, and the method is named next
to the number. Subscription runs report a *notional* dollar figure only. Time is reported as active
time, excluding idle gaps above a stated threshold.

**Gate.** `cost_time_toolcalls.yaml` records `usage_source`, `billing_mode` and `usage_complete`.

## 9. Say whether the seal is formal

**Rule.** A result is labelled `formal` only if the run reached its post-freeze public+hidden grade on
its own. An operator seal of an incomplete run (`formal_complete: false`) is reported as such, and a
run that died before sealing is reported as **in-run only**.

**Gate.** `completion.formal_grade_complete` and the seal reason on `run_manifest.yaml`.

## 10. Prompts are part of the treatment

**Rule.** A prompt edit is a treatment change. It is replayed offline against a recorded session
before it is spent, and it lands for all arms at once.

**Gate.** The prompt text in `codex_agent.py` and the per-driver agents; `test_continuous_is_the_default.py`
for the run shape.

## 11. Run lifecycle is part of the experiment

**Rule.** Runs launch inside a long-lived terminal multiplexer, never from a session that can
restart. Temporary files go on the large filesystem. Disk and memory are watched with a threshold
that pauses new work. Every run that ends is classified: sealed, operator-sealed, or died (with the
recorded reason).

**Gate.** Partly process. The Codex driver resumes the same session after a transient
model-capacity refusal, a bounded number of times inside the wall budget and never on another
model, and records `capacity_retries` and `unrecovered_errors`, so a recovered turn is not read as
a death and an unrecovered one is not hidden.

## 12. Paper numbers come from one generated ledger

**Rule.** Every number that appears in a paper, a figure, or a status update is generated from sealed
artifacts by one script, which records the source path of each number. Nothing is copied by hand
between documents.

**Gate.** Not yet built. This is the cheapest rule with the largest effect on a submission.

## What these rules do not cover

These rules make a single measurement honest. They do not replace **repetition**: one run per arm
cannot separate a treatment effect from run-to-run variance. Repeated matched runs remain the requirement
for a causal claim about authoring treatments.
