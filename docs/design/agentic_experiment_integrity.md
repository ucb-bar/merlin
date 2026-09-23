---
title: "Design: rules for citable agentic-compiler experiments"
kind: design
status: current
owner: core
last_verified: 2026-09-23
related: [capsule_generation, capsule_phase_split, radiance_staged_evaluation]
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

A capsule-bench campaign asks one question: given the same target, budget and model, does a richer
tool arm let an agent build a correct compiler sooner, cheaper, or further? Answering it takes
several hundred agent-hours. Between August and September 2026 we found that most of what separated
arms in our headline numbers was the **harness, the grader, or the bookkeeping**, not the treatment.
Each such finding is a small story. This note keeps the rule each one taught, the failure that
motivated it, and the gate that now enforces it, so the next campaign starts from the rules rather
than rediscovering them.

The rules are ordered by how much they would have changed a published number.

Current functional feedback implementations live in `merlin_experiments.phase1.feedback`:
`qa` owns grading and redaction, `promotion` owns per-capsule certification, and `snapshots`
owns enqueue-time candidate copies. Native launchers call these owners; relocating them does
not reattribute historical experiment evidence or qualify a complete installed controller.
The newer implementation-source inventory in `merlin_experiments.phase1.source_inputs` binds
native and outer Phase-1 owners for freshly frozen runs; it does not retroactively establish
ownership for legacy freezes that recorded only a repository SHA and submission bytes.

## 1. Cite the tier, never the bare score

**Rule.** A pass count is quoted together with `tier_reached` and `pass_evidence`
(`rtl_backed`, `cheap_tier_only`). A pass only at the loop tier (L2) is not a certified result.

**Why.** One "20/20" headline had 1 of 20 capsules at L3. Late in the campaign, the arm with the
larger score (94 against 88) had exactly the **same** RTL-certified count (88); its six-capsule margin
was all `cheap_tier_only`. A tier *name* is not evidence either: one target's L3 is Verilator, another's
is an RTL-derived model, so `rtl_backed` counts what the oracle itself reported as derived from RTL.

**Gate.** `capsule_grade.py` writes `pass_evidence` on every score. `agg_agentic_results.py` reads
L3 evidence rather than the headline.

## 2. Compare arms on one cohort, pinned before launch

**Rule.** The capsule set is pinned in the descriptor (`grading.search_cohort`) before any arm
launches. Cross-arm numbers are computed on the intersection of the capsules each arm actually
measured. The descriptor of a live run is never edited.

**Why.** Statuses that are *not measured* (`gated`, `screened_only`, `budget_exhausted`, …) shrink an
arm's denominator, so 75/95 and 93/97 are not comparable until both are restated on the common rows.
Adding one capsule without bumping the descriptor cost one campaign 34 failed grades and zero
verdicts. Editing a live run's descriptor kills the run at its next grade, because the grader
refuses a cohort whose digest no longer matches.

**Gate.** `agg_agentic_results.py` (`common_cohort`, `--cohort`); the descriptor-digest check in the
grader; `n_not_measured` on the score.

## 3. Freeze the harness for the whole measurement window

**Rule.** A campaign runs against one harness revision. No grader plane, prompt, or library change
lands between the first and last arm. When a fix must land, **every** arm is re-graded under it with
`regrade_run_snapshot.py`, and the comparison is restated.

**Why.** Grading runs in-process: a live run never sees a harness fix unless it is resumed. But the
agent's submission imports library code lazily from the working tree at grade time, so a run *does*
see half-finished edits from other sessions. A refusal plane (`epilogue_applicability`) that landed
mid-batch made the later-graded arms look ten capsules worse, and turned an earlier 20/20 into 15/20
when that submission was re-frozen. A single sealed run whose in-run grade and official grade
disagree by several capsules is the signature of this rule being broken.

**Gate.** `freeze_run.py` records `repo_sha` and the submission digest. Enqueue-time promotion
snapshots bind candidate source bytes; the shared implementation inventory binds source ownership
for newly frozen Phase-1 runs. Neither makes a legacy campaign's in-process mid-batch working-tree
imports trustworthy. Such a campaign still needs a revision pin or a fresh freeze and regrade of
every arm before its comparison can be cited.

## 4. One stop rule for every arm, decided in advance

**Rule.** The rule that ends a run (plateau over N grades, wall budget, token budget) is written down
before launch and applied to every arm identically. Whole-run totals are reported only for arms
stopped by the same rule. Otherwise compare **cost to reach a milestone**, each arm on its own clock.

**Why.** One arm was stopped while its plateau detector reported no stall, and another ran on after
three flat grades. Their whole-run token and hour totals therefore measured the operator, not the
arms. The milestone view (time and tokens to reach K certified capsules) survives that, because it
does not depend on when a run was stopped.

**Gate.** The plateau detector (`plateau.json`) is operator-side only and is never shown to the agent.
Applying it uniformly is still a process rule.

## 5. A check must be able to fail

**Rule.** Every gate is mutation-proved: break the thing it exists to catch and confirm that it
fails. A skipped tier is not a pass (`not_run_is_not_pass`).

**Why.** Fifteen separate checks in this repo reported success while unable to fail. Examples: a
smoke test whose `n/a` counted as true; a contract verifier that checked 0 modules and exited cleanly;
a compiled gate whose NaN guard was deleted by `-ffast-math`; and a semantic check that ignored one
attribute and so verified a different, easier program. None was found by reading the code; all were
found by mutation.

**Gate.** Mutation tests next to each gate. Examples: the promotion-wiring tests, and the holdout
check that must fail when it is blind. See `merlin/tests/infra/`.

## 6. Prove the arm was served before blaming the model

**Rule.** Before a result is read as "the agent could not", confirm from **inside** the sandbox that
the arm's grants were visible, the feedback channel was healthy, and the conformance parser could see
the tool use.

**Why.** A harness defect was read as an agent defect more than ten times. In one case the sandbox's
mount order hid the ISA files, and the agent invented an ISA across 29 builds while readiness said
GO. In another, the conformance parser could not read composed shell commands and marked a compliant
arm non-conformant. In a third, the same model on the same task scored 0/20 in one driver and 15/20 in
another.

**Gate.** `readiness_check.py` (visibility checked from inside the sandbox); `conformance.py`;
feedback-channel health on every verdict.

## 7. Answer keys stay out of reach, and holdouts are real

**Rule.** Goldens and hidden capsules are untracked, excluded from the wheel, and masked by the
sandbox. Hidden sets are genuine holdouts along a declared axis, not renamed public capsules. Goldens
are never restored while a run is live.

**Why.** Answer keys were found in four places: tracked in the repository, copied into the wheel by
`setup.py`, restored mid-run, and present in pre-spend replays. The sandbox computes its mask list
once, at launch, so a golden restored mid-run is visible unmasked. Hidden sets that were renames
measured nothing. Real holdouts showed a target that generalises over K and N tiling but not M.

**Gate.** `check_no_answer_keys.py` (pre-commit and Stop hook); `verify_no_cheat.py`;
`check_provenance.py` for hardware revisions.

## 8. Tokens and time come from the transcript, with the method stated

**Rule.** Cost is computed from the agent's own per-turn usage records, and the method is named next
to the number. Subscription runs report a *notional* dollar figure only. Time is reported as active
time, excluding idle gaps above a stated threshold.

**Why.** One CLI booked a $0.10 round as $21.68. A redactor turned `rc=0` into `rc=#`. Tool-call
counts read 0 against a real 125. Two honest methods applied to the same run differed by about a third
(one sums the harness's round totals, the other reconstructs every session from rollouts, including
resumes). Neither is wrong, but a comparison must use one of them for every arm.

**Gate.** `cost_time_toolcalls.yaml` records `usage_source`, `billing_mode` and `usage_complete`.

## 9. Say whether the seal is formal

**Rule.** A result is labelled `formal` only if the run reached its post-freeze public+hidden grade on
its own. An operator seal of an incomplete run (`formal_complete: false`) is reported as such, and a
run that died before sealing is reported as **in-run only**.

**Why.** At the end of the campaign every sealed result was an operator seal. Several runs died
unsealed after their parent process disappeared (the agent CLI then panics on a broken pipe). Quoting
their best in-run grade as if it were official would have overstated three targets.

**Gate.** `completion.formal_grade_complete` and the seal reason on `run_manifest.yaml`.

## 10. Prompts are part of the treatment

**Rule.** A prompt edit is a treatment change. It is replayed offline against a recorded session
before it is spent, and it lands for all arms at once.

**Why.** One sentence meant to stop polling ("wait for the verdict") told agents to wait for a
verdict that could only exist after their first submission. Two runs spent their first 26 and 40
minutes waiting. Poll directives earlier cost 118 of 786 tool calls in one run.

**Gate.** The prompt text in `codex_agent.py` and the per-driver agents; `test_continuous_is_the_default.py`
for the run shape.

## 11. Run lifecycle is part of the experiment

**Rule.** Runs launch inside a long-lived terminal multiplexer, never from a session that can
restart. Temporary files go on the large filesystem. Disk and memory are watched with a threshold
that pauses new work. Every run that ends is classified: sealed, operator-sealed, or died (with the
recorded reason).

**Why.** Runs died when the launching session restarted, when the root filesystem filled with
abandoned temp directories, and when a console buffer grew to 72 GB. Each death looked, from the
headline, like a slow or stuck agent.

## 12. Paper numbers come from one generated ledger

**Rule.** Every number that appears in a paper, a figure, or a status update is generated from sealed
artifacts by one script, which records the source path of each number. Nothing is copied by hand
between documents.

**Why.** One baseline ratio appeared with three different values in three documents, and a headline
sentence carried two ratios with no artifact behind either.

**Gate.** Not yet built. This is the cheapest rule with the largest effect on a submission.

## What these rules do not cover

These rules make a single measurement honest. They do not replace **repetition**: one run per arm
cannot separate a treatment effect from run-to-run variance. We measured two runs of the same arm,
under identical treatment, landing 26 capsules apart. Repeated matched runs remain the requirement
for any causal claim about arms.
