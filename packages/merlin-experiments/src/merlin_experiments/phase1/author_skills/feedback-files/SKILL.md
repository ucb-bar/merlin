---
name: feedback-files
description: "Which grading feedback file answers which question (newest result per test case, the live verdict, failure clusters, a running grade, certificates by bytes, self-check responses), how fresh each one is, what tiers and certification mean, and when to look again. Use before reading anything under qa/ or selfcheck_out/, after a grade lands, or when two results seem to disagree."
---

# Feedback files

Read the smallest file that answers the question. The large JSON files are for scripted queries
(load with `json`, print only the keys you need), never whole reads. `SKILL_DIR` is this file's
directory (`.agents/skills/feedback-files` or `.claude/skills/feedback-files`).

`python SKILL_DIR/scripts/qa_status.py` prints every `qa/` file's age and size and the fields each
carries about itself: grade time, `snapshot_current`, graded and current source digests, counts.

## Which file for which question

- **Which cases pass, which fail, how failures group:** `qa/latest_grade_summary.txt`, one line per
  test case (status, tiers, failure plane/category, the grade a row came from when not the newest)
  and every failure cluster with members and reproduce command. Read it first.
- **One case's full row** (detail, tiers, mismatch count, trace results): `per_capsule` in
  `qa/latest_completed_grade.json`. Rows merge across grades: a case a newer grade did not run keeps
  its last result, and its `graded` names the grade and source digest it describes. Top-level counts
  are the newest grade's only; `completed_rows` counts the merged rows (`from_current_source` says
  how many describe your current bytes).
- **Is the grade of my current bytes in:** `qa/verdict.json`. It is written only for a snapshot that
  is still current when the grade finishes, so while you keep editing it can lag far behind the
  completed grade. Compare its `graded_source_sha256` with `current_source_sha256`.
- **What to fix next, as groups:** `failure_clusters` in both files above; open `fix-failure-cluster`.
- **How far the running grade is:** `qa/progress.json` (`state`, `n_completed`, rows as they finish).
  Interim only; it never converges anything.
- **Which bytes passed which tier:** `qa/tier_state.json`, the certificate ledger (megabytes: query
  one case's entry).
- **My own check:** the self-check response (`python agent_selfcheck.py --attach ID`; `--list` shows
  your requests); its artifacts are in `selfcheck_out/NAME/`. It describes the bytes you submitted.
- **Harness notes for this round:** `qa/round_brief.md`, rebuilt at each round or launch.
- **Workflow evidence:** `qa/workflow_conformance.json`, when present.
- **Performance** (`feedback_brief`, each response's `performance_brief`, roofline and utilization,
  timing model against gSIM, idle-time timing and tuning, `submission/TUNING_KNOBS.json`, fences
  per tile `fence_bound`, switches `tuning_knob_candidates`): open
  `performance-feedback-levels`, `roofline-and-utilization` and `cycle-model-and-gsim`.

## Tiers and certification

- Each row's `tiers` maps a tier label to its status; TASK.md says which engine runs each tier and
  which tiers a case requires. `pending`, `unavailable`, `skipped` and not-run are not passes, and
  `all_pass` stays null while a mandatory tier is outstanding.
- A case is certified when every tier it declares mandatory passed for the same bytes. A development
  self-check on a cheaper engine can refute a case but does not certify an unrun tier.
- A certificate belongs to the case's executable bytes (its emitted program), not to your source. A
  compiler edit that leaves a case's emitted program byte-identical keeps its certificate; an edit
  that changes it invalidates the earlier certifying-tier (e.g. L3) result, and that tier runs again
  on the new bytes after the loop tier passes. A broad edit late in a run therefore re-buys the
  certifying tier for every case whose program changed.

## When to look

1. After each completed grade: `python await_verdict.py --verdict qa/latest_completed_grade.json`
   blocks until one lands (plain `await_verdict.py` waits on `qa/verdict.json`, which only moves
   for a current snapshot). Then read the summary.
2. Before choosing the next work, and before freezing or reporting: summary, then the verdict's
   digest check.
3. After a context compaction or when briefing a helper: the summary, not the full JSON.
4. Do not poll in a loop: a grade takes minutes, and every look is a model round trip. Use the
   blocking waits (`await_verdict.py`, `agent_selfcheck.py --attach ID`) and keep working meanwhile.
