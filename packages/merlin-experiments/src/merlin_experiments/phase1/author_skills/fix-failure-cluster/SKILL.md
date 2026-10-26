---
name: fix-failure-cluster
description: "Work through the grader's failure clusters: list them with reproduce commands, pick one, reproduce it, locate the failing compiler stage from its plane, category and detail, fix it, re-check the cluster's cases and confirm in the next grade; when and how to hand a cluster to a helper. Use when a grade or qa/latest_grade_summary.txt shows failing test cases."
---

# Fix a failure cluster

A cluster is a group of failing test cases with one redacted failure (plane, category, detail with
numbers, paths and the case name normalized). One fix often clears all of it. The grader publishes
`failure_clusters` in `qa/verdict.json` (only while that verdict is current) and in
`qa/latest_completed_grade.json` (newest result per case); `qa/latest_grade_summary.txt` lists them
as text. Cases that did not run (gated, deferred, incomplete) are never clustered; their row says why.

`SKILL_DIR` below is this file's directory: `.agents/skills/fix-failure-cluster` or
`.claude/skills/fix-failure-cluster` (identical copies).

1. **List.** `python SKILL_DIR/scripts/clusters.py`. Per cluster: size, `plane/category`, how many
   members were graded on the current source, one example detail, the `reproduce` command, an
   `inspect` command when a command buffer exists, and the members. `--cluster N` prints one in full
   (each member's status, tiers, mismatch count, grade time) plus an all-members command.
2. **Pick.** Clusters are ordered by size, not priority. Weigh: cases cleared per fix; whether the
   plane is early in the pipeline (a build or emission failure hides everything after it, so a fix may
   expose a later failure in the same cases); whether members were graded on your current source (if
   not, re-check first, it may be fixed already); independence from files you or helpers are editing.
3. **Reproduce.** Run the `reproduce` command: an ordinary self-check of one example on the engine
   where it failed. `--wait-budget 5` returns a request id at once; collect it with
   `python agent_selfcheck.py --attach ID`. Confirm the same plane/category. If it passes, the cluster
   is stale: run the all-members command.
4. **Locate the stage.** The plane names the harness stage that rejected the case, the category the
   check, the detail that stage's redacted message. Map it onto your pipeline:
   - build, link or ABI messages name an entrypoint, file or operation: go to the frontend, lowering
     or emission code that produced it. `python agent_selfcheck.py --compile-only --capsules NAME
     --timeout 120` (when `--help` lists it) isolates build-side failures quickly;
   - execution, numeric or trace planes (mismatch count, trace violations): the emitted program is
     wrong. Run the `inspect` command and compare what the program does with what the operation
     needs; open `read-command-buffer` if it is listed;
   - compare 2-3 members (`--cluster N`): what they share (an operation, a shape property, a tail,
     an epilogue) usually is the cause.
5. **Fix** in the compiler, generally: derive from the input IR's shapes, types and layouts, never
   from a test-case name. Unsure between two fixes? Put one in `variants/NAME/` (same layout as
   `submission/`) and check both, the variant with `--submission variants/NAME`.
6. **Re-check** all members in one request (the all-members command) plus a few passing cases that
   use the edited code: one request runs its cases in parallel. `--capsules failing` repeats the last
   check's failures. An engine that takes one case per request says so in its response.
7. **Confirm in the grade.** `python await_verdict.py --verdict qa/latest_completed_grade.json` blocks
   until the next completed grade; rerun `clusters.py`. The cluster should shrink or vanish with its
   members graded on source that contains the fix. A grade of older bytes still shows it: compare
   `graded_source` with `current_source` in the header. `feedback-files` explains each file's freshness.

## Hand a cluster to a helper

A cluster is a self-contained unit of work. `python SKILL_DIR/scripts/clusters.py --brief N` drafts
a brief (goal, failure, cases, reproduce and all-members commands, report format); add the files the
helper owns and where it works. The brief carries everything, so a helper with a fresh context
(Codex: `fork_turns "none"`) needs none of your history. Clusters whose fixes touch different files
can go to different helpers at the same time; two clusters that need the same file belong to one
owner. Open `delegate-to-helpers` for writing briefs and running helpers in parallel.
