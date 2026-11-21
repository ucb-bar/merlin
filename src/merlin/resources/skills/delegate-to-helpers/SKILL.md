---
name: delegate-to-helpers
description: "Hand well-scoped work to helper agents: write a self-contained brief (goal, owned files, test cases with reproduce command, skills to open, what to report), choose a fresh-context helper or a full-history fork, and run several helpers in parallel on independent failure clusters or modules while you integrate. Use when the work splits into parts that do not need your whole history, or when helper capacity sits idle."
---

# Delegate to helpers

Helpers share your workspace, instructions, skills and tools, and see each other's edits at once.
Your instructions state how many can run at the same time. Delegate when independent work and file ownership are clear; an idle slot alone does not establish
useful parallel work.

## Fresh helper or full-history fork

- **Fresh context** (Codex: `spawn_agent` with `fork_turns "none"`; Claude Code: an Agent/Task
  subagent, which always starts fresh): starts from the instructions, the skills and your brief only.
  Its context begins with the brief rather than your whole history; usage and latency still depend
  on its subsequent work. Suits one failure cluster, one module, one ISA or ABI question, one variant.
- **Full-history fork** (Codex: `fork_turns` omitted or `"all"`): re-sends your whole conversation to
  the helper with every step. Worth it when the task depends on reasoning you have not written down;
  a notes file can make that reasoning available to a fresh helper.

## A self-contained brief

A fresh helper knows only what you write. Include:

1. **Goal:** the observable outcome (these cases pass; this module handles that operation; this
   question answered with evidence).
2. **Owned files:** exact paths it may edit; no other agent edits them meanwhile. For an experiment,
   a copy instead: `cp -r submission variants/NAME`, checked with `--submission variants/NAME`.
3. **Cases and commands:** test-case names, the reproduce command and one check command for all of
   them. For a failure cluster, `python .agents/skills/fix-failure-cluster/scripts/clusters.py
   --brief N` (or the `.claude/skills` copy) drafts this part.
4. **Skills to open, by name:** e.g. `fix-failure-cluster`, `feedback-files`, the task-* section the
   work depends on. Each skill is also a readable file, `.agents/skills/NAME/SKILL.md` or
   `.claude/skills/NAME/SKILL.md`.
5. **Constraints:** interfaces other parts rely on; TASK.md's rules stay in force.
6. **Report:** root cause, files changed, self-check request ids with pass counts for its cases and a
   regression sample, what is still open. Ask for it short, so your own context stays small.

## Several helpers in parallel

1. **Partition by files, not by cases.** List the clusters
   (`python .agents/skills/fix-failure-cluster/scripts/clusters.py`) or modules, note which files each
   fix touches, and give each helper a disjoint set. Parts that need the same file go to one helper,
   or run one after another.
2. **Spawn, then keep working** on your own part. Wait with the driver's blocking wait (Codex:
   `wait_agent` with a long timeout) rather than polling.
3. **Each helper checks only its cases.** Self-check requests from several agents run side by side;
   one request runs its cases in parallel, so a helper names all its cases in one request.
4. **Integrate yourself.** Merge results into `submission/` (from the owned files or a
   `variants/NAME/` copy), then check touched cases and a relevant regression sample on that immutable snapshot. Unfinished
   unrelated helper work does not gate it.
   Send a follow-up to the helper that did the work (Codex: `followup_task`) instead of redoing it.
5. **Refill the slot.** When a helper reports, the next independent part (another cluster, a variant
   comparison, a performance experiment on one case family, a scale or tail check) can take its place.

## Choosing parts

Parts that delegate well: one failure cluster; one operation family's lowering; a variant to compare
with `submission/`; an ISA or ABI question answered with the workspace debugger; a performance
experiment on one case family. Parts that do not: anything that needs a file another agent is
editing; a brief without a checkable outcome.
