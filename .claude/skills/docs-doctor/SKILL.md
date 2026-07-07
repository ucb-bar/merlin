---
name: docs-doctor
description: >-
  Refresh documentation that has fallen behind the code in oscar-merlin. Use when asked to
  audit docs for drift, update stale docs, re-verify documentation, run the docs-doctor, or on
  a schedule. Consumes the deterministic drift worklist and updates docs + last_verified only
  after verifying against the actual code change.
---

# docs-doctor — keep docs from falling behind the code

The linters (`check_docs.py`, wired into the Stop/pre-commit/CI gates) catch *mechanical* drift:
stale generated files, invalid front-matter, retired paths. They CANNOT catch *semantic* drift —
a guide that still parses fine but describes code that has since changed. That is this skill's job.

## The deterministic backbone (LLM proposes, gate disposes)

Every durable doc under `docs/` carries front-matter with `last_verified: YYYY-MM-DD` and
`code_refs: [<paths the doc describes>]`. The worklist is computed deterministically — you never
guess what's stale:

```bash
.venv/bin/python build_tools/scripts/check_docs_freshness.py --json
```

Returns `{"drift": [...], "uncategorized": [...]}`. Each `drift` entry is a doc whose
`last_verified` predates the newest git commit touching one of its `code_refs`:

```json
{"doc": "guides/dse.md", "last_verified": "2026-07-07",
 "stale_code_refs": [{"path": "merlin/python/merlin/dse", "last_commit": "2026-07-20"}]}
```

## The loop (per drift candidate)

1. **See what changed.** For each stale `code_ref`, read the diff since the doc was last verified:
   `git log --oneline --since=<last_verified> -- <path>` then `git diff <first>~1..HEAD -- <path>`.
2. **Read the doc** (`docs/<doc>`).
3. **Judge:**
   - *Doc still accurate* (change was internal/irrelevant) → only bump `last_verified` to today.
   - *Doc now wrong/incomplete* → edit the prose to match the code, THEN bump `last_verified`.
   - *Can't tell* → leave `last_verified` alone and flag it for a human; do NOT bump.
4. **Never bump `last_verified` without having actually reconciled the doc against the diff.** The
   date is a claim that a human/agent verified the doc against the code as of that day.
5. After edits, **regenerate the hub and re-check:**
   `.venv/bin/python build_tools/scripts/gen_docs_index.py && .venv/bin/python build_tools/scripts/check_docs.py`.
6. Commit in a small batch (shared tree — see `CLAUDE.md`): `docs(doctor): re-verify <doc> vs <ref>`.

Also handle `uncategorized` (docs with no front-matter): either add front-matter (if durable) or
relocate to `artifacts/` (if it's a point-in-time report) — see the `docs-layout` skill.

## Running it

- **On demand:** "run the docs-doctor" / "audit docs for drift".
- **Recurring:** `/loop 1d docs-doctor` (self-paced) or a scheduled cloud routine (`/schedule`) that
  runs the worklist and opens a PR with the proposed re-verifications. Keep batches small and never
  auto-bump `last_verified` without the reconciliation step above.

## Guardrails

- Shared working tree: never switch branches; commit small verified batches; never stage another
  session's uncommitted files.
- A doc that legitimately has no front-matter yet (e.g. work-in-progress) is soft — don't force it.
- The freshness date is only as honest as the reconciliation behind it. When unsure, escalate, don't bump.
