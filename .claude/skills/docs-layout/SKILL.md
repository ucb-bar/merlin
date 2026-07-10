---
name: docs-layout
description: >-
  Where documentation lives in merlin and how to keep it fresh — the docs/ tree, front-matter,
  the generated hub, and where reports go. Use whenever you add, edit, move, or reorganize a doc,
  write a new *.md, add a CLI/schema/package, or wonder whether something belongs in docs/ or
  artifacts/. Durable docs only: reference/ guides/ design/, each with front-matter.
---

# Documentation layout (MANDATORY convention)

Documentation splits into three surfaces. Put each thing in the right one:

| Surface | What | Where |
|---|---|---|
| **Central docs** | Durable reference/guides/design for the whole repo | `docs/reference/` · `docs/guides/` · `docs/design/` |
| **Per-directory** | Local purpose/invariants of one dir/package | that dir's `AGENT.md` (package ones auto-generated) |
| **Point-in-time reports** | Results, findings, status, presentations | `artifacts/` (NOT `docs/`) — see the `artifact-layout` skill |

`docs/README.md` is the **generated hub** — do not hand-edit it; run the generator.

## The three doc kinds (subdir = kind)

- `docs/reference/` — durable, code-derived facts (architecture, dialects, runtime, generated
  `cli.md`/`module_index.md`/`schemas.md`).
- `docs/guides/` — task-oriented how-tos (getting_started, dse, kernel_mining, targetgen, …).
- `docs/design/` — rationale / design notes / methodology.

If it snapshots a run (numbers, a specific experiment's outcome, a presentation), it is a **report**
— it goes to the matching `artifacts/` concern (`kernel-mining/`, `compare/`, `presentation/`, …),
kept tracked via an explicit `.gitignore` negation if it's hand-written. It does **not** go in `docs/`.

## Front-matter (required on every durable doc)

```yaml
---
title: <human title>
kind: reference | guide | design
status: current | draft | superseded
owner: <area, e.g. dse | kernels | targetgen | runtime | core>
last_verified: YYYY-MM-DD          # a claim someone reconciled this doc vs the code that day
related: [<slug>, ...]             # other doc stems → cross-links in the hub
code_refs: [<paths the doc describes>]   # → powers the drift detector
---
```

`slug` = the file stem (e.g. `architecture`). `related` and `code_refs` are what make docs navigable
and self-freshening — fill them in.

## After you touch docs — regenerate + check

```bash
.venv/bin/python build_tools/scripts/gen_docs_index.py      # rebuild docs/README.md hub
.venv/bin/python build_tools/scripts/check_docs.py          # cli/package/schema/index/paths/freshness
```

Generated docs have their own source of truth — regenerate, never hand-edit:
`gen_cli_docs.py` (cli.md ← pyproject), `gen_package_docs.py` (module_index.md ← docstrings),
`gen_schema_docs.py` (schemas.md ← merlin/schemas/).

## Enforcement (don't bypass)

- **Stop hook** + optional **pre-commit** (`build_tools/scripts/install_git_hooks.py`) + **CI** run
  `check_docs.py`: stale generated docs, invalid front-matter, retired paths, and scaffold-era
  phrasing in the root README/AGENT.md all fail the gate.
- **Semantic drift** (a doc whose `code_refs` changed after `last_verified`) is surfaced by
  `check_docs_freshness.py --json` and fixed by the **`docs-doctor`** skill — reconcile, then bump
  `last_verified`. Never bump the date without actually re-verifying against the code.

Shared working tree: never switch branches; commit small verified batches (see `CLAUDE.md`).
