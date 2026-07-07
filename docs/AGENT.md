# AGENT.md — docs

## Purpose

Durable, cross-linked project documentation. Start at the generated hub `docs/README.md`.
See the `docs-layout` skill for the full convention.

## Structure (kind = subdir)

- `reference/` — durable, code-derived facts (architecture, dialects, runtime; generated
  `cli.md` / `module_index.md` / `schemas.md`).
- `guides/` — task-oriented how-tos (`getting_started`, dse, kernel_mining, targetgen, …).
- `design/` — rationale / design notes / methodology.
- `README.md` — the **generated** hub (by-kind + by-area + cross-links). Do not hand-edit.

## What belongs here

- Only **durable** reference/guides/design. Each carries YAML front-matter
  (`title, kind, status, owner, last_verified, related, code_refs`).

## What does NOT belong here

- **Point-in-time reports** (results, findings, status, presentations) → `artifacts/` (concern-first),
  never `docs/`. See the `artifact-layout` skill.
- Generated outputs, code, unrelated artifacts.

## Invariants

- After adding/editing a doc: run `build_tools/scripts/gen_docs_index.py` then `check_docs.py`.
- Generated docs (`cli.md`/`module_index.md`/`schemas.md`) are regenerated, never hand-edited.
- Front-matter must be schema-valid; `last_verified` is a re-verification claim (see `docs-doctor`).
- Every subdirectory must also contain an AGENT.md.
