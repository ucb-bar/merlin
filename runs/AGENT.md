# AGENT.md — runs

## Purpose

Gitignored: aet-managed experiment runs, one per directory at `runs/<target>/<suite>/<run-id>/` (logs/ metrics/ artifacts/ generated/ contracts/ + run_record.json). Target is the top folder level so a target's runs group together; inner file names are shared across targets for diffing. Created ONLY via `merlin.common.artifacts.start_run(..., target=...)`.

## What belongs here

- Files appropriate to the purpose above, written via `merlin.common.artifacts` (never by hand-constructed paths).

## What does not belong here

- Hand-authored source or schemas.
- Anything that should be tracked in git (contents are gitignored).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Never commit generated artifacts here.
- Run-id = `<UTC-TS>_<method>_seed<NNN>_<sha7>` (timestamp-first; chronological sort). Query with `aet runs`.
