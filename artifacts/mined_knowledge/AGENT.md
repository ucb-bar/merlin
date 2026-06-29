# AGENT.md — artifacts/mined_knowledge

## Purpose

Gitignored: versioned kernel-mining products + manifest.yaml (was /mined_knowledge), one per `<target>/<run-id>/`.

## What belongs here

- Files appropriate to the purpose above, written via `merlin.common.artifacts` (never by hand-constructed paths).

## What does not belong here

- Hand-authored source or schemas.
- Anything that should be tracked in git (contents are gitignored).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Never commit generated artifacts here.
- Each product dir carries a manifest.yaml (run_id/timestamp/git_sha/version/artifacts/sources).
