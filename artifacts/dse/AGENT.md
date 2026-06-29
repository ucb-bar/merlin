# AGENT.md — artifacts/dse

## Purpose

Gitignored: DSE guidance analysis + insight-mining runs (was results/*_dse_analysis, output/dse_guidance).

## What belongs here

- Files appropriate to the purpose above, written via `merlin.common.artifacts` (never by hand-constructed paths).

## What does not belong here

- Hand-authored source or schemas.
- Anything that should be tracked in git (contents are gitignored).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Never commit generated artifacts here.
- Regenerable via merlin-dse-guidance; emitted through new_product('dse', ...).
