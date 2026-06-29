# AGENT.md — artifacts/cache

## Purpose

Gitignored: large regenerable caches (kernel caches, intermediate compute). PURGEABLE — safe to delete; regenerated on demand. Created via `merlin.common.artifacts.cache_dir(namespace)`.

## What belongs here

- Files appropriate to the purpose above, written via `merlin.common.artifacts` (never by hand-constructed paths).

## What does not belong here

- Hand-authored source or schemas.
- Anything that should be tracked in git (contents are gitignored).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Never commit generated artifacts here.
- Never rely on cache contents for reproducibility; the manifest/source is the source of truth.
