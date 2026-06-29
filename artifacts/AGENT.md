# AGENT.md — artifacts

## Purpose

Gitignored: ALL generated products that are not aet runs — versioned products, caches, plots, presentations, mined knowledge, dse analysis, selfcheck, recaptures. One of the three sanctioned output roots (with runs/ and build/).

## What belongs here

- Files appropriate to the purpose above, written via `merlin.common.artifacts` (never by hand-constructed paths).

## What does not belong here

- Hand-authored source or schemas.
- Anything that should be tracked in git (contents are gitignored).

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Never commit generated artifacts here.
- Sub-namespaces: products at `<topic>/v<ver>/<topic>_v<ver>_<TS>_<sha7>/` + manifest.yaml; caches under cache/; the 130 GB model recaptures under recaptures/ (PURGEABLE).
