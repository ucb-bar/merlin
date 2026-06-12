# AGENT.md — merlin/targets/saturn/docs

## Purpose

Saturn reference-target docs (runtime adapter behavior, backend mapping).

## What belongs here

- Markdown describing how saturn adapts the Merlin runtime (spike/vcs/zephyr backends).

## What does not belong here

- Plans (those are `../contracts/`), code, or vendored upstream docs.

## Interfaces

Read by humans and by TargetGen evidence ingestion when pointed at this directory.

## Invariants

- Keep claims scoped to the spike model unless verified on Saturn RTL.

## Testing expectations

None directly; keep consistent with `../contracts/`.
