# ADR 0001: Repository Maintenance Model

- Status: Accepted
- Date: 2026-03-02

## Context

Merlin is maintained by a very small team while carrying out-of-tree changes
on top of IREE and nested LLVM forks. Maintenance cost and upstream drift are
key risks.

## Decision

1. Keep upstream forks pinned and patch-driven:
   - Pins: `patches/manifest.*`
   - Ordered patch series: `patches/series.*`
2. Prefer out-of-tree Merlin logic in this repository.
3. Use one stable developer entrypoint:
   - `tools/merlin.py`
4. Add CI gates for:
   - script lint/syntax
   - patch verify
   - drift check
5. Keep board flows in `benchmark/target/<board>/` with deploy/run/parser
   scripts and environment templates.

## Consequences

Positive:

1. Lower maintenance overhead for 1-2 maintainers.
2. Faster detection of upstream drift and accidental fork edits.
3. Clearer onboarding via single command interface and process docs.

Trade-offs:

1. Requires disciplined patch refresh/update when in-tree fork edits happen.
2. Some workflows remain script wrappers around existing build systems.
