# AGENT.md — artifacts

## Purpose

Gitignored root for ALL generated products that are not aet runs. Organized **concern-first**:
each tool/concern owns a subtree and uses ITS natural axis (target, workload, model, framework,
or cross-cutting). See CLAUDE.md "Generated-output convention" and .claude/skills/artifact-layout.

## Concerns

- `dse-guidance/`, `dse/`, `design-pressure/` — the three DSE tools (axis: workload/model, NOT target)
- `kernel-mining/<target>/` — rvvgen mining/autotune (axis: target backend)
- `kernel-index/<framework>/` — source-framework kernel indexing
- `ceiling/`, `compare/` — cross-framework / cross-product comparisons
- `measurements/<model>/` — K1 board measurement campaigns
- `recaptures/<model>_<dtype>/` — model captures (PURGEABLE, 130 GB)
- `perf-bench/<target>/`, `capsule-bench/<target>/` — experiment report figures
- `presentation/`, `cache/`, `selfcheck/`

## Invariants

- Contents gitignored; only AGENT.md / README.md / .gitkeep tracked.
- Versioned PRODUCTS carry `<name>_v<ver>_<TS>_<sha7>/` + manifest.yaml; analysis/plots/dumps keep descriptive names.
