---
title: Repository structure
kind: reference
status: current
owner: core
last_verified: 2026-07-14
related: [architecture]
code_refs: [.]
---

# Repository structure

XLA-style: a small root, almost everything under the internal `merlin/` tree.

```
build_tools/   build/dev tooling + measurement/analysis/sweep runners + repo linters (scripts/, cmake, docker)
docs/          this documentation (incl. generated cli.md = CLI surface)
third_party/   hard build/test deps only (no analysis repos)
out/           the SINGLE generated-output root (gitignored)   ── see CLAUDE.md "Generated-output convention"
  runs/        aet experiment runs
  artifacts/   all other generated products; artifacts/targets/ = codegen packages
               (replaces retired generated_targets/; tracked baselines/champions via .gitignore negations)
  build/       generated build outputs + buildable OOT codegen repos (build/generated/)
merlin/
  python/      Python + xDSL compiler plane + workstream packages (the active compiler)
  runtime/     target-independent C runtime substrate (c/ + baremetal/ + abi/)
  targets/     reference targets (toy_npu, saturn, gemmini)
  schemas/     cross-workstream coordination contract
  benchmarks/  shared workload descriptions
  experiments/ workstream experiments + benchmark harnesses (agent/capsule/perf-bench, targetgen_evals)
  tests/       integration/conformance/golden/data
tmp/           gitignored local scratch (cross-session notes in tmp/help/)
```

All generated output lives under the single `out/` root, with exactly three subdirs — `out/runs/`,
`out/artifacts/`, and `out/build/` (see CLAUDE.md "Generated-output convention"). The old top-level
`runs/`/`artifacts/`/`build/` and the retired `output/` (model recaptures now live at
`out/artifacts/recaptures/` via `recaptures_dir()`) are gone; the guard hook blocks writes outside
`out/`. Every directory contains an `AGENT.md`; under `out/` only `AGENT.md` / `README.md` /
`.gitkeep` (plus curated `.gitignore` negations) are tracked.
