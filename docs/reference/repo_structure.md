---
title: Repository structure
kind: reference
status: current
owner: core
last_verified: 2026-07-07
related: [architecture]
code_refs: [.]
---

# Repository structure

XLA-style: a small root, almost everything under the internal `merlin/` tree.

```
build_tools/   build/dev tooling + measurement/analysis/sweep runners + repo linters (scripts/, cmake, docker)
docs/          this documentation (incl. generated cli.md = CLI surface)
third_party/   hard build/test deps only (no analysis repos)
runs/          aet experiment runs (gitignored)   ── see CLAUDE.md "Generated-output convention"
artifacts/     all other generated products (gitignored); artifacts/targets/ = codegen packages
               (replaces retired generated_targets/; tracked baselines/champions via .gitignore negations)
merlin/
  python/      Python + xDSL compiler plane + workstream packages (the active compiler)
  runtime/     target-independent C runtime substrate (c/ + baremetal/ + abi/)
  targets/     reference targets (toy_npu, saturn, gemmini)
  schemas/     cross-workstream coordination contract
  benchmarks/  shared workload descriptions
  experiments/ workstream experiments + benchmark harnesses (agent/capsule/perf-bench, targetgen_evals)
  tests/       integration/conformance/golden/data
build/         gitignored generated build outputs (structured)
output/        DEPRECATED — holds only the regenerable model recaptures (via recaptures_dir());
               never write new generated content here (folded into artifacts/; the guard hook blocks it)
tmp/           gitignored local scratch (cross-session notes in tmp/help/)
```

The three generated-output roots are `runs/`, `artifacts/`, and `build/` (see CLAUDE.md
"Generated-output convention"). Every directory contains an `AGENT.md`. `build/`, `output/`, and
`tmp/` are gitignored; only their `AGENT.md` / `README.md` / `.gitkeep` are tracked.
