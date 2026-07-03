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
output/        gitignored produced artifacts (structured)
tmp/           gitignored local scratch (cross-session notes in tmp/help/)
```

Every directory contains an `AGENT.md`. `build/`, `output/`, and `tmp/` are gitignored; only
their `AGENT.md` / `README.md` / `.gitkeep` are tracked.
