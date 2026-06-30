# Repository structure

XLA-style: a small root, almost everything under the internal `merlin/` tree.

```
build_tools/   build/dev tooling + measurement/analysis/sweep runners + repo linters (scripts/, cmake, docker)
docs/          this documentation (incl. generated cli.md = CLI surface)
third_party/   hard build/test deps only (no analysis repos)
runs/          aet experiment runs (gitignored)   ── see CLAUDE.md "Generated-output convention"
artifacts/     all other generated products (gitignored)
merlin/
  compiler/    stable MLIR/C++ plane (scaffold)
  python/      Python plane (xDSL prototyping + workstream pipelines)
  runtime/     target-independent runtime substrate
  integrations/adapters to external projects (not vendored)
  targets/     toy/reference targets (toy_npu, example_vector)
  schemas/     cross-workstream coordination contract
  benchmarks/  shared workload descriptions
  experiments/ throwaway/bookkeeping experiments
  tests/       integration/conformance/golden/data
build/         gitignored generated build outputs (structured)
output/        gitignored produced artifacts (structured)
tmp/           gitignored local scratch (cross-session notes in tmp/help/)
```

Every directory contains an `AGENT.md`. `build/`, `output/`, and `tmp/` are gitignored; only
their `AGENT.md` / `README.md` / `.gitkeep` are tracked.
