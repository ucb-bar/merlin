# CLI reference

_Generated from `pyproject.toml [project.scripts]` by `build_tools/scripts/gen_cli_docs.py` — do not edit by hand; run the generator._

These console-scripts are installed with the package (`uv sync --all-extras`, or `pip install -e .` from the repo root -- `merlin/python` is the package DIR, not an installable project). Each is a thin entrypoint over a module in the `merlin` package (no separate `tools/` layer). Run any with `--help`.

| Command | Backing module |
|---|---|
| `kernel-audit` | `merlin.kernels.audit:main` |
| `kernel-bench` | `merlin.kernels.bench_ceiling:main` |
| `kernel-extract` | `merlin.kernels.cli_extract:main` |
| `kernel-index` | `merlin.kernels.cli_index:main` |
| `merlin-asm-audit` | `merlin.kernels.cli_asm:main` |
| `merlin-bundle-pretranspose` | `merlin.baselines.pretranspose_cli:main` |
| `merlin-cca-route` | `merlin.mining.route_report:main` |
| `merlin-compare` | `merlin.compare.cli:main` |
| `merlin-compile` | `merlin.compile_cli:main` |
| `merlin-compile-kernel` | `merlin.triton.cli:main` |
| `merlin-design-pressure` | `merlin.design_pressure.cli:main` |
| `merlin-dse` | `merlin.dse.cli:main` |
| `merlin-dse-guidance` | `merlin.dse_guidance.cli:main` |
| `merlin-kernel-autotune` | `merlin.mining.autotune:main` |
| `merlin-kernel-beam` | `merlin.mining.beam_cli:main` |
| `merlin-kernel-mine` | `merlin.mining.mine:main` |
| `merlin-kernel-opt` | `merlin.mining.op_sweep:main` |
| `merlin-kernel-report` | `merlin.mining.report:main` |
| `merlin-lit-suite` | `merlin.targetgen.lit_suite:main` |
| `merlin-liveness` | `merlin.liveness.cli:main` |
| `merlin-onboard` | `merlin.targetgen.onboard:main` |
| `merlin-opt` | `merlin.xdsl_dialects.opt:main` |
| `merlin-rvv-autotune` | `merlin.mining.autotune:main` |
| `merlin-rvv-beam` | `merlin.mining.beam_cli:main` |
| `merlin-rvv-mine` | `merlin.mining.mine:main` |
| `merlin-rvv-opt` | `merlin.mining.op_sweep:main` |
| `merlin-rvv-report` | `merlin.mining.report:main` |
| `merlin-surface` | `merlin.kernels.cli_surface:main` |
| `merlin-target-fetch` | `merlin.targetgen.oot_fetch:main` |
| `merlin-target-publish` | `merlin.targetgen.publish:main` |
| `merlin-targetgen` | `merlin.targetgen.cli:main` |
| `merlin-verify` | `merlin.verify.cli:main` |
