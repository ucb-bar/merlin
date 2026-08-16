# CLI reference

_Generated from `pyproject.toml [project.scripts]` by `build_tools/scripts/gen_cli_docs.py` — do not edit by hand; run the generator._

These console-scripts are installed by `pip install -e merlin/python`. Each is a thin entrypoint over a module in the `merlin` package (no separate `tools/` layer). Run any with `--help`.

| Command | Backing module |
|---|---|
| `kernel-audit` | `merlin.kernels.audit:main` |
| `kernel-bench` | `merlin.kernels.bench_ceiling:main` |
| `kernel-extract` | `merlin.kernels.cli_extract:main` |
| `kernel-index` | `merlin.kernels.cli_index:main` |
| `merlin-cca-route` | `merlin.rvvgen.route_report:main` |
| `merlin-compare` | `merlin.compare.cli:main` |
| `merlin-compile` | `merlin.compile_cli:main` |
| `merlin-design-pressure` | `merlin.design_pressure.cli:main` |
| `merlin-dse` | `merlin.dse.cli:main` |
| `merlin-dse-guidance` | `merlin.dse_guidance.cli:main` |
| `merlin-liveness` | `merlin.liveness.cli:main` |
| `merlin-onboard` | `merlin.targetgen.onboard:main` |
| `merlin-rvv-autotune` | `merlin.rvvgen.autotune:main` |
| `merlin-rvv-beam` | `merlin.rvvgen.beam_cli:main` |
| `merlin-rvv-mine` | `merlin.rvvgen.mine:main` |
| `merlin-rvv-opt` | `merlin.rvvgen.op_sweep:main` |
| `merlin-rvv-report` | `merlin.rvvgen.report:main` |
| `merlin-target-fetch` | `merlin.targetgen.oot_fetch:main` |
| `merlin-target-publish` | `merlin.targetgen.publish:main` |
| `merlin-targetgen` | `merlin.targetgen.cli:main` |
