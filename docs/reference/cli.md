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
| `merlin-cpu-host-beam-figures` | `merlin.plotting.cpu_host_beam_figures:main` |
| `merlin-cpu-host-figures` | `merlin.plotting.cpu_host_experiment_figures:main` |
| `merlin-design-pressure` | `merlin.design_pressure.cli:main` |
| `merlin-dse` | `merlin.dse.cli:main` |
| `merlin-dse-guidance` | `merlin.dse_guidance.cli:main` |
| `merlin-kernel-autotune` | `merlin.mining.autotune:main` |
| `merlin-kernel-beam` | `merlin.mining.beam_cli:main` |
| `merlin-kernel-mine` | `merlin.mining.mine:main` |
| `merlin-kernel-opt` | `merlin.mining.op_sweep:main` |
| `merlin-kernel-report` | `merlin.mining.report:main` |
| `merlin-liveness` | `merlin.liveness.cli:main` |
| `merlin-onboard` | `merlin.targetgen.onboard:main` |
| `merlin-paper-capture` | `merlin.compare.capture_workflow:main` |
| `merlin-paper-executorch-packages` | `merlin.compare.executorch_packages:main` |
| `merlin-paper-figures` | `merlin.plotting.rvv_paper_figures:main` |
| `merlin-paper-k1-matrix` | `merlin.compare.paper_k1_orchestrator:main` |
| `merlin-paper-merlin-packages` | `merlin.compare.paper_merlin_packages:main` |
| `merlin-paper-merlin-producers` | `merlin.compare.paper_merlin_producers:main` |
| `merlin-rvv-autotune` | `merlin.mining.autotune:main` |
| `merlin-rvv-beam` | `merlin.mining.beam_cli:main` |
| `merlin-rvv-mine` | `merlin.mining.mine:main` |
| `merlin-rvv-opt` | `merlin.mining.op_sweep:main` |
| `merlin-rvv-report` | `merlin.mining.report:main` |
| `merlin-surface` | `merlin.kernels.cli_surface:main` |
| `merlin-target-fetch` | `merlin.targetgen.oot_fetch:main` |
| `merlin-target-publish` | `merlin.targetgen.publish:main` |
| `merlin-targetgen` | `merlin.targetgen.cli:main` |
