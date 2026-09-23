# CLI reference

_Generated from `pyproject.toml [project.scripts]` by `build_tools/scripts/gen_cli_docs.py` — do not edit by hand; run the generator._

Core console-scripts are installed with `pip install -e .` from the repo root. Optional research distributions under `packages/` are installed separately; their commands are listed below when declared. `src/merlin` is source, not an installable project. Each command is a thin module entrypoint. Run any with `--help`.

| Command | Backing module |
|---|---|
| `kernel-audit` | `merlin.kernels.audit:main` |
| `kernel-bench` | `merlin.kernels.bench_ceiling:main` |
| `kernel-extract` | `merlin.kernels.cli_extract:main` |
| `kernel-index` | `merlin.kernels.cli_index:main` |
| `merlin` | `merlin.cli:main` |
| `merlin-asm-audit` | `merlin.kernels.cli_asm:main` |
| `merlin-compile` | `merlin.compile_cli:main` |
| `merlin-compile-kernel` | `merlin.triton.cli:main` |
| `merlin-deployment-check` | `merlin.perf.deployment_admissibility:main` |
| `merlin-firesim-checkpoint` | `merlin.perf.firesim_checkpoint:main` |
| `merlin-firesim-receipt` | `merlin.perf.firesim_receipt:main` |
| `merlin-lit-suite` | `merlin.targetgen.lit_suite:main` |
| `merlin-liveness` | `merlin.liveness.cli:main` |
| `merlin-onboard` | `merlin.targetgen.onboard:main` |
| `merlin-opt` | `merlin.xdsl_dialects.opt:main` |
| `merlin-storage` | `merlin.common.storage_cli:main` |
| `merlin-surface` | `merlin.kernels.cli_surface:main` |
| `merlin-target-fetch` | `merlin.targetgen.oot_fetch:main` |
| `merlin-target-publish` | `merlin.targetgen.publish:main` |
| `merlin-target-tools` | `merlin.targetgen.tool_cli:main` |
| `merlin-targetgen` | `merlin.targetgen.cli:main` |
| `merlin-verify` | `merlin.verify.cli:main` |

## merlin-analysis

| Command | Backing module |
|---|---|
| `merlin-bundle-pretranspose` | `merlin.baselines.pretranspose_cli:main` |
| `merlin-compare` | `merlin.compare.cli:main` |
| `merlin-recovery` | `merlin.perf.recovery:main` |

## merlin-dse

| Command | Backing module |
|---|---|
| `merlin-design-pressure` | `merlin_dse.cli:pressure_main` |
| `merlin-dse` | `merlin_dse.cli:search_main` |
| `merlin-dse-guidance` | `merlin_dse.cli:guidance_main` |

## merlin-experiments

| Command | Backing module |
|---|---|
| `merlin-evaluation-cohort` | `merlin.targetgen.evaluation_cohort:main` |
| `merlin-experiment` | `merlin_experiments.cli:main` |

## merlin-mining

| Command | Backing module |
|---|---|
| `merlin-cca-route` | `merlin.mining.route_report:main` |
| `merlin-kernel-autotune` | `merlin.mining.autotune:main` |
| `merlin-kernel-beam` | `merlin.mining.beam_cli:main` |
| `merlin-kernel-mine` | `merlin.mining.mine:main` |
| `merlin-kernel-opt` | `merlin.mining.op_sweep:main` |
| `merlin-kernel-report` | `merlin.mining.report:main` |
| `merlin-rvv-autotune` | `merlin.mining.autotune:main` |
| `merlin-rvv-beam` | `merlin.mining.beam_cli:main` |
| `merlin-rvv-mine` | `merlin.mining.mine:main` |
| `merlin-rvv-opt` | `merlin.mining.op_sweep:main` |
| `merlin-rvv-report` | `merlin.mining.report:main` |
