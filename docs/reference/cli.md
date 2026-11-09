# CLI reference

_Generated from `pyproject.toml [project.scripts]` by `build_tools/scripts/gen_cli_docs.py` — do not edit by hand; run the generator._

Merlin commands are installed from the repository root with `pip install -e .`. Optional extras select workflow dependencies within this single distribution. The independent timing observer under `packages/` is installed separately. Run any command with `--help`.

| Command | Backing module |
|---|---|
| `kernel-audit` | `merlin.kernels.audit:main` |
| `kernel-bench` | `merlin.kernels.bench_ceiling:main` |
| `kernel-extract` | `merlin.kernels.cli_extract:main` |
| `kernel-index` | `merlin.kernels.cli_index:main` |
| `merlin` | `merlin.cli:main` |
| `merlin-asm-audit` | `merlin.kernels.cli_asm:main` |
| `merlin-bundle-pretranspose` | `merlin.baselines.pretranspose_cli:main` |
| `merlin-cca-route` | `merlin.mining.route_report:main` |
| `merlin-compare` | `merlin.compare.cli:main` |
| `merlin-compile` | `merlin.compile_cli:main` |
| `merlin-compile-kernel` | `merlin.triton.cli:main` |
| `merlin-deployment-check` | `merlin.perf.deployment_admissibility:main` |
| `merlin-design-pressure` | `merlin.analysis.dse.pressure.cli:main` |
| `merlin-dse` | `merlin.analysis.dse.search.cli:main` |
| `merlin-dse-guidance` | `merlin.analysis.dse.guidance.cli:main` |
| `merlin-evaluation-cohort` | `merlin.targetgen.evaluation_cohort:main` |
| `merlin-experiment` | `merlin.experiments.cli:main` |
| `merlin-firesim-checkpoint` | `merlin.perf.firesim_checkpoint:main` |
| `merlin-firesim-receipt` | `merlin.perf.firesim_receipt:main` |
| `merlin-isa-generate` | `merlin.targetgen.isa_codegen_cli:main` |
| `merlin-kernel-autotune` | `merlin.mining.autotune:main` |
| `merlin-kernel-beam` | `merlin.mining.beam_cli:main` |
| `merlin-kernel-mine` | `merlin.mining.mine:main` |
| `merlin-kernel-opt` | `merlin.mining.op_sweep:main` |
| `merlin-kernel-reconstruct` | `merlin.mining.reconstruction_cli:main` |
| `merlin-kernel-report` | `merlin.mining.report:main` |
| `merlin-lit-suite` | `merlin.targetgen.lit_suite:main` |
| `merlin-liveness` | `merlin.liveness.cli:main` |
| `merlin-onboard` | `merlin.targetgen.onboard:main` |
| `merlin-opt` | `merlin.xdsl_dialects.opt:main` |
| `merlin-phase2-normal-timing-study` | `merlin.experiments.phase2.ordinary_timing_study:main` |
| `merlin-phase2-public-diagnostic` | `merlin.experiments.phase2.public_diagnostic_launch:main` |
| `merlin-phase2-timing-readiness` | `merlin.experiments.phase2.timing_domain_readiness:main` |
| `merlin-phase2-timing-study` | `merlin.experiments.phase2.timing_study:main` |
| `merlin-recovery` | `merlin.perf.recovery:main` |
| `merlin-rvv-autotune` | `merlin.mining.autotune:main` |
| `merlin-rvv-beam` | `merlin.mining.beam_cli:main` |
| `merlin-rvv-mine` | `merlin.mining.mine:main` |
| `merlin-rvv-opt` | `merlin.mining.op_sweep:main` |
| `merlin-rvv-report` | `merlin.mining.report:main` |
| `merlin-semantic-facts` | `merlin.targetgen.rtl.semantic_facts:main` |
| `merlin-storage` | `merlin.common.storage_cli:main` |
| `merlin-surface` | `merlin.kernels.cli_surface:main` |
| `merlin-target-fetch` | `merlin.targetgen.oot_fetch:main` |
| `merlin-target-publish` | `merlin.targetgen.publish:main` |
| `merlin-target-tools` | `merlin.targetgen.tool_cli:main` |
| `merlin-targetgen` | `merlin.targetgen.cli:main` |
| `merlin-verify` | `merlin.verification.cli:main` |
| `merlin-verify-lean` | `merlin.verification.lean_cli:main` |
| `merlin-whole-model-gate` | `merlin.perf.whole_model_gate:main` |
| `merlin-whole-model-gsim` | `merlin.perf.whole_model_gsim:main` |

## merlin-timing

| Command | Backing module |
|---|---|
| `spike-timing` | `merlin_timing.cli:main` |
