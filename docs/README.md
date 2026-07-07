# Documentation

**AUTO-GENERATED index** by `build_tools/scripts/gen_docs_index.py` from each doc's front-matter.
Do not edit by hand — run the generator (it's `--check`ed by `check_structure.py`).

## Start here

- **New to the repo?** Read [Architecture](reference/architecture.md) then
  [Repository structure](reference/repo_structure.md).
- **Running something?** [Getting started](guides/getting_started.md) →
  the [CLI reference](reference/cli.md) → the relevant guide below.
- **Writing generated output?** See `CLAUDE.md` "Generated-output convention" and the
  `artifact-layout` skill (three roots only: `runs/`, `artifacts/`, `build/`).
- **Point-in-time results/reports** do not live here — they live under `artifacts/`.

Each entry shows **status** and **last-verified** date; `⚠` flags a doc whose `last_verified`
predates the newest change to the code it documents (see `check_docs_freshness.py`).

## Reference

- [Architecture](reference/architecture.md) — `current`, verified 2026-07-07 · owner: core — see also: [repo_structure](reference/repo_structure.md), [core_dialects](reference/core_dialects.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [CLI reference](reference/cli.md) — `generated` · owner: tooling
- [Contracts](reference/contracts.md) — `current`, verified 2026-07-07 · owner: ir — see also: [core_dialects](reference/core_dialects.md)
- [Core dialects](reference/core_dialects.md) — `current`, verified 2026-07-07 · owner: ir — see also: [dialects](reference/dialects.md), [contracts](reference/contracts.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [Dialects](reference/dialects.md) — `current`, verified 2026-07-07 · owner: ir — see also: [core_dialects](reference/core_dialects.md), [xdsl](reference/xdsl.md)
- [Experiment ABI](reference/experiment_abi.md) — `current`, verified 2026-07-07 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md)
- [Generated target repositories](reference/generated_target_repos.md) — `current`, verified 2026-07-07 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md)
- [Lowering pipeline](reference/lowering_pipeline.md) — `current`, verified 2026-07-07 · owner: ir — see also: [core_dialects](reference/core_dialects.md), [llvm_integration](guides/llvm_integration.md)
- [Package module index](reference/module_index.md) — `generated` · owner: tooling
- [Plotting house style](reference/plot_style.md) — `current`, verified 2026-07-07 · owner: plotting
- [Repository structure](reference/repo_structure.md) — `current`, verified 2026-07-07 · owner: core — see also: [architecture](reference/architecture.md)
- [Runtime](reference/runtime.md) — `current`, verified 2026-07-07 · owner: runtime — see also: [zephyr](guides/zephyr.md)
- [xDSL prototyping plane](reference/xdsl.md) — `current`, verified 2026-07-07 · owner: ir — see also: [dialects](reference/dialects.md), [core_dialects](reference/core_dialects.md)

## Guides

- [Adding a target](guides/adding_a_target.md) — `current`, verified 2026-07-07 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [generated_target_repos](reference/generated_target_repos.md)
- [Compilation strategies](guides/compilation_strategies.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [Design pressure](guides/design_pressure.md) — `current`, verified 2026-07-07 · owner: design_pressure — see also: [dse](guides/dse.md)
- [Design-space exploration](guides/dse.md) — `current`, verified 2026-07-07 · owner: dse — see also: [search](guides/search.md), [compilation_strategies](guides/compilation_strategies.md), [design_pressure](guides/design_pressure.md)
- [DSE guidance](guides/dse_guidance.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md), [design_pressure](guides/design_pressure.md)
- [Getting started](guides/getting_started.md) — `current`, verified 2026-07-07 · owner: core — see also: [architecture](reference/architecture.md), [repo_structure](reference/repo_structure.md), [dse](guides/dse.md), [kernel_mining](guides/kernel_mining.md), [targetgen](guides/targetgen.md)
- [Integrations](guides/integrations.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [kernel_mining](guides/kernel_mining.md)
- [Kernel abstraction mining](guides/kernel_mining.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [integrations](guides/integrations.md), [dse](guides/dse.md)
- [LLVM integration](guides/llvm_integration.md) — `current`, verified 2026-07-07 · owner: runtime — see also: [lowering_pipeline](reference/lowering_pipeline.md)
- [model2MLIR frontend](guides/model2mlir.md) — `current`, verified 2026-07-07 · owner: frontends — see also: [lowering_pipeline](reference/lowering_pipeline.md)
- [Search policy](guides/search.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md)
- [Target generation](guides/targetgen.md) — `current`, verified 2026-07-07 · owner: targetgen — see also: [adding_a_target](guides/adding_a_target.md), [experiment_abi](reference/experiment_abi.md), [generated_target_repos](reference/generated_target_repos.md)
- [Zephyr runtime backend](guides/zephyr.md) — `current`, verified 2026-07-07 · owner: runtime — see also: [runtime](reference/runtime.md)

## Design notes

- [Design audit: standalone merlin wheel](design/standalone_packaging.md) — `current`, verified 2026-07-07 · owner: core — see also: [repo_structure](reference/repo_structure.md)
- [Design note: integration adapters](design/integrations.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [integrations](guides/integrations.md)
- [Design note: the future MLIR/C++ compiler plane](design/compiler_plane.md) — `current`, verified 2026-07-07 · owner: core — see also: [architecture](reference/architecture.md)
- [Parallel workstreams](design/parallel_workstreams.md) — `current`, verified 2026-07-07 · owner: core — see also: [architecture](reference/architecture.md), [kernel_mining](guides/kernel_mining.md), [dse](guides/dse.md), [targetgen](guides/targetgen.md)

## By area

- **core** — [Architecture](reference/architecture.md), [Design audit: standalone merlin wheel](design/standalone_packaging.md), [Design note: the future MLIR/C++ compiler plane](design/compiler_plane.md), [Getting started](guides/getting_started.md), [Parallel workstreams](design/parallel_workstreams.md), [Repository structure](reference/repo_structure.md)
- **design_pressure** — [Design pressure](guides/design_pressure.md)
- **dse** — [Compilation strategies](guides/compilation_strategies.md), [Design-space exploration](guides/dse.md), [DSE guidance](guides/dse_guidance.md), [Search policy](guides/search.md)
- **frontends** — [model2MLIR frontend](guides/model2mlir.md)
- **ir** — [Contracts](reference/contracts.md), [Core dialects](reference/core_dialects.md), [Dialects](reference/dialects.md), [Lowering pipeline](reference/lowering_pipeline.md), [xDSL prototyping plane](reference/xdsl.md)
- **kernels** — [Design note: integration adapters](design/integrations.md), [Integrations](guides/integrations.md), [Kernel abstraction mining](guides/kernel_mining.md)
- **plotting** — [Plotting house style](reference/plot_style.md)
- **runtime** — [LLVM integration](guides/llvm_integration.md), [Runtime](reference/runtime.md), [Zephyr runtime backend](guides/zephyr.md)
- **targetgen** — [Adding a target](guides/adding_a_target.md), [Experiment ABI](reference/experiment_abi.md), [Generated target repositories](reference/generated_target_repos.md), [Target generation](guides/targetgen.md)
- **tooling** — [CLI reference](reference/cli.md), [Package module index](reference/module_index.md)

## Uncategorized (needs front-matter or relocation to artifacts/)

- [baselines_cross_framework_k1.md](baselines_cross_framework_k1.md)
- [beam_rvv_v2_report.md](beam_rvv_v2_report.md)
- [gemmini_requant_reconciliation.md](gemmini_requant_reconciliation.md)
- [gemmini_rtl_oracle_status.md](gemmini_rtl_oracle_status.md)
- [gemmini_target_prototype.md](gemmini_target_prototype.md)
- [implementation_milestones.md](implementation_milestones.md)
- [results.md](results.md)
- [rvv_kernel_mining_methodology.md](rvv_kernel_mining_methodology.md)
- [rvv_kernel_mining_results.md](rvv_kernel_mining_results.md)
- [rvv_mining_report.md](rvv_mining_report.md)
- [saturn_vec_findings.md](saturn_vec_findings.md)
- [validation_plan.md](validation_plan.md)
