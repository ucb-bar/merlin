# Documentation

**AUTO-GENERATED index** by `build_tools/scripts/gen_docs_index.py` from each doc's front-matter.
Do not edit by hand — run the generator (it's `--check`ed by `check_structure.py`).

## Start here

- **New to the repo?** Read [Architecture](reference/architecture.md) then
  [Repository structure](reference/repo_structure.md).
- **Running something?** [Getting started](guides/getting_started.md) →
  the [CLI reference](reference/cli.md) → the relevant guide below.
- **Writing generated output?** See `CLAUDE.md` "Generated-output convention" and the
  `artifact-layout` skill (one root only: `out/`, with three subdirs `out/runs/`, `out/artifacts/`,
  `out/build/`).
- **Point-in-time reports** (results, findings, status snapshots) do not live here — they live under `out/artifacts/`.

Each entry shows **status** and **last-verified** date; `⚠` flags a doc whose `last_verified`
predates the newest change to the code it documents (see `check_docs_freshness.py`).

## Reference

- [Architecture](reference/architecture.md) — `current`, verified 2026-07-14 · owner: core — see also: [repo_structure](reference/repo_structure.md), [core_dialects](reference/core_dialects.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [CLI reference](reference/cli.md) — `generated` · owner: tooling
- [Contracts](reference/contracts.md) — `current`, verified 2026-07-14 · owner: ir — see also: [core_dialects](reference/core_dialects.md)
- [Core dialects](reference/core_dialects.md) — `current`, verified 2026-07-14 · owner: ir — see also: [dialects](reference/dialects.md), [contracts](reference/contracts.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [Dialects](reference/dialects.md) — `current`, verified 2026-07-14 · owner: ir — see also: [core_dialects](reference/core_dialects.md), [xdsl](reference/xdsl.md)
- [Experiment ABI](reference/experiment_abi.md) — `current`, verified 2026-07-14 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md)
- [Generated target repositories](reference/generated_target_repos.md) — `current`, verified 2026-07-07 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md)
- [Lowering pipeline](reference/lowering_pipeline.md) — `current`, verified 2026-07-14 · owner: ir — see also: [core_dialects](reference/core_dialects.md), [llvm_integration](guides/llvm_integration.md)
- [merlin/ layout — what goes where](reference/merlin_layout.md) — `current`, verified 2026-07-14 · owner: core — see also: [repo_structure](reference/repo_structure.md), [architecture](reference/architecture.md)
- [Package module index](reference/module_index.md) — `generated` · owner: tooling
- [Plotting house style](reference/plot_style.md) — `current`, verified 2026-07-14 · owner: plotting
- [Repository structure](reference/repo_structure.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md)
- [Runtime](reference/runtime.md) — `current`, verified 2026-07-14 · owner: runtime — see also: [zephyr](guides/zephyr.md)
- [RVV kernel-mining methodology](reference/rvv_kernel_mining_methodology.md) — `current`, verified 2026-07-14 · owner: kernels — see also: [kernel_mining](guides/kernel_mining.md), [dse](guides/dse.md)
- [Schema reference](reference/schemas.md) — `generated` · owner: tooling
- [xDSL prototyping plane](reference/xdsl.md) — `current`, verified 2026-07-14 · owner: ir — see also: [dialects](reference/dialects.md), [core_dialects](reference/core_dialects.md)

## Guides

- [Adding a target](guides/adding_a_target.md) — `current`, verified 2026-07-14 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [generated_target_repos](reference/generated_target_repos.md)
- [Compilation strategies](guides/compilation_strategies.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [Design pressure](guides/design_pressure.md) — `current`, verified 2026-07-10 · owner: design_pressure — see also: [dse](guides/dse.md)
- [Design-space exploration](guides/dse.md) — `current`, verified 2026-07-10 · owner: dse — see also: [search](guides/search.md), [compilation_strategies](guides/compilation_strategies.md), [design_pressure](guides/design_pressure.md)
- [DSE guidance](guides/dse_guidance.md) — `current`, verified 2026-07-14 · owner: dse — see also: [dse](guides/dse.md), [design_pressure](guides/design_pressure.md)
- [Getting started](guides/getting_started.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md), [repo_structure](reference/repo_structure.md), [dse](guides/dse.md), [kernel_mining](guides/kernel_mining.md), [targetgen](guides/targetgen.md)
- [Integrations](guides/integrations.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [kernel_mining](guides/kernel_mining.md)
- [Kernel abstraction mining](guides/kernel_mining.md) — `current`, verified 2026-07-14 · owner: kernels — see also: [integrations](guides/integrations.md), [dse](guides/dse.md), [rvv_kernel_mining_methodology](reference/rvv_kernel_mining_methodology.md)
- [LLVM integration](guides/llvm_integration.md) — `current`, verified 2026-07-14 · owner: runtime — see also: [lowering_pipeline](reference/lowering_pipeline.md)
- [model2MLIR frontend](guides/model2mlir.md) — `current`, verified 2026-07-10 · owner: frontends — see also: [lowering_pipeline](reference/lowering_pipeline.md), [reproducibility](guides/reproducibility.md)
- [Reproducibility & core workflows](guides/reproducibility.md) — `current`, verified 2026-07-14 · owner: targetgen — see also: [getting_started](guides/getting_started.md), [adding_a_target](guides/adding_a_target.md), [kernel_mining](guides/kernel_mining.md), [dse](guides/dse.md), [model2mlir](guides/model2mlir.md)
- [RVV beam search — reproducible expert-driven compiler improvement, gated on real K1 speedup](guides/beam_search.md) — `current`, verified 2026-07-17 · owner: rvvgen — see also: [kernel_mining](guides/kernel_mining.md), [rvv_e2e](guides/rvv_e2e.md), [adding_a_target](guides/adding_a_target.md), [dse_guidance](guides/dse_guidance.md)
- [RVV end-to-end — lower a model through model2MLIR and run it on the Merlin runtime](guides/rvv_e2e.md) — `current`, verified 2026-07-16 · owner: runtime — see also: [model2mlir](guides/model2mlir.md), [reproducibility](guides/reproducibility.md), [getting_started](guides/getting_started.md), [kernel_mining](guides/kernel_mining.md)
- [Search policy](guides/search.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md)
- [Target generation](guides/targetgen.md) — `current`, verified 2026-07-14 · owner: targetgen — see also: [adding_a_target](guides/adding_a_target.md), [experiment_abi](reference/experiment_abi.md), [generated_target_repos](reference/generated_target_repos.md)
- [Zephyr runtime backend](guides/zephyr.md) — `current`, verified 2026-07-14 · owner: runtime — see also: [runtime](reference/runtime.md)

## Design notes

- ["Design note: attributing the expert-kernel gap (instructions vs stalls)"](design/expert_gap_attribution.md) — `current`, verified 2026-07-19 · owner: core — see also: [beam_search](guides/beam_search.md)
- ["Design note: auditing for runtime escapes in emitted compute regions"](design/runtime_escape_audit.md) — `current`, verified 2026-07-19 · owner: core — see also: [expert_gap_attribution](design/expert_gap_attribution.md), [compiler_plane](design/compiler_plane.md)
- [Design audit: standalone merlin wheel](design/standalone_packaging.md) — `current`, verified 2026-07-14 · owner: core — see also: [repo_structure](reference/repo_structure.md)
- [Design note: integration adapters](design/integrations.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [integrations](guides/integrations.md)
- [Design note: the future MLIR/C++ compiler plane](design/compiler_plane.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md)
- [Parallel workstreams](design/parallel_workstreams.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md), [kernel_mining](guides/kernel_mining.md), [dse](guides/dse.md), [targetgen](guides/targetgen.md)
- [Target publishing — the "target becomes its own repo" bridge (WS-E)](design/target_publishing.md) — `current`, verified 2026-07-15 · owner: core — see also: [repo_structure](reference/repo_structure.md), [integrations](guides/integrations.md)

## By area

- **core** — ["Design note: attributing the expert-kernel gap (instructions vs stalls)"](design/expert_gap_attribution.md), ["Design note: auditing for runtime escapes in emitted compute regions"](design/runtime_escape_audit.md), [Architecture](reference/architecture.md), [Design audit: standalone merlin wheel](design/standalone_packaging.md), [Design note: the future MLIR/C++ compiler plane](design/compiler_plane.md), [Getting started](guides/getting_started.md), [merlin/ layout — what goes where](reference/merlin_layout.md), [Parallel workstreams](design/parallel_workstreams.md), [Repository structure](reference/repo_structure.md), [Target publishing — the "target becomes its own repo" bridge (WS-E)](design/target_publishing.md)
- **design_pressure** — [Design pressure](guides/design_pressure.md)
- **dse** — [Compilation strategies](guides/compilation_strategies.md), [Design-space exploration](guides/dse.md), [DSE guidance](guides/dse_guidance.md), [Search policy](guides/search.md)
- **frontends** — [model2MLIR frontend](guides/model2mlir.md)
- **ir** — [Contracts](reference/contracts.md), [Core dialects](reference/core_dialects.md), [Dialects](reference/dialects.md), [Lowering pipeline](reference/lowering_pipeline.md), [xDSL prototyping plane](reference/xdsl.md)
- **kernels** — [Design note: integration adapters](design/integrations.md), [Integrations](guides/integrations.md), [Kernel abstraction mining](guides/kernel_mining.md), [RVV kernel-mining methodology](reference/rvv_kernel_mining_methodology.md)
- **plotting** — [Plotting house style](reference/plot_style.md)
- **runtime** — [LLVM integration](guides/llvm_integration.md), [Runtime](reference/runtime.md), [RVV end-to-end — lower a model through model2MLIR and run it on the Merlin runtime](guides/rvv_e2e.md), [Zephyr runtime backend](guides/zephyr.md)
- **rvvgen** — [RVV beam search — reproducible expert-driven compiler improvement, gated on real K1 speedup](guides/beam_search.md)
- **targetgen** — [Adding a target](guides/adding_a_target.md), [Experiment ABI](reference/experiment_abi.md), [Generated target repositories](reference/generated_target_repos.md), [Reproducibility & core workflows](guides/reproducibility.md), [Target generation](guides/targetgen.md)
- **tooling** — [CLI reference](reference/cli.md), [Package module index](reference/module_index.md), [Schema reference](reference/schemas.md)
