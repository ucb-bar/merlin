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
- [DSE boundary-placement vocabulary](reference/dse_boundary_vocabulary.md) — `current`, verified 2026-08-10 · owner: dse — see also: [dse_guidance](guides/dse_guidance.md), [contracts](reference/contracts.md), [design_pressure](guides/design_pressure.md)
- [Experiment ABI](reference/experiment_abi.md) — `current`, verified 2026-07-14 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md)
- [Generated target repositories](reference/generated_target_repos.md) — `current`, verified 2026-07-07 · owner: targetgen — see also: [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md)
- [Lowering pipeline](reference/lowering_pipeline.md) — `current`, verified 2026-07-14 · owner: ir — see also: [core_dialects](reference/core_dialects.md), [llvm_integration](guides/llvm_integration.md)
- [merlin/ layout — what goes where](reference/merlin_layout.md) — `current`, verified 2026-07-14 · owner: core — see also: [repo_structure](reference/repo_structure.md), [architecture](reference/architecture.md)
- [Package module index](reference/module_index.md) — `generated` · owner: tooling
- [Plotting house style](reference/plot_style.md) — `current`, verified 2026-07-14 · owner: plotting
- [Repository structure](reference/repo_structure.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md)
- [Runtime](reference/runtime.md) — `current`, verified 2026-07-14 · owner: runtime — see also: [zephyr](guides/zephyr.md)
- [RVV kernel-mining methodology](reference/rvv_kernel_mining_methodology.md) — `current`, verified 2026-07-19 · owner: kernels — see also: [kernel_mining](guides/kernel_mining.md), [dse](guides/dse.md)
- [Schema reference](reference/schemas.md) — `generated` · owner: tooling
- [xDSL prototyping plane](reference/xdsl.md) — `current`, verified 2026-07-14 · owner: ir — see also: [dialects](reference/dialects.md), [core_dialects](reference/core_dialects.md)

## Guides

- ["Compiling Triton kernels with Merlin"](guides/triton_kernels.md) — `draft`, verified 2026-08-10 · owner: ir — see also: [triton_frontend](design/triton_frontend.md), [lowering_pipeline](reference/lowering_pipeline.md), [target_resolution](guides/target_resolution.md)
- [Adding a target](guides/adding_a_target.md) — `current`, verified 2026-07-22 · owner: targetgen — see also: [getting_started](guides/getting_started.md), [targetgen](guides/targetgen.md), [generated_target_repos](reference/generated_target_repos.md)
- [Building the pinned LLVM/MLIR toolchain (third_party/llvm-install)](guides/llvm_toolchain.md) — `current`, verified 2026-08-10 · owner: runtime — see also: [getting_started](guides/getting_started.md), [llvm_integration](guides/llvm_integration.md), [reproducibility](guides/reproducibility.md), [model2mlir](guides/model2mlir.md), [zephyr](guides/zephyr.md), [targetgen](guides/targetgen.md)
- [Compilation strategies](guides/compilation_strategies.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md), [lowering_pipeline](reference/lowering_pipeline.md)
- [Design pressure](guides/design_pressure.md) — `current`, verified 2026-07-22 · owner: design_pressure — see also: [getting_started](guides/getting_started.md), [dse](guides/dse.md), [dse_guidance](guides/dse_guidance.md)
- [Design-space exploration](guides/dse.md) — `current`, verified 2026-07-22 · owner: dse — see also: [getting_started](guides/getting_started.md), [search](guides/search.md), [compilation_strategies](guides/compilation_strategies.md), [design_pressure](guides/design_pressure.md), [dse_guidance](guides/dse_guidance.md)
- [DSE guidance](guides/dse_guidance.md) — `current`, verified 2026-07-22 · owner: dse — see also: [getting_started](guides/getting_started.md), [dse](guides/dse.md), [design_pressure](guides/design_pressure.md)
- [FireSim — whole-model cycle truth on the FPGA](guides/firesim.md) — `current`, verified 2026-07-26 · owner: runtime — see also: [zephyr](guides/zephyr.md), [tinyllama_int8_rvv_zephyr](guides/tinyllama_int8_rvv_zephyr.md), [getting_started](guides/getting_started.md), [reproducibility](guides/reproducibility.md)
- [Gemmini target-dialect-generation experiment (case study)](guides/gemmini_experiment.md) — `current`, verified 2026-07-22 · owner: targetgen — see also: [getting_started](guides/getting_started.md), [reproducibility](guides/reproducibility.md), [targetgen](guides/targetgen.md), [adding_a_target](guides/adding_a_target.md), [target_publishing](design/target_publishing.md), [experiment_abi](reference/experiment_abi.md)
- [Getting started — the setup and prerequisites reference](guides/getting_started.md) — `current`, verified 2026-07-22 · owner: core
- [Integrations](guides/integrations.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [kernel_mining](guides/kernel_mining.md)
- [Kernel abstraction mining](guides/kernel_mining.md) — `current`, verified 2026-07-22 · owner: kernels — see also: [getting_started](guides/getting_started.md), [integrations](guides/integrations.md), [dse](guides/dse.md), [beam_search](guides/beam_search.md), [rvv_kernel_mining_methodology](reference/rvv_kernel_mining_methodology.md)
- [LLVM integration](guides/llvm_integration.md) — `current`, verified 2026-07-14 · owner: runtime — see also: [lowering_pipeline](reference/lowering_pipeline.md), [llvm_toolchain](guides/llvm_toolchain.md)
- [model2MLIR frontend](guides/model2mlir.md) — `current`, verified 2026-07-22 · owner: frontends — see also: [getting_started](guides/getting_started.md), [lowering_pipeline](reference/lowering_pipeline.md), [reproducibility](guides/reproducibility.md), [rvv_e2e](guides/rvv_e2e.md)
- [Reproducibility & core workflows — the master guide](guides/reproducibility.md) — `current`, verified 2026-07-22 · owner: targetgen
- [Running capsule-bench experiments on AWS Bedrock](guides/bedrock_experiments.md) — `current`, verified 2026-07-30 · owner: capsule-bench — see also: [adding_a_target](guides/adding_a_target.md), [getting_started](guides/getting_started.md)
- [RVV beam search — reproducible expert-driven compiler improvement, gated on real K1 speedup](guides/beam_search.md) — `current`, verified 2026-07-22 · owner: rvvgen — see also: [getting_started](guides/getting_started.md), [reproducibility](guides/reproducibility.md), [kernel_mining](guides/kernel_mining.md), [rvv_e2e](guides/rvv_e2e.md), [adding_a_target](guides/adding_a_target.md), [dse_guidance](guides/dse_guidance.md)
- [RVV end-to-end — lower a model through model2MLIR and run it on the Merlin runtime](guides/rvv_e2e.md) — `current`, verified 2026-07-22 · owner: runtime
- [Search policy](guides/search.md) — `current`, verified 2026-07-07 · owner: dse — see also: [dse](guides/dse.md)
- [Selecting a target definition package](guides/target_resolution.md) — `current`, verified 2026-07-27 · owner: targetgen — see also: [adding_a_target](guides/adding_a_target.md), [generated_target_repos](reference/generated_target_repos.md), [targetgen](guides/targetgen.md), [target_publishing](design/target_publishing.md)
- [Target generation](guides/targetgen.md) — `current`, verified 2026-07-22 · owner: targetgen — see also: [getting_started](guides/getting_started.md), [adding_a_target](guides/adding_a_target.md), [experiment_abi](reference/experiment_abi.md), [generated_target_repos](reference/generated_target_repos.md)
- [TinyLlama int8 on multicore RVV under Zephyr — end to end](guides/tinyllama_int8_rvv_zephyr.md) — `current`, verified 2026-07-26 · owner: runtime — see also: [getting_started](guides/getting_started.md), [rvv_e2e](guides/rvv_e2e.md), [zephyr](guides/zephyr.md), [model2mlir](guides/model2mlir.md), [reproducibility](guides/reproducibility.md), [compilation_strategies](guides/compilation_strategies.md), [vision_workloads_rvv_zephyr](guides/vision_workloads_rvv_zephyr.md)
- [Vision, audio and control workloads on Kodiak — multicore RVV under Zephyr](guides/vision_workloads_rvv_zephyr.md) — `current`, verified 2026-08-04 · owner: runtime — see also: [tinyllama_int8_rvv_zephyr](guides/tinyllama_int8_rvv_zephyr.md), [model2mlir](guides/model2mlir.md), [rvv_e2e](guides/rvv_e2e.md), [zephyr](guides/zephyr.md), [compilation_strategies](guides/compilation_strategies.md)
- [Zephyr runtime backend](guides/zephyr.md) — `current`, verified 2026-07-22 · owner: runtime — see also: [getting_started](guides/getting_started.md), [reproducibility](guides/reproducibility.md), [runtime](reference/runtime.md), [tinyllama_int8_rvv_zephyr](guides/tinyllama_int8_rvv_zephyr.md)

## Design notes

- ["Design note: attributing the expert-kernel gap (instructions vs stalls)"](design/expert_gap_attribution.md) — `current`, verified 2026-07-19 · owner: core — see also: [beam_search](guides/beam_search.md)
- ["Design note: auditing for runtime escapes in emitted compute regions"](design/runtime_escape_audit.md) — `current`, verified 2026-07-19 · owner: core — see also: [expert_gap_attribution](design/expert_gap_attribution.md), [compiler_plane](design/compiler_plane.md)
- ["Design note: VL-agnostic (scalable) RVV codegen — dropping the VLEN pin"](design/vl_agnostic_codegen.md) — `current`, verified 2026-07-19 · owner: core — see also: [expert_gap_attribution](design/expert_gap_attribution.md)
- ["Design note: where the whole-model codegen-vs-hand-C gap lives (it is the emitted matmul)"](design/codegen_vs_handc_wholemodel.md) — `current`, verified 2026-07-19 · owner: core — see also: [expert_gap_attribution](design/expert_gap_attribution.md)
- ["Design note: whole-model per-op profiler and where model time actually goes"](design/whole_model_op_profile.md) — `current`, verified 2026-07-19 · owner: core — see also: [expert_gap_attribution](design/expert_gap_attribution.md), [runtime_escape_audit](design/runtime_escape_audit.md)
- ["Design note: whole-model transpose-b fusion (fuse_transpose_b)"](design/transpose_fusion.md) — `current`, verified 2026-07-19 · owner: core — see also: [whole_model_op_profile](design/whole_model_op_profile.md), [expert_gap_attribution](design/expert_gap_attribution.md), [compiler_plane](design/compiler_plane.md)
- ["Design: drive the core to zero target-specific literals"](design/target_agnostic_core.md) — `draft`, verified 2026-07-29 · owner: targetgen — see also: [target_resolution](guides/target_resolution.md)
- ["Design: incremental target evolution — RVV + Saturn OPU as the driving delta"](design/incremental_target_evolution_opu.md) — `draft`, verified 2026-08-10 · owner: targetgen — see also: [beam_cca_architecture](design/beam_cca_architecture.md), [lowering_pipeline](reference/lowering_pipeline.md), [llvm_integration](guides/llvm_integration.md), [reproducibility](guides/reproducibility.md)
- ["Design: the CCA beam — two CCAs, cross-framework analysis, and autonomous whole-model improvement"](design/beam_cca_architecture.md) — `current`, verified 2026-07-20 · owner: rvvgen — see also: [beam_search](guides/beam_search.md), [expert_gap_attribution](design/expert_gap_attribution.md), [whole_model_op_profile](design/whole_model_op_profile.md), [transpose_fusion](design/transpose_fusion.md), [kernel_mining](guides/kernel_mining.md)
- ["Design: Triton as a target-independent kernel frontend"](design/triton_frontend.md) — `draft`, verified 2026-08-10 · owner: ir — see also: [lowering_pipeline](reference/lowering_pipeline.md), [core_dialects](reference/core_dialects.md), [target_agnostic_core](design/target_agnostic_core.md), [target_resolution](guides/target_resolution.md)
- [Design audit: standalone merlin wheel](design/standalone_packaging.md) — `current`, verified 2026-07-14 · owner: core — see also: [repo_structure](reference/repo_structure.md)
- [Design note: integration adapters](design/integrations.md) — `current`, verified 2026-07-07 · owner: kernels — see also: [integrations](guides/integrations.md)
- [Design note: the future MLIR/C++ compiler plane](design/compiler_plane.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md)
- [Parallel workstreams](design/parallel_workstreams.md) — `current`, verified 2026-07-14 · owner: core — see also: [architecture](reference/architecture.md), [kernel_mining](guides/kernel_mining.md), [dse](guides/dse.md), [targetgen](guides/targetgen.md)
- [Target publishing — the "target becomes its own repo" bridge (WS-E)](design/target_publishing.md) — `current`, verified 2026-07-15 · owner: core — see also: [repo_structure](reference/repo_structure.md), [integrations](guides/integrations.md)
- [The cross-target dialect test bar](design/dialect_test_bar.md) — `current`, verified 2026-07-25 · owner: core — see also: [compiler_plane](design/compiler_plane.md), [target_publishing](design/target_publishing.md)

## By area

- **capsule-bench** — [Running capsule-bench experiments on AWS Bedrock](guides/bedrock_experiments.md)
- **core** — ["Design note: attributing the expert-kernel gap (instructions vs stalls)"](design/expert_gap_attribution.md), ["Design note: auditing for runtime escapes in emitted compute regions"](design/runtime_escape_audit.md), ["Design note: VL-agnostic (scalable) RVV codegen — dropping the VLEN pin"](design/vl_agnostic_codegen.md), ["Design note: where the whole-model codegen-vs-hand-C gap lives (it is the emitted matmul)"](design/codegen_vs_handc_wholemodel.md), ["Design note: whole-model per-op profiler and where model time actually goes"](design/whole_model_op_profile.md), ["Design note: whole-model transpose-b fusion (fuse_transpose_b)"](design/transpose_fusion.md), [Architecture](reference/architecture.md), [Design audit: standalone merlin wheel](design/standalone_packaging.md), [Design note: the future MLIR/C++ compiler plane](design/compiler_plane.md), [Getting started — the setup and prerequisites reference](guides/getting_started.md), [merlin/ layout — what goes where](reference/merlin_layout.md), [Parallel workstreams](design/parallel_workstreams.md), [Repository structure](reference/repo_structure.md), [Target publishing — the "target becomes its own repo" bridge (WS-E)](design/target_publishing.md), [The cross-target dialect test bar](design/dialect_test_bar.md)
- **design_pressure** — [Design pressure](guides/design_pressure.md)
- **dse** — [Compilation strategies](guides/compilation_strategies.md), [Design-space exploration](guides/dse.md), [DSE boundary-placement vocabulary](reference/dse_boundary_vocabulary.md), [DSE guidance](guides/dse_guidance.md), [Search policy](guides/search.md)
- **frontends** — [model2MLIR frontend](guides/model2mlir.md)
- **ir** — ["Compiling Triton kernels with Merlin"](guides/triton_kernels.md), ["Design: Triton as a target-independent kernel frontend"](design/triton_frontend.md), [Contracts](reference/contracts.md), [Core dialects](reference/core_dialects.md), [Dialects](reference/dialects.md), [Lowering pipeline](reference/lowering_pipeline.md), [xDSL prototyping plane](reference/xdsl.md)
- **kernels** — [Design note: integration adapters](design/integrations.md), [Integrations](guides/integrations.md), [Kernel abstraction mining](guides/kernel_mining.md), [RVV kernel-mining methodology](reference/rvv_kernel_mining_methodology.md)
- **plotting** — [Plotting house style](reference/plot_style.md)
- **runtime** — [Building the pinned LLVM/MLIR toolchain (third_party/llvm-install)](guides/llvm_toolchain.md), [FireSim — whole-model cycle truth on the FPGA](guides/firesim.md), [LLVM integration](guides/llvm_integration.md), [Runtime](reference/runtime.md), [RVV end-to-end — lower a model through model2MLIR and run it on the Merlin runtime](guides/rvv_e2e.md), [TinyLlama int8 on multicore RVV under Zephyr — end to end](guides/tinyllama_int8_rvv_zephyr.md), [Vision, audio and control workloads on Kodiak — multicore RVV under Zephyr](guides/vision_workloads_rvv_zephyr.md), [Zephyr runtime backend](guides/zephyr.md)
- **rvvgen** — ["Design: the CCA beam — two CCAs, cross-framework analysis, and autonomous whole-model improvement"](design/beam_cca_architecture.md), [RVV beam search — reproducible expert-driven compiler improvement, gated on real K1 speedup](guides/beam_search.md)
- **targetgen** — ["Design: drive the core to zero target-specific literals"](design/target_agnostic_core.md), ["Design: incremental target evolution — RVV + Saturn OPU as the driving delta"](design/incremental_target_evolution_opu.md), [Adding a target](guides/adding_a_target.md), [Experiment ABI](reference/experiment_abi.md), [Gemmini target-dialect-generation experiment (case study)](guides/gemmini_experiment.md), [Generated target repositories](reference/generated_target_repos.md), [Reproducibility & core workflows — the master guide](guides/reproducibility.md), [Selecting a target definition package](guides/target_resolution.md), [Target generation](guides/targetgen.md)
- **tooling** — [CLI reference](reference/cli.md), [Package module index](reference/module_index.md), [Schema reference](reference/schemas.md)
