# Repository Guide

Merlin is organized to separate model frontends, compiler internals, and hardware-targeted runtimes.

This page is contributor-facing: file placement rules and the tracked-tree snapshot. If you are using Merlin for the first time and just want to know which entry points to invoke, read [user_paths.md](user_paths.md) instead.

## Contributor Layers

- **User layer**: `docs/`, `tools/`, `models/`, `build/`
- **Target bring-up layer**: `target_specs/`, `build_tools/hardware/`, `models/*.yaml`
- **Implementation layer**: `compiler/`, `runtime/`, `third_party/iree_bar`
- **Research / sidecar tooling**: `benchmarks/`, `samples/research/`, `projects/`

## Core Directories

- `compiler/`: C++ and MLIR compiler code (dialects, passes, plugins).
- `tools/`: Python developer entrypoints behind `./merlin <subcommand>` (see `tools/merlin.py`).
- `models/`: Model definitions, exports, and quantization helpers.
- `samples/`: C/C++ runtime examples and hardware-facing sample flows.
- `benchmarks/`: Benchmark scripts and board-specific profiling helpers.
- `docs/`: Documentation source consumed by MkDocs.

## Placement Conventions (Where New Code Should Go)

- New compiler dialects/passes/transforms: `compiler/src/merlin/`.
- New plugin/target registration glue: `compiler/plugins/`.
- New model exports or conversion flows: `models/<model_name>/`.
- New target flag bundles for `./merlin compile`: `models/<target>.yaml`.
- New board/runtime sample executables: `samples/<platform>/`.
- New benchmark flows and parsers: `benchmarks/<target>/`.
- New end-user docs and guides: `docs/`.

## Tracked Tree Snapshot (Depth 3)

```text
merlin/
├── .clang-format
├── .clang-tidy
├── .devcontainer
│   └── devcontainer.json
├── .dockerignore
├── .gitattributes
├── .gitignore
├── .gitmodules
├── .pre-commit-config.yaml
├── .python-version
├── AGENTS.md
├── CLAUDE.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── benchmarks
│   ├── CMakeLists.txt
│   ├── KernelBench
│   │   ├── documentation.md
│   │   ├── iree_compilation.py
│   │   ├── iree_run_subprocess.py
│   │   ├── report
│   │   └── results
│   ├── SaturnNPU
│   │   ├── .gitignore
│   │   ├── README.md
│   │   ├── kernel_library
│   │   └── scripts
│   ├── SaturnOPU
│   │   ├── README.md
│   │   ├── analyze_opu_coverage.py
│   │   ├── compile_matmul_opu_fp8_ukernel_all.sh
│   │   ├── compile_matmul_opu_i8_ukernel_all.sh
│   │   ├── firesim_v128d64_results.csv
│   │   ├── optimization_journey.csv
│   │   ├── optimization_journey.png
│   │   ├── performance_scaling.png
│   │   ├── plot_firesim_results.py
│   │   ├── plot_opu_coverage.py
│   │   ├── speedup_vs_rvv.png
│   │   └── utilization.png
│   └── SpacemiTX60
│       ├── README.md
│       ├── baseline_dual_model_async
│       ├── compile_dual_model_vmfb.sh
│       ├── compile_matmul_rvv_i8_ukernel_all.sh
│       ├── compile_matmul_xsmt_fp8.sh
│       ├── compile_matmul_xsmt_i8_ukernel_all.sh
│       ├── parse_dual_model_log.py
│       ├── run_dual_model_remote.sh
│       └── spacemitx60.env.example
├── build_tools
│   ├── README.md
│   ├── SpacemiT
│   │   └── setup_toolchain.sh
│   ├── docker
│   │   ├── build_release.sh
│   │   ├── dev.Dockerfile
│   │   ├── in_container_release.sh
│   │   └── linux-builder.Dockerfile
│   ├── firesim
│   │   ├── htif-nano.spec
│   │   ├── htif.ld
│   │   ├── htif_large_model.ld
│   │   ├── riscv_firesim.toolchain.cmake
│   │   ├── setup_toolchain.sh
│   │   └── spike.cfg
│   ├── hardware
│   │   ├── gemmini_mx.yaml
│   │   ├── gemmini_u250.yaml
│   │   ├── saturn_opu_u250.yaml
│   │   ├── scripts
│   │   ├── spacemit_x60.yaml
│   │   └── templates
│   └── patches
│       ├── README.md
│       ├── manifest.env
│       └── upstream
├── compiler
│   ├── plugins
│   │   ├── CMakeLists.txt
│   │   └── target
│   └── src
│       └── merlin
├── docker-compose.yml
├── docs
│   ├── --help
│   │   ├── OLD-3.7.0-iree-compile--help.txt
│   │   ├── iree-compile--help.txt
│   │   └── iree-run-module--help.txt
│   ├── Doxyfile.in
│   ├── _archive
│   │   └── radiance_author_questions.md
│   ├── architecture
│   │   ├── cmake_presets.md
│   │   ├── npu_compilation_pipeline.md
│   │   ├── plugin_and_patch_model.md
│   │   ├── ray_control_plane.md
│   │   ├── repo-maintenance-model.md
│   │   └── target_generator.md
│   ├── assets
│   │   └── merlin_transparent.png
│   ├── build_tracy_ubuntu.md
│   ├── dev_blog
│   │   ├── 2026-03-11-gemmini-workstream-log.md
│   │   ├── 2026-03-11-npu-dialect-e2e.md
│   │   ├── 2026-03-11-radiance-hal-workstream-log.md
│   │   ├── 2026-03-12-smolvla-fp8-int8-global-opt-workstream.md
│   │   ├── 2026-03-13-riscv-mmt4d-ukernel-workstream.md
│   │   ├── 2026-03-16-dispatch-level-async.md
│   │   ├── 2026-03-18-chipyard-bare-metal-integration.md
│   │   ├── 2026-03-25-ray-control-plane-bootstrap.md
│   │   ├── 2026-03-25-targetgen-generation-staging.md
│   │   ├── 2026-04-06-opu-utilization-e2e-benchmarking.md
│   │   ├── 2026-04-13-saturn-opu-vfredusum-scalarization.md
│   │   ├── 2026-04-14-f32-reduction-hang-findings.md
│   │   ├── TEMPLATE.md
│   │   └── index.md
│   ├── different_build_types.md
│   ├── getting_started.md
│   ├── hardware_backends
│   │   ├── chipyard_concepts.md
│   │   ├── compatibility_matrix.md
│   │   ├── overview.md
│   │   └── recipes
│   ├── hooks.py
│   ├── how_to
│   │   ├── add_a_new_npu_kernel.md
│   │   ├── add_compile_target.md
│   │   ├── add_compiler_dialect_plugin.md
│   │   ├── add_hardware_spec.md
│   │   ├── add_runtime_hal_driver.md
│   │   ├── add_sample_application.md
│   │   ├── index.md
│   │   ├── smolvla_end_to_end.md
│   │   └── use_build_py.md
│   ├── index.md
│   ├── reference
│   │   ├── cli.md
│   │   ├── cmake_targets.md
│   │   ├── cpp.md
│   │   ├── mlir.md
│   │   ├── overview.md
│   │   └── python
│   ├── repository_guide.md
│   ├── reproducibility
│   │   ├── cross_compile_banana_pi.md
│   │   ├── export_tinyllama_8bit_sharktank_to_mlir.md
│   │   └── reproduce_ukernel_benchmark_firesim.md
│   ├── requirements.txt
│   ├── stylesheets
│   │   └── extra.css
│   └── user_paths.md
├── env_linux.yml
├── env_macOS.yml
├── iree_compiler_plugin.cmake
├── iree_runtime_plugin.cmake
├── merlin
├── mkdocs.yml
├── models
│   ├── README.md
│   ├── compile_spacemit.sh
│   ├── depth_anything_v2
│   │   ├── depth_anything_v2.q.int8.mlir
│   │   └── depth_anything_v2_onnx.py
│   ├── diffusion
│   │   └── diffusion_policy.py
│   ├── dronet
│   │   ├── dronet.mlir
│   │   ├── dronet.py
│   │   └── dronet.q.int8.mlir
│   ├── fastdepth
│   │   └── fastdepth.mlir
│   ├── gemmini_mx.yaml
│   ├── glpdepth
│   │   ├── bear_image_data.h
│   │   ├── glpdepth.mlir
│   │   ├── glpdepth.q.int8.mlir
│   │   └── processed_image.png
│   ├── midas
│   │   └── midas_onnx.py
│   ├── mlp
│   │   ├── mlp.mlir
│   │   ├── mlp.py
│   │   └── mlp.q.int8.mlir
│   ├── mlp_wide
│   │   ├── mlp_wide.py
│   │   └── mlp_wide.q.int8.mlir
│   ├── mobilenet_v2
│   │   ├── mobilenet_v2.mlir
│   │   └── mobilenet_v2.q.int8.mlir
│   ├── models_config.json
│   ├── npu_ucb.yaml
│   ├── opu_bench_suite
│   │   ├── mlp_fast.q.int8.mlir
│   │   ├── opu_bench_convnet.q.int8.mlir
│   │   ├── opu_bench_hybrid.q.int8.mlir
│   │   ├── opu_bench_large_mlp.q.int8.mlir
│   │   ├── opu_bench_models.py
│   │   ├── opu_bench_vit.q.int8.mlir
│   │   └── opu_bench_vit_small.q.int8.mlir
│   ├── quantize_models.py
│   ├── saturn_opu.yaml
│   ├── smolVLA
│   │   ├── README.md
│   │   ├── export_smolvla.py
│   │   └── export_smolvla_int8.py
│   ├── spacemit_x60.yaml
│   ├── tinydepth
│   │   └── tinydepth.py
│   ├── tinyllama
│   │   ├── tinyllama.q8.json
│   │   └── tinyllama_export.py
│   └── yolov8_nano
│       ├── yolov8n.q.int8.mlir
│       └── yolov8nano_export.py
├── pyproject.toml
├── requirements.txt
├── runtime
│   └── src
│       └── iree
├── samples
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── SaturnOPU
│   │   ├── CMakeLists.txt
│   │   ├── custom_dispatch_ukernels
│   │   └── simple_embedding_ukernel
│   ├── SpacemiTX60
│   │   ├── CMakeLists.txt
│   │   ├── baseline_async
│   │   └── dispatch_scheduler
│   ├── common
│   │   ├── core
│   │   ├── dispatch
│   │   ├── runtime
│   │   └── xpu-rt
│   └── research
│       ├── mlir_pipelining
│       ├── model_computation_graph_generation
│       ├── promise_devices_layer
│       └── promise_schedule_multi_model
├── target_specs
│   ├── examples
│   │   ├── ara_rvv_vlen256
│   │   ├── gemmini_mx
│   │   ├── npu_ucb
│   │   ├── nvidia_cuda_sm89
│   │   ├── nvidia_vulkan_ada
│   │   ├── qualcomm_adreno_vulkan
│   │   ├── qualcomm_qnn_htp_snapdragon8elite
│   │   ├── radiance_muon
│   │   ├── saturn_opu_v128
│   │   ├── spacemit_x60_xsmtvdot
│   │   └── vortex_gpgpu_u250
│   └── schema
│       ├── capability_spec.yaml
│       └── deployment_overlay.yaml
├── tests
│   ├── raycp
│   │   └── test_raycp.py
│   └── targetgen
│       └── test_targetgen.py
├── third_party
│   ├── Understanding-PI0
│   ├── autocomp
│   ├── gemmini-mx
│   ├── gluon
│   ├── iree-turbine
│   ├── iree_bar
│   ├── lerobot
│   ├── npu_model
│   ├── saturn-vectors
│   └── torch-mlir
├── tools
│   ├── README.md
│   ├── analyze_quant_ir.py
│   ├── benchmark.py
│   ├── build.py
│   ├── chipyard.py
│   ├── ci.py
│   ├── compile.py
│   ├── install_prebuilt.py
│   ├── merlin.py
│   ├── patches.py
│   ├── ray_cmd.py
│   ├── raycp
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── service.py
│   ├── setup.py
│   ├── strip_mlir_weights.py
│   ├── targetgen
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── generator.py
│   │   ├── loader.py
│   │   ├── model.py
│   │   ├── orchestrator.py
│   │   ├── planner.py
│   │   ├── prompt_library
│   │   └── prompts.py
│   ├── targetgen_cmd.py
│   └── utils.py
└── uv.lock
```

This tree is generated from `git ls-files` so it reflects tracked repository state.
