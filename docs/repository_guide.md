# Repository Guide

Merlin is organized to separate model frontends, compiler internals, and hardware-targeted runtimes.

## Core Directories

- `compiler/`: C++ and MLIR compiler code (dialects, passes, plugins).
- `tools/`: Python developer entrypoints (`build.py`, `compile.py`, `setup.py`, `ci.py`, etc.).
- `models/`: Model definitions, exports, and quantization helpers.
- `samples/`: C/C++ runtime examples and hardware-facing sample flows.
- `benchmarks/`: Benchmark scripts and board-specific profiling helpers.
- `docs/`: Documentation source consumed by MkDocs.

## Placement Conventions (Where New Code Should Go)

- New compiler dialects/passes/transforms: `compiler/src/merlin/`.
- New plugin/target registration glue: `compiler/plugins/`.
- New model exports or conversion flows: `models/<model_name>/`.
- New target flag bundles for `tools/compile.py`: `models/<target>.yaml`.
- New board/runtime sample executables: `samples/<platform>/`.
- New benchmark flows and parsers: `benchmarks/<target>/`.
- New end-user docs and guides: `docs/`.

## Tracked Tree Snapshot (Depth 3)

```text
merlin/
├── .clang-format
├── .clang-tidy
├── .dockerignore
├── .gitattributes
├── .gitignore
├── .gitmodules
├── .pre-commit-config.yaml
├── .python-version
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
│   ├── SaturnOPU
│   │   ├── README.md
│   │   ├── compile_matmul_opu_fp8_ukernel_all.sh
│   │   └── compile_matmul_opu_i8_ukernel_all.sh
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
│   ├── SpacemiT
│   │   └── setup_toolchain.sh
│   ├── docker
│   │   ├── build_release.sh
│   │   ├── in_container_release.sh
│   │   └── linux-builder.Dockerfile
│   ├── firesim
│   │   ├── htif-nano.spec
│   │   ├── htif.ld
│   │   ├── riscv_firesim.toolchain.cmake
│   │   ├── setup_toolchain.sh
│   │   └── spike.cfg
│   └── patches
│       ├── README.md
│       ├── manifest.env
│       ├── series.iree
│       ├── series.llvm
│       └── tools
├── compiler
│   ├── plugins
│   │   ├── CMakeLists.txt
│   │   └── target
│   └── src
│       └── merlin
├── docs
│   ├── --help
│   │   ├── OLD-3.7.0-iree-compile--help.txt
│   │   ├── iree-compile--help.txt
│   │   └── iree-run-module--help.txt
│   ├── architecture
│   │   ├── cmake_presets.md
│   │   ├── plugin_and_patch_model.md
│   │   ├── radiance_author_questions.md
│   │   └── repo-maintenance-model.md
│   ├── assets
│   │   └── merlin_transparent.png
│   ├── build_tracy_ubuntu.md
│   ├── dev_blog
│   │   ├── 2026-03-11-gemmini-workstream-log.md
│   │   ├── 2026-03-11-npu-dialect-e2e.md
│   │   ├── 2026-03-11-radiance-hal-workstream-log.md
│   │   ├── 2026-03-12-smolvla-fp8-int8-global-opt-workstream.md
│   │   ├── 2026-03-13-riscv-mmt4d-ukernel-workstream.md
│   │   ├── TEMPLATE.md
│   │   └── index.md
│   ├── different_build_types.md
│   ├── getting_started.md
│   ├── hooks.py
│   ├── how_to
│   │   ├── add_compile_target.md
│   │   ├── add_compiler_dialect_plugin.md
│   │   ├── add_runtime_hal_driver.md
│   │   ├── add_sample_application.md
│   │   ├── index.md
│   │   └── use_build_py.md
│   ├── index.md
│   ├── iree_setup.md
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
│   └── stylesheets
│       └── extra.css
├── env_linux.yml
├── env_macOS.yml
├── iree_compiler_plugin.cmake
├── iree_runtime_plugin.cmake
├── mkdocs.yml
├── models
│   ├── compile_spacemit.sh
│   ├── depth_anything_v2
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
│   ├── mobilenet_v2
│   │   └── mobilenet_v2.mlir
│   ├── models_config.json
│   ├── npu_ucb.yaml
│   ├── quantize_models.py
│   ├── saturn_opu.yaml
│   ├── smolVLA
│   │   ├── README.md
│   │   ├── export_smolvla.py
│   │   └── export_smolvla_int8.py
│   ├── spacemit_x60.yaml
│   └── tinydepth
│       └── tinydepth.py
├── projects
│   ├── mlirAgent
│   └── xpu-rt
│       ├── CMakeLists.txt
│       ├── CMakeLists_standalone.cmake
│       ├── xpurt_scheduler_backend_iree.c
│       ├── xpurt_scheduler_core.c
│       └── xpurt_scheduler_core.h
├── pyproject.toml
├── requirements.txt
├── runtime
│   └── src
│       └── iree
├── samples
│   ├── CMakeLists.txt
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
│   │   └── runtime
│   └── research
│       ├── mlir_pipelining
│       ├── model_computation_graph_generation
│       ├── promise_devices_layer
│       └── promise_schedule_multi_model
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
│   ├── analyze_quant_ir.py
│   ├── benchmark.py
│   ├── build.py
│   ├── ci.py
│   ├── compile.py
│   ├── install_prebuilt.py
│   ├── merlin.py
│   ├── patches.py
│   ├── setup.py
│   └── utils.py
└── uv.lock
```

This tree is generated from `git ls-files` so it reflects tracked repository state.
