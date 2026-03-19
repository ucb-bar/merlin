# CMake Targets

This inventory is generated from repository `CMakeLists.txt` files.

| Kind | Name | Declared In |
| --- | --- | --- |
| `iree_cc_library` | `registration` | `compiler/plugins/target/Gemmini/CMakeLists.txt` |
| `iree_compiler_register_plugin` | `gemmini -> ::registration` | `compiler/plugins/target/Gemmini/CMakeLists.txt` |
| `iree_cc_library` | `registration` | `compiler/plugins/target/NPU/CMakeLists.txt` |
| `iree_compiler_register_plugin` | `npu -> ::registration` | `compiler/plugins/target/NPU/CMakeLists.txt` |
| `add_library` | `merlin::defs` | `compiler/src/merlin/CMakeLists.txt` |
| `add_library` | `merlin_defs` | `compiler/src/merlin/CMakeLists.txt` |
| `iree_cc_library` | `GemminiDialect` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `GemminiAttrsGen` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `GemminiDialectGen` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `GemminiOpsGen` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_cc_library` | `PassHeaders` | `compiler/src/merlin/Dialect/Gemmini/Transforms/CMakeLists.txt` |
| `iree_cc_library` | `Transforms` | `compiler/src/merlin/Dialect/Gemmini/Transforms/CMakeLists.txt` |
| `iree_tablegen_library` | `PassesIncGen` | `compiler/src/merlin/Dialect/Gemmini/Transforms/CMakeLists.txt` |
| `iree_cc_library` | `NPUISADialect` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_cc_library` | `NPUKernelDialect` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_cc_library` | `NPUScheduleDialect` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `NPUISADialectGen` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `NPUISAOpsGen` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `NPUKernelDialectGen` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `NPUKernelOpsGen` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `NPUScheduleDialectGen` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `NPUScheduleOpsGen` | `compiler/src/merlin/Dialect/NPU/IR/CMakeLists.txt` |
| `iree_cc_library` | `Passes` | `compiler/src/merlin/Dialect/NPU/Transforms/CMakeLists.txt` |
| `iree_cc_library` | `TextISATranslation` | `compiler/src/merlin/Dialect/NPU/Translation/CMakeLists.txt` |
| `add_library` | `xpurt_iree_plugin` | `samples/common/xpu-rt/CMakeLists.txt` |
| `add_library` | `xpurt_iree_plugin_objs` | `samples/common/xpu-rt/CMakeLists.txt` |
| `iree_cc_library` | `radiance` | `runtime/src/iree/hal/drivers/radiance/CMakeLists.txt` |
| `iree_cc_library` | `registration` | `runtime/src/iree/hal/drivers/radiance/registration/CMakeLists.txt` |
| `iree_cc_library` | `fake_transport` | `runtime/src/iree/hal/drivers/radiance/testing/CMakeLists.txt` |
| `add_custom_target` | `compile_custom_model` | `samples/SaturnOPU/custom_dispatch_ukernels/CMakeLists.txt` |
| `add_executable` | `${_NAME}` | `samples/SaturnOPU/custom_dispatch_ukernels/CMakeLists.txt` |
| `add_custom_target` | `riscv_qgemm_object` | `samples/SaturnOPU/custom_dispatch_ukernels/dispatch_opu_m2_gemm/CMakeLists.txt` |
| `add_custom_target` | `benchmark_suite` | `samples/SaturnOPU/simple_embedding_ukernel/CMakeLists.txt` |
| `add_executable` | `${_NAME}` | `samples/SpacemiTX60/baseline_async/CMakeLists.txt` |
| `add_executable` | `${_NAME}` | `samples/SpacemiTX60/dispatch_scheduler/CMakeLists.txt` |
| `add_library` | `merlin_core` | `samples/common/core/CMakeLists.txt` |
| `add_library` | `merlin_dispatch` | `samples/common/dispatch/CMakeLists.txt` |
| `add_library` | `merlin_runtime` | `samples/common/runtime/CMakeLists.txt` |
