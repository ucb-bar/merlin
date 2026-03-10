# CMake Targets

This inventory is generated from repository `CMakeLists.txt` files.

| Kind | Name | Declared In |
| --- | --- | --- |
| `iree_cc_library` | `registration` | `compiler/plugins/target/Gemmini/CMakeLists.txt` |
| `iree_compiler_register_plugin` | `gemmini -> ::registration` | `compiler/plugins/target/Gemmini/CMakeLists.txt` |
| `add_library` | `merlin::defs` | `compiler/src/merlin/CMakeLists.txt` |
| `add_library` | `merlin_defs` | `compiler/src/merlin/CMakeLists.txt` |
| `iree_cc_library` | `StreamTransformsOverrides` | `compiler/src/merlin/Codegen/Dialect/Stream/Transforms/CMakeLists.txt` |
| `iree_cc_library` | `Overrides` | `compiler/src/merlin/Codegen/ExternalInterfaces/CMakeLists.txt` |
| `iree_cc_library` | `Overrides` | `compiler/src/merlin/Codegen/LLVMCPU/CMakeLists.txt` |
| `iree_cc_library` | `GemminiDialect` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `GemminiAttrsGen` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `GemminiDialectGen` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_tablegen_library` | `GemminiOpsGen` | `compiler/src/merlin/Dialect/Gemmini/IR/CMakeLists.txt` |
| `iree_cc_library` | `PassHeaders` | `compiler/src/merlin/Dialect/Gemmini/Transforms/CMakeLists.txt` |
| `iree_cc_library` | `Transforms` | `compiler/src/merlin/Dialect/Gemmini/Transforms/CMakeLists.txt` |
| `iree_tablegen_library` | `PassesIncGen` | `compiler/src/merlin/Dialect/Gemmini/Transforms/CMakeLists.txt` |
| `add_custom_target` | `compile_custom_model` | `samples/SaturnOPU/custom_dispatch_ukernels/CMakeLists.txt` |
| `add_executable` | `${_NAME}` | `samples/SaturnOPU/custom_dispatch_ukernels/CMakeLists.txt` |
| `add_custom_target` | `riscv_qgemm_object` | `samples/SaturnOPU/custom_dispatch_ukernels/dispatch_opu_m2_gemm/CMakeLists.txt` |
| `add_executable` | `${_NAME}` | `samples/SpacemiTX60/baseline_dual_model_async/CMakeLists.txt` |
