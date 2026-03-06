# Copyright 2026 UCB-BAR
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# 1. Define a library for the plugin registration code.
#    This compiles PluginRegistration.cpp.
add_library(MerlinGemminiPluginRegistration STATIC
  PluginRegistration.cpp
)

# 2. Link against the necessary dependencies.
#    - IREECompilerPluginAPI: Required for the plugin system.
#    - MerlinDialectGemminiRegistration: Your library that registers the Dialect & LLVM Translation.
#    - MerlinDialectGemminiTransforms: Your library containing the passes (Linalg->Gemmini, etc).
target_link_libraries(MerlinGemminiPluginRegistration
  PUBLIC
    IREECompilerPluginAPI
    MerlinDialectGemminiRegistration
    MerlinDialectGemminiTransforms
    MLIRIR
    MLIRPass
)

# 3. Add the include directory for your Merlin source tree.
#    This allows PluginRegistration.cpp to include "merlin/Dialect/Gemmini/..."
target_include_directories(MerlinGemminiPluginRegistration
  PRIVATE
    "${MERLIN_SOURCE_DIR}/compiler/src"
)

# 4. Register the plugin with the IREE compiler system.
#    - PLUGIN_ID: Must match the name used in your C++ code:
#                 registrar->registerPlugin<GemminiSession>("merlin_gemmini");
#    - TARGET: The library defined above.
iree_compiler_register_plugin(
  PLUGIN_ID
    merlin_gemmini
  TARGET
    MerlinGemminiPluginRegistration
)