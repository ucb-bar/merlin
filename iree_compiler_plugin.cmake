# Copyright 2026 UCB-BAR
#
# This file is included by IREE via -DIREE_CMAKE_PLUGIN_PATHS. It uses unified
# hardware flags to enable compiler-side support.

set(MERLIN_COMPILER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MERLIN_COMPILER_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/merlin")

# --- Core Library ---
# Required by all Merlin target plugins.
if(MERLIN_ENABLE_CORE)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/compiler/src/merlin"
                   "${MERLIN_COMPILER_BINARY_ROOT}/compiler/src/merlin")
endif()

# --- Target Plugins ---
# We use the same flags defined in build.py and the runtime plugin.

# 1. SpacemiT X60 Support
if(MERLIN_BUILD_SPACEMITX60)
  if(NOT MERLIN_ENABLE_CORE)
    message(
      FATAL_ERROR "MERLIN_BUILD_SPACEMITX60 requires MERLIN_ENABLE_CORE=ON")
  endif()

  if(EXISTS
     "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/SpacemiT/CMakeLists.txt"
  )
    add_subdirectory(
      "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/SpacemiT"
      "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/SpacemiT")
  endif()
endif()

# 1. Saturn OPU Support
if(MERLIN_BUILD_SATURN_OPU)
  if(NOT MERLIN_ENABLE_CORE)
    message(
      FATAL_ERROR "MERLIN_BUILD_SATURN_OPU requires MERLIN_ENABLE_CORE=ON")
  endif()

  if(EXISTS
     "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/Saturn/CMakeLists.txt"
  )
    add_subdirectory(
      "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/Saturn"
      "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/Saturn")
  endif()
endif()

# 1. Gemmini Support Keep backward compatibility with legacy MERLIN_BUILD_GEMMINI
#   while using MERLIN_ENABLE_TARGET_GEMMINI as the primary knob from
#   tools/build.py.
if(MERLIN_ENABLE_TARGET_GEMMINI OR MERLIN_BUILD_GEMMINI)
  add_subdirectory(
    "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/Gemmini"
    "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/Gemmini")
endif()

# 1. Radiance/Muon Support (Phase 2.6). Compile-side plugin for the UCB-BAR Muon
#   GPU. Phase 2.6a scaffolds the plugin (options + empty session); 2.6b adds
#   the dialect; 2.6c adds the LowerRadianceToLLVM pass that emits a
#   kernel_body.ll consumable by tools/kernels/precompile.py with
#   source_lang=ll. The runtime side is the existing kernel-embed manifest
#   pipeline (Phase 2 in this branch).
if(MERLIN_ENABLE_TARGET_RADIANCE OR MERLIN_BUILD_RADIANCE)
  if(NOT MERLIN_ENABLE_CORE)
    message(
      FATAL_ERROR "MERLIN_ENABLE_TARGET_RADIANCE requires MERLIN_ENABLE_CORE=ON"
    )
  endif()
  add_subdirectory(
    "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/Radiance"
    "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/Radiance")
endif()

# 1. NPU Support Keep backward compatibility with legacy MERLIN_BUILD_NPU while
#   using MERLIN_ENABLE_TARGET_NPU as the primary knob from tools/build.py.
if(MERLIN_ENABLE_TARGET_NPU OR MERLIN_BUILD_NPU)
  if(NOT MERLIN_ENABLE_CORE)
    message(
      FATAL_ERROR "MERLIN_ENABLE_TARGET_NPU requires MERLIN_ENABLE_CORE=ON")
  endif()

  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/NPU"
                   "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/NPU")
endif()

# QNN compile-side target. Mirrors the runtime-side
# MERLIN_RUNTIME_ENABLE_HAL_QNN flag so a host iree-compile build can emit
# `#hal.executable.target<"qnn", ...>` variants that pair with the runtime HAL
# driver. The compile-side plugin is small (lookup-based serializer) and has no
# toolchain dependencies, so we wire it under a dedicated flag that defaults ON
# when the runtime QNN HAL is enabled.
option(MERLIN_BUILD_QNN_TARGET
       "Enable Merlin QNN compiler target backend (lookup-based serializer)"
       OFF)
if(MERLIN_RUNTIME_ENABLE_HAL_QNN)
  set(MERLIN_BUILD_QNN_TARGET ON)
endif()

if(MERLIN_BUILD_QNN_TARGET)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/QNN"
                   "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/QNN")
endif()
