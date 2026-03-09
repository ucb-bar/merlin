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
