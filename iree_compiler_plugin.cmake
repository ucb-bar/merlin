# Copyright 2026 UCB-BAR
#
# This file is included by IREE via -DIREE_CMAKE_PLUGIN_PATHS.
# It uses unified hardware flags to enable compiler-side support.

set(MERLIN_COMPILER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MERLIN_COMPILER_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/merlin")

# --- Core Library ---
# Required by all Merlin target plugins.
if(MERLIN_ENABLE_CORE)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/src/merlin"
                   "${MERLIN_COMPILER_BINARY_ROOT}/src/merlin")
endif()

# --- Target Plugins ---
# We use the same flags defined in build.py and the runtime plugin.

# 1. SpacemiT X60 Support
if(MERLIN_BUILD_SPACEMITX60)
  if(NOT MERLIN_ENABLE_CORE)
    message(FATAL_ERROR "MERLIN_BUILD_SPACEMITX60 requires MERLIN_ENABLE_CORE=ON")
  endif()
  
  if(EXISTS "${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/SpacemiT/CMakeLists.txt")
    add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/SpacemiT"
                     "${MERLIN_COMPILER_BINARY_ROOT}/target/SpacemiT")
  endif()
endif()

# 2. Saturn OPU Support
if(MERLIN_BUILD_SATURN_OPU)
  if(NOT MERLIN_ENABLE_CORE)
    message(FATAL_ERROR "MERLIN_BUILD_SATURN_OPU requires MERLIN_ENABLE_CORE=ON")
  endif()

  if(EXISTS "${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/Saturn/CMakeLists.txt")
    add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/Saturn"
                     "${MERLIN_COMPILER_BINARY_ROOT}/target/Saturn")
  endif()
endif()

# 3. Gemmini Support
# If Gemmini is still used by specific targets (like FireSim), 
# you can gate it with MERLIN_BUILD_GEMMINI or a similar flag.
if(MERLIN_BUILD_GEMMINI)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/Gemmini"
                   "${MERLIN_COMPILER_BINARY_ROOT}/target/Gemmini")
endif()