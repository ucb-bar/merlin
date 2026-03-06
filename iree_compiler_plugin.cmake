# Copyright 2026 UCB-BAR
#
# This file is included by IREE via -DIREE_CMAKE_PLUGIN_PATHS.
# It uses unified hardware flags to enable compiler-side support.

set(MERLIN_COMPILER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/compiler")
set(MERLIN_COMPILER_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/merlin")

# --- Gemmini Plugin ---
# Disable the in-tree gemmini plugin (from the IREE fork) in favour of
# the one provided here in Merlin.
set(IREE_GEMMINI_EXTERNAL_PLUGIN ON CACHE BOOL "" FORCE)

# Include paths so that #include "merlin/Dialect/Gemmini/..." resolves
# from the source tree.  Generated .inc files are found via the build
# directory (added by the plugin CMakeLists.txt itself).
include_directories(
  "${MERLIN_COMPILER_SOURCE_DIR}/src"
)

# Build the Gemmini plugin (dialect + passes + registration) in one package.
add_subdirectory(
  "${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/Gemmini"
  "${MERLIN_COMPILER_BINARY_ROOT}/target/Gemmini"
)