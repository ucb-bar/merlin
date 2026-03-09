# Runtime plugin entry point for Merlin's xpu-rt runtime wrapper.
#
# This file is discovered by IREE via -DIREE_CMAKE_PLUGIN_PATHS=<merlin root>
# (see merlin/tools/build.py). It is included from within the IREE runtime/
# CMake context and can freely reference IREE runtime targets such as
# iree_runtime_runtime.
#
# We forward to a dedicated subdirectory that defines the xpurt_iree_plugin
# static library.

get_filename_component(MERLIN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

add_subdirectory(
  "${MERLIN_SOURCE_DIR}/xpu-rt"
  "${IREE_BINARY_DIR}/runtime/plugins/xpu-rt"
)

# Merlin runtime plugin entrypoint used by IREE plugin CMake integration.
# Keep this intentionally minimal and only register Merlin-owned sample/runtime
# targets from this repository.

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/samples" "merlin-samples")
#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/benchmarks" "merlin-benchmarks")