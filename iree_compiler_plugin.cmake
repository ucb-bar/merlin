# iree_compiler_plugin.cmake — Merlin compiler extensions for IREE.
#
# This file is included by IREE's build system via IREE_CMAKE_PLUGIN_PATHS.
# It registers Merlin-specific compiler passes as IREE plugins so they
# are available in iree-opt, iree-compile, etc.

# cuda-tile text-emission pass (subprocess integration boundary).
add_subdirectory(
  ${CMAKE_CURRENT_LIST_DIR}/compile/src/merlin/Codegen/CudaTile
  merlin-codegen-cuda-tile)
