# CMakeLists.txt — Merlin project root
#
# This file provides project-level options.  When building through IREE
# (the primary path), set these options on the CMake command line:
#   cmake ... -DMERLIN_ENABLE_CUDA_TILE=ON

option(MERLIN_ENABLE_CUDA_TILE
  "Enable cuda-tile subprocess integration (builds cuda-tile with its own LLVM)" OFF)

if(MERLIN_ENABLE_CUDA_TILE)
  include(cmake/CudaTile.cmake)
endif()
