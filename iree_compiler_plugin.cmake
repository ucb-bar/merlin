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

# 1. CudaTile Support (NVIDIA cuda_tile IR → tileiras → cubin)
#    In-process compilation: links cuda-tile dialect + BytecodeWriter directly,
#    builds cuda_tile ops via OpBuilder, serializes to tilebc, then tileiras.
if(MERLIN_ENABLE_TARGET_CUDA_TILE)
  if(NOT MERLIN_ENABLE_CORE)
    message(
      FATAL_ERROR
        "MERLIN_ENABLE_TARGET_CUDA_TILE requires MERLIN_ENABLE_CORE=ON")
  endif()

  # Build the cuda-tile dialect libraries using IREE's LLVM.
  # cuda-tile's cmake supports CUDA_TILE_USE_LLVM_INSTALL_DIR to point at a
  # pre-built LLVM. We point it at IREE's LLVM build directory so that
  # cuda-tile links against the same LLVM/MLIR as the rest of the compiler.
  set(MERLIN_CUDA_TILE_SOURCE_DIR
      "${MERLIN_COMPILER_SOURCE_DIR}/third_party/cuda-tile")
  set(MERLIN_CUDA_TILE_BINARY_DIR
      "${MERLIN_COMPILER_BINARY_ROOT}/third_party/cuda-tile")

  if(EXISTS "${MERLIN_CUDA_TILE_SOURCE_DIR}/CMakeLists.txt")
    # Point cuda-tile at IREE's LLVM build via the pre-installed path.
    # IREE's in-tree build places LLVMConfig.cmake under ${LLVM_BINARY_DIR}/
    # lib/cmake/llvm/ but MLIRConfig.cmake at ${CMAKE_BINARY_DIR}/lib/cmake/
    # mlir/. cuda-tile's cmake expects both under the same prefix, so create
    # a symlink to bridge the gap.
    set(CUDA_TILE_USE_LLVM_INSTALL_DIR "${LLVM_BINARY_DIR}" CACHE PATH
        "Point cuda-tile at IREE's LLVM build" FORCE)
    if(NOT EXISTS "${LLVM_BINARY_DIR}/lib/cmake/mlir/MLIRConfig.cmake"
       AND EXISTS "${CMAKE_BINARY_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
      file(CREATE_LINK "${CMAKE_BINARY_DIR}/lib/cmake/mlir"
           "${LLVM_BINARY_DIR}/lib/cmake/mlir" SYMBOLIC)
    endif()
    set(CUDA_TILE_ENABLE_TESTING OFF CACHE BOOL
        "Disable cuda-tile tests when building inside merlin" FORCE)
    add_subdirectory("${MERLIN_CUDA_TILE_SOURCE_DIR}"
                     "${MERLIN_CUDA_TILE_BINARY_DIR}" EXCLUDE_FROM_ALL)
  else()
    message(WARNING "cuda-tile submodule not found at "
                    "${MERLIN_CUDA_TILE_SOURCE_DIR}; "
                    "CudaTile codegen will not be available")
  endif()

  add_subdirectory(
    "${MERLIN_COMPILER_SOURCE_DIR}/compiler/plugins/target/CudaTile"
    "${MERLIN_COMPILER_BINARY_ROOT}/compiler/target/CudaTile")
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
