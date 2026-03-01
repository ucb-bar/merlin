# cmake/CudaTile.cmake — Build cuda-tile as a standalone external project.
#
# cuda-tile ships its own LLVM via FetchContent (cmake/IncludeLLVM.cmake),
# so it builds a completely isolated LLVM/MLIR that never links into Merlin.
# The only integration surface is the installed binaries.

include(ExternalProject)

ExternalProject_Add(cuda-tile-external
  SOURCE_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda-tile"
  BINARY_DIR    "${CMAKE_BINARY_DIR}/cuda-tile-build"
  INSTALL_DIR   "${CMAKE_BINARY_DIR}/cuda-tile-install"
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCUDA_TILE_ENABLE_TESTING=OFF
    -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=OFF
  # Only build the tools we need — no need for the full library set.
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target cuda-tile-translate cuda-tile-opt
  INSTALL_COMMAND
    ${CMAKE_COMMAND} --install <BINARY_DIR>
)

# --------------------------------------------------------------------------- #
# Exported cache variables — consumed by merlin_cuda_tile_compile() and by
# downstream users who want to invoke the tools directly.
# --------------------------------------------------------------------------- #
set(CUDA_TILE_TRANSLATE_BIN
  "${CMAKE_BINARY_DIR}/cuda-tile-install/bin/cuda-tile-translate"
  CACHE FILEPATH "Path to cuda-tile-translate binary")

set(CUDA_TILE_OPT_BIN
  "${CMAKE_BINARY_DIR}/cuda-tile-install/bin/cuda-tile-opt"
  CACHE FILEPATH "Path to cuda-tile-opt binary")

# tileiras is NOT built by cuda-tile — it comes from conda (cuda-tileiras pkg)
# or a user-provided path.  Auto-detect from conda if not explicitly set.
find_program(_tileiras_default tileiras)
set(CUDA_TILE_TILEIRAS_BIN "${_tileiras_default}" CACHE FILEPATH
  "Path to tileiras binary (auto-detected from PATH or user-provided)")

# --------------------------------------------------------------------------- #
# merlin_cuda_tile_compile()
#
#   merlin_cuda_tile_compile(
#     TARGET    my_cubin_target
#     MLIR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/kernel.mlir
#     OUTPUT    ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
#     GPU_NAME  sm_120          # optional, defaults to sm_120
#   )
#
# Adds a custom command that:
#   1. cuda-tile-translate --cuda-tile-to-bytecode input -o tmp.tilebc
#   2. tileiras --gpu-name <GPU> tmp.tilebc -o output.cubin
# --------------------------------------------------------------------------- #
function(merlin_cuda_tile_compile)
  cmake_parse_arguments(ARG "" "TARGET;MLIR_FILE;OUTPUT;GPU_NAME" "" ${ARGN})

  if(NOT ARG_MLIR_FILE)
    message(FATAL_ERROR "merlin_cuda_tile_compile: MLIR_FILE is required")
  endif()
  if(NOT ARG_OUTPUT)
    message(FATAL_ERROR "merlin_cuda_tile_compile: OUTPUT is required")
  endif()
  if(NOT CUDA_TILE_TILEIRAS_BIN)
    message(FATAL_ERROR
      "CUDA_TILE_TILEIRAS_BIN must be set to use merlin_cuda_tile_compile()")
  endif()

  if(NOT ARG_GPU_NAME)
    set(ARG_GPU_NAME "sm_120")
  endif()

  get_filename_component(_base "${ARG_MLIR_FILE}" NAME_WE)
  set(_tmp "${CMAKE_CURRENT_BINARY_DIR}/${_base}.tilebc")

  add_custom_command(
    OUTPUT  "${ARG_OUTPUT}"
    COMMAND "${CUDA_TILE_TRANSLATE_BIN}"
            --mlir-to-cudatilebc --no-implicit-module --bytecode-version=13.1
            "${ARG_MLIR_FILE}" -o "${_tmp}"
    COMMAND "${CUDA_TILE_TILEIRAS_BIN}"
            --gpu-name "${ARG_GPU_NAME}" "${_tmp}" -o "${ARG_OUTPUT}"
    DEPENDS "${ARG_MLIR_FILE}" cuda-tile-external
    COMMENT "cuda-tile: ${ARG_MLIR_FILE} -> ${ARG_OUTPUT}"
    VERBATIM
  )

  if(ARG_TARGET)
    add_custom_target(${ARG_TARGET} DEPENDS "${ARG_OUTPUT}")
  endif()
endfunction()
