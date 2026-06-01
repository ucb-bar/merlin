# merlin — root CMake configuration (SCAFFOLD).
#
# This file documents the intended MLIR/C++ structure for the stable compiler plane.
# It deliberately does NOT require a full LLVM/MLIR build yet: the project's default
# plane during prototyping is xDSL (Python). Promote dialects to C++ here only once
# they stabilize (see docs/dialects.md, docs/xdsl.md).
#
# To enable the C++ plane later:
#   1. Provide an LLVM/MLIR build (e.g. -DMLIR_DIR=/path/to/lib/cmake/mlir).
#   2. Flip MERLIN_ENABLE_COMPILER to ON.
#   3. Wire the add_subdirectory() calls below.

cmake_minimum_required(VERSION 3.20)
project(merlin LANGUAGES CXX)

option(MERLIN_ENABLE_COMPILER "Build the C++/MLIR compiler plane (requires LLVM/MLIR)" OFF)

if(MERLIN_ENABLE_COMPILER)
  # find_package(MLIR REQUIRED CONFIG)
  # add_subdirectory(merlin/compiler/lib)
  # add_subdirectory(merlin/compiler/tools)
  # add_subdirectory(merlin/compiler/tests)
  message(STATUS "merlin: C++ compiler plane requested (not yet wired).")
else()
  message(STATUS "merlin: C++ compiler plane disabled. "
                 "Prototype in xDSL (Python). See docs/xdsl.md.")
endif()
