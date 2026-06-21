cmake_minimum_required(VERSION 3.20)
project(gemmini_oot LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# ---- TableGen: merlin_iface dialect ----
set(LLVM_TARGET_DEFINITIONS Iface.td)
mlir_tablegen(IfaceDialect.h.inc -gen-dialect-decls -dialect=merlin_iface)
mlir_tablegen(IfaceDialect.cpp.inc -gen-dialect-defs -dialect=merlin_iface)
mlir_tablegen(IfaceTypes.h.inc -gen-typedef-decls)
mlir_tablegen(IfaceTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(IfaceOps.h.inc -gen-op-decls)
mlir_tablegen(IfaceOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(IfaceIncGen)

# ---- TableGen: gemmini dialect ----
set(LLVM_TARGET_DEFINITIONS Gemmini.td)
mlir_tablegen(GemminiDialect.h.inc -gen-dialect-decls -dialect=gemmini)
mlir_tablegen(GemminiDialect.cpp.inc -gen-dialect-defs -dialect=gemmini)
mlir_tablegen(GemminiOps.h.inc -gen-op-decls)
mlir_tablegen(GemminiOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(GemminiIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(gemmini-opt
  gemmini-opt.cpp
  Dialects.cpp
  Backend.cpp)
add_dependencies(gemmini-opt IfaceIncGen GemminiIncGen)

target_link_libraries(gemmini-opt PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRLLVMDialect)

llvm_update_compile_flags(gemmini-opt)
