# compile/src/merlin/Codegen/CudaTile/CMakeLists.txt
#
# Builds the Merlin cuda-tile codegen pass and registers it as an IREE
# compiler plugin.  This is compiled as part of the IREE build (via
# iree_compiler_plugin.cmake) so all MLIR/LLVM targets are available.
# It does NOT depend on any cuda-tile headers or libraries.

iree_compiler_register_plugin(
  PLUGIN_ID
    merlin_cuda_tile
  TARGET
    ::CudaTilePlugin
)

iree_cc_library(
  NAME
    CudaTileCodegen
  HDRS
    "LinalgToCudaTileText.h"
  SRCS
    "LinalgToCudaTileText.cpp"
  DEPS
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRIR
    MLIRPass
    MLIRSupport
    LLVMSupport
  PUBLIC
)

iree_cc_library(
  NAME
    CudaTilePlugin
  SRCS
    "CudaTilePlugin.cpp"
  DEPS
    ::CudaTileCodegen
    iree::compiler::PluginAPI
  PUBLIC
)

iree_add_all_subdirs()
