set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX
    "iree::gemmini::compiler::Target::LLVMIR::Dialect::Gemmini")

iree_cc_library(
  NAME
  GemminiToLLVMIRTranslation
  HDRS
  "GemminiToLLVMIRTranslation.h"
  SRCS
  "GemminiToLLVMIRTranslation.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::gemmini::compiler::Dialect::Gemmini::IR::GemminiDialect
  iree::gemmini::compiler::Dialect::Gemmini::IR::GemminiConversionsGen
  LLVMCore
  LLVMSupport
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRSupport
  MLIRTargetLLVMIRExport
  PUBLIC)
