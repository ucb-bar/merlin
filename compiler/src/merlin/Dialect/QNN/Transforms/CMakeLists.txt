set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::merlin::compiler::Dialect::QNN::Transforms")

iree_cc_library(
  NAME
  Transforms
  HDRS
  "GenericMatchUtils.h"
  "Passes.h"
  SRCS
  "ApplyPlacementRequantization.cpp"
  "ConvertLinalgToQNN.cpp"
  "FoldBodyQDQRoundtrip.cpp"
  "GenericMatchUtils.cpp"
  "InlineConstantUtilGlobals.cpp"
  "LegalizeLayoutToNHWC.cpp"
  "Passes.cpp"
  "RewriteQDQToQuantUniform.cpp"
  "RewriteToNHWCBindings.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::compiler::Dialect::Util::IR
  iree::merlin::compiler::Dialect::QNN::IR::QNNDialect
  LLVMSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRQuantDialect
  MLIRTensorDialect
  MLIRTransforms
  PUBLIC)
