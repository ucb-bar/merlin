set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::npu::compiler::Dialect::NPU::Translation")

iree_cc_library(
  NAME
  TextISATranslation
  HDRS
  "TranslateToTextISA.h"
  SRCS
  "TextISATranslate.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::npu::compiler::Dialect::NPU::IR::NPUKernelDialect
  iree::npu::compiler::Dialect::NPU::IR::NPUScheduleDialect
  iree::npu::compiler::Dialect::NPU::IR::NPUISADialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTranslateLib
  PUBLIC)

iree_cc_binary(
  NAME
  npu-translate
  SRCS
  "npu-translate.cpp"
  DEPS
  ::TextISATranslation
  MLIRTranslateLib)
