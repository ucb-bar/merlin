set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::merlin::compiler::Dialect::QNN::IR")

iree_tablegen_library(
  NAME
  QNNDialectGen
  TD_FILE
  "QNNDialect.td"
  OUTS
  --gen-dialect-decls
  QNNDialect.h.inc
  --gen-dialect-defs
  QNNDialect.cpp.inc)

iree_tablegen_library(
  NAME
  QNNOpsGen
  TD_FILE
  "QNNOps.td"
  OUTS
  -gen-op-decls
  QNNOps.h.inc
  -gen-op-defs
  QNNOps.cpp.inc)

iree_cc_library(
  NAME
  QNNDialect
  HDRS
  "QNNDialect.h"
  TEXTUAL_HDRS
  "QNNDialect.h.inc"
  "QNNDialect.cpp.inc"
  "QNNOps.h.inc"
  "QNNOps.cpp.inc"
  SRCS
  "QNNDialect.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  ::QNNDialectGen
  ::QNNOpsGen
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  PUBLIC)
