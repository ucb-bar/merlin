set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::radiance::compiler::Dialect::Radiance::IR")

iree_tablegen_library(
  NAME
  RadianceDialectGen
  TD_FILE
  "RadianceDialect.td"
  OUTS
  --gen-dialect-decls
  RadianceDialect.h.inc
  --gen-dialect-defs
  RadianceDialect.cpp.inc)

iree_tablegen_library(
  NAME
  RadianceAttrsGen
  TD_FILE
  "RadianceAttrs.td"
  OUTS
  --gen-attrdef-decls
  --attrdefs-dialect=radiance
  RadianceAttrs.h.inc
  --gen-attrdef-defs
  --attrdefs-dialect=radiance
  RadianceAttrs.cpp.inc
  --gen-enum-decls
  RadianceEnums.h.inc
  --gen-enum-defs
  RadianceEnums.cpp.inc)

iree_cc_library(
  NAME
  RadianceDialect
  HDRS
  "RadianceDialect.h"
  "RadianceAttrs.h"
  TEXTUAL_HDRS
  "RadianceDialect.h.inc"
  "RadianceDialect.cpp.inc"
  "RadianceAttrs.h.inc"
  "RadianceAttrs.cpp.inc"
  "RadianceEnums.h.inc"
  "RadianceEnums.cpp.inc"
  SRCS
  "RadianceDialect.cpp"
  "RadianceAttrs.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  ::RadianceDialectGen
  ::RadianceAttrsGen
  LLVMSupport
  MLIRIR
  MLIRSupport
  PUBLIC)
