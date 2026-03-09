set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::gemmini::compiler::plugin")

iree_cc_library(
  NAME
  registration
  HDRS
  "GemminiOptions.h"
  SRCS
  "PluginRegistration.cpp"
  "GemminiOptions.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::compiler::PluginAPI
  iree::compiler::Utils
  iree::gemmini::compiler::Dialect::Gemmini::IR::GemminiDialect
  iree::gemmini::compiler::Dialect::Gemmini::Transforms
  MLIRFuncDialect
  MLIRIR
  MLIRTransforms
  PUBLIC)

iree_compiler_register_plugin(PLUGIN_ID gemmini TARGET ::registration)
