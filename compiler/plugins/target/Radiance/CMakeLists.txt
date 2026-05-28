set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::radiance::compiler::plugin")

iree_cc_library(
  NAME
  registration
  HDRS
  "RadianceOptions.h"
  SRCS
  "PluginRegistration.cpp"
  "RadianceOptions.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::compiler::PluginAPI
  iree::compiler::Utils
  iree::radiance::compiler::Dialect::Radiance::IR::RadianceDialect
  iree::radiance::compiler::Dialect::Radiance::Transforms
  MLIRFuncDialect
  MLIRIR
  MLIRTransforms
  PUBLIC)

iree_compiler_register_plugin(PLUGIN_ID radiance TARGET ::registration)
