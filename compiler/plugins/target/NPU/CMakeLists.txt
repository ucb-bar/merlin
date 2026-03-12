set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::npu::compiler::plugin")

iree_cc_library(
  NAME
  registration
  HDRS
  "NPUOptions.h"
  SRCS
  "PluginRegistration.cpp"
  "NPUOptions.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::compiler::PluginAPI
  iree::compiler::Utils
  iree::npu::compiler::Dialect::NPU::IR::NPUKernelDialect
  iree::npu::compiler::Dialect::NPU::IR::NPUScheduleDialect
  iree::npu::compiler::Dialect::NPU::IR::NPUISADialect
  iree::npu::compiler::Dialect::NPU::Transforms::Passes
  MLIRFuncDialect
  MLIRIR
  MLIRTransforms
  PUBLIC)

iree_compiler_register_plugin(PLUGIN_ID npu TARGET ::registration)
