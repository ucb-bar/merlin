set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::cuda_tile::compiler::plugin")

iree_cc_library(
  NAME
    registration
  HDRS
    "CudaTileOptions.h"
  SRCS
    "CudaTileTarget.cpp"
    "CudaTileOptions.cpp"
  INCLUDES
    "${MERLIN_COMPILER_SOURCE_DIR}"
  DEPS
    iree::compiler::PluginAPI
    iree::compiler::Utils
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::HAL::Utils::ExecutableDebugInfoUtils
    iree::base::internal::flatcc::building
    iree::schemas::cuda_executable_def_c_fbs
    iree::schemas::executable_debug_info_c_fbs
    MLIRPass
    MLIRSupport
    LLVMSupport
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    hal_target_cuda_tile
  TARGET
    ::registration
)
