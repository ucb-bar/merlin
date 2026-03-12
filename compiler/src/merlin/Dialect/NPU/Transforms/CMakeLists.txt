set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::npu::compiler::Dialect::NPU::Transforms")

iree_cc_library(
  NAME
  Passes
  HDRS
  "Passes.h"
  SRCS
  "Passes.cpp"
  "ConvertLinalgToNPUKernel.cpp"
  "ConvertNPUKernelToSchedule.cpp"
  "VerifyNPUUkernelSymbols.cpp"
  "ConvertNPUScheduleToISA.cpp"
  "PlanNPUISAMemory.cpp"
  INCLUDES
  "${MERLIN_COMPILER_SOURCE_DIR}"
  "${IREE_BINARY_DIR}/compiler/plugins/merlin"
  DEPS
  iree::npu::compiler::Dialect::NPU::IR::NPUKernelDialect
  iree::npu::compiler::Dialect::NPU::IR::NPUScheduleDialect
  iree::npu::compiler::Dialect::NPU::IR::NPUISADialect
  LLVMSupport
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  PUBLIC)

add_subdirectory(tests)
