# QNN HAL target backend (compiler-side).
#
# Registers iree-compile target backend "qnn" + target device "qnn" so MLIR
# inputs can declare #hal.device.target<"qnn", ...> and dispatches can be routed
# there via stream.affinity. See QNNTarget.cpp for the lookup-based
# serialization model.

iree_compiler_register_plugin(PLUGIN_ID hal_target_qnn TARGET ::QNN)

iree_cc_library(
  NAME
  QNN
  SRCS
  "QNNTarget.cpp"
  "Codegen/SerializeGraph.cpp"
  DEPS
  LLVMSupport
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  iree::compiler::Dialect::HAL::IR
  iree::compiler::Dialect::HAL::Target
  iree::compiler::Dialect::TensorExt::IR
  iree::compiler::PluginAPI
  iree::compiler::Utils
  iree::merlin::compiler::Dialect::QNN::IR::QNNDialect
  iree::merlin::compiler::Dialect::QNN::Transforms::Transforms
  PUBLIC)
