#ifndef IREE_NPU_COMPILER_DIALECT_NPU_TRANSFORMS_AUTOCOMP_KERNEL_REGISTRY_H_
#define IREE_NPU_COMPILER_DIALECT_NPU_TRANSFORMS_AUTOCOMP_KERNEL_REGISTRY_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::NPU {

/// Descriptor for a single ISA instruction from an autocomp kernel.
struct AutocompInstruction {
  std::string mnemonic;
  std::map<std::string, int64_t> args;
};

/// A complete autocomp-generated kernel (tile or layer level).
struct AutocompKernel {
  std::string symbol;     // e.g., "npu_uk_matmul_f8E4M3FN_f8E4M3FN_bf16"
  std::string kernelType; // e.g., "matmul", "gemma_mlp_layer"
  int latencyCycles = 0;
  std::vector<AutocompInstruction> instructions;
};

/// Registry that loads autocomp-generated kernels from the results directory
/// and provides lookup by ukernel symbol or layer pattern.
class AutocompKernelRegistry {
public:
  /// Load all kernels from the autocomp results directory.
  /// Scans for subdirectories with winning candidates (correct=true).
  void loadFromDirectory(llvm::StringRef resultsDir);

  /// Lookup a kernel by ukernel symbol prefix.
  std::optional<AutocompKernel> lookup(llvm::StringRef symbol) const;

  /// Check if the registry has any loaded kernels.
  bool empty() const { return kernels_.empty(); }

  /// Get the number of loaded kernels.
  size_t size() const { return kernels_.size(); }

private:
  std::map<std::string, AutocompKernel> kernels_;
};

} // namespace mlir::iree_compiler::NPU

#endif // IREE_NPU_COMPILER_DIALECT_NPU_TRANSFORMS_AUTOCOMP_KERNEL_REGISTRY_H_
