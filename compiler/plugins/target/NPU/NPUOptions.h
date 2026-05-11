#ifndef IREE_NPU_COMPILER_PLUGIN_NPUOPTIONS_H_
#define IREE_NPU_COMPILER_PLUGIN_NPUOPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

#include <string>

namespace mlir::iree_compiler {

struct NPUOptions {
	bool enable = false;
	bool enableUkernelVerify = true;
	bool strictUkernelVerify = true;
	bool allowUnknownUkernelFallback = true;
	bool matmulUseMxu1Weights = false;
	bool enableMemoryPlanner = false;
	bool nativeKernelLowering = false;
	bool strictNativeKernelCoverage = true;
	std::string kernelManifestPath;
	int64_t dmaFlagModulo = 3;
	int64_t loadBase = 0;
	int64_t weightBase = 0x2000;
	int64_t storeBase = 0x5000;

	void bindOptions(OptionsBinder &binder);
	using FromFlags = OptionsFromFlags<NPUOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_NPU_COMPILER_PLUGIN_NPUOPTIONS_H_
