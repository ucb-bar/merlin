#ifndef IREE_GEMMINI_COMPILER_PLUGIN_GEMMINIOPTIONS_H_
#define IREE_GEMMINI_COMPILER_PLUGIN_GEMMINIOPTIONS_H_

#include <string>

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

struct GemminiOptions {
	bool enable = false;

	bool lowerBackToIREE = true;

	bool enableMatmul = true;
	bool enableConv2D = true;
	bool enableRequantize = true;
	bool enableClamp = true;

	std::string dataflow = "os";

	int64_t tileM = 16;
	int64_t tileN = 16;
	int64_t tileK = 16;

	void bindOptions(OptionsBinder &binder);
	using FromFlags = OptionsFromFlags<GemminiOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_GEMMINI_COMPILER_PLUGIN_GEMMINIOPTIONS_H_
