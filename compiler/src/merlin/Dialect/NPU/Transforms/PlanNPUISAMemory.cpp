#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <algorithm>
#include <deque>

namespace mlir::iree_compiler::NPU {
namespace {

static void setI64Attr(
	Operation *op, StringRef attrName, int64_t value, Builder &builder) {
	op->setAttr(attrName, builder.getI64IntegerAttr(value));
}

struct PlanNPUISAMemoryPass
	: public PassWrapper<PlanNPUISAMemoryPass, OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanNPUISAMemoryPass)

	explicit PlanNPUISAMemoryPass(const NPUMemoryPlannerOptions &options)
		: options(options) {}
	PlanNPUISAMemoryPass() = default;

	StringRef getArgument() const final {
		return "npu-plan-memory";
	}
	StringRef getDescription() const final {
		return "Assign deterministic DMA base addresses/flags for npu_isa ops";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUISA::NPUISADialect>();
	}

	void runOnOperation() override {
		Builder builder(&getContext());

		int64_t flagModulo = std::max<int64_t>(1, options.dmaFlagModulo);
		getOperation().walk([&](FunctionOpInterface funcLike) {
			int64_t nextLoadBase = options.loadBase;
			int64_t nextWeightBase = options.weightBase;
			int64_t nextStoreBase = options.storeBase;
			int64_t nextIssuedFlag = 0;
			std::deque<int64_t> pendingWaitFlags;

			auto issueFlag = [&]() {
				int64_t flag = nextIssuedFlag % flagModulo;
				++nextIssuedFlag;
				pendingWaitFlags.push_back(flag);
				return flag;
			};
			auto consumeWaitFlag = [&]() {
				if (pendingWaitFlags.empty()) {
					return int64_t{0};
				}
				int64_t flag = pendingWaitFlags.front();
				pendingWaitFlags.pop_front();
				return flag;
			};

			funcLike.walk([&](Operation *op) {
				if (auto load = dyn_cast<NPUISA::DmaLoadOp>(op)) {
					setI64Attr(op, "base", nextLoadBase, builder);
					setI64Attr(op, "flag", issueFlag(), builder);
					nextLoadBase += std::max<int64_t>(
						1, static_cast<int64_t>(load.getSize()));
					return;
				}
				if (auto load = dyn_cast<NPUISA::DmaLoadMxu0Op>(op)) {
					setI64Attr(op, "base", nextWeightBase, builder);
					setI64Attr(op, "flag", issueFlag(), builder);
					nextWeightBase += std::max<int64_t>(
						1, static_cast<int64_t>(load.getSize()));
					return;
				}
				if (auto load = dyn_cast<NPUISA::DmaLoadMxu1Op>(op)) {
					setI64Attr(op, "base", nextWeightBase, builder);
					setI64Attr(op, "flag", issueFlag(), builder);
					nextWeightBase += std::max<int64_t>(
						1, static_cast<int64_t>(load.getSize()));
					return;
				}
				if (auto store = dyn_cast<NPUISA::DmaStoreOp>(op)) {
					setI64Attr(op, "base", nextStoreBase, builder);
					setI64Attr(op, "flag", issueFlag(), builder);
					nextStoreBase += std::max<int64_t>(
						1, static_cast<int64_t>(store.getSize()));
					return;
				}
				if (isa<NPUISA::DmaWaitOp>(op)) {
					setI64Attr(op, "flag", consumeWaitFlag(), builder);
				}
			});
		});
	}

	NPUMemoryPlannerOptions options;
};

} // namespace

std::unique_ptr<Pass> createPlanNPUISAMemoryPass(
	const NPUMemoryPlannerOptions &options) {
	return std::make_unique<PlanNPUISAMemoryPass>(options);
}

void registerPlanNPUISAMemoryPass() {
	PassRegistration<PlanNPUISAMemoryPass>();
}

} // namespace mlir::iree_compiler::NPU
