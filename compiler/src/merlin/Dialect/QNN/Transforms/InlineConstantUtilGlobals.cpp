// Inline constant-initialized `util.global` tensors.
//
// IREE's frontend (torch-mlir + global-optimization) hoists every conv weight
// and bias tensor into a module-level `util.global private @__constant_*`
// (or `__hoisted_*`) with `inlining_policy = #util.inline.never`. The
// downstream `util.global.load` becomes a `flow.dispatch` operand, which the
// dispatch turns into an external binding (APP_WRITE at QNN runtime).
//
// QNN HTA's Conv2d validator demands `QNN_TENSOR_TYPE_STATIC` weights with
// inline data — APP_WRITE bindings are rejected at `graphAddNode rc=6000`
// (HMX weight-layout precondition). The compiler emits STATIC weights via
// `SerializeGraph::extractConstantBytes`, which only fires when the conv
// weight operand inside the dispatch body is an `arith.constant` directly.
//
// This pass runs at preprocessing phase — BEFORE dispatch creation — so the
// materialized constants live in func.func scope. IREE's dispatch-creation
// clone-into-region step then pulls them inside the dispatch body (provided
// `iree-flow-inline-constants-max-byte-length` is high enough; the QNN
// session bumps the default 256B → 16MiB).
//
// Strictly only inlines globals whose `initial_value` is an ElementsAttr or
// other constant-materializable attribute. The `!hal.device` device globals
// have `#device_target_local` style attrs that won't match — preserved.

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler::QNN {
namespace {

// Returns true if `attr` is a tensor/elements constant that
// `arith.constant` can materialize as its value attribute.
static bool isInlinableConstantAttr(Attribute attr) {
	if (!attr)
		return false;
	// Most yolov8 constants come through as DenseElementsAttr /
	// DenseResourceElementsAttr.
	if (isa<ElementsAttr>(attr))
		return true;
	// Allow plain int/float scalars too, in case any hoisted globals are
	// rank-0 scalars.
	if (isa<IntegerAttr, FloatAttr>(attr))
		return true;
	return false;
}

struct InlineConstantUtilGlobalsPass
	: public PassWrapper<InlineConstantUtilGlobalsPass,
		  OperationPass<ModuleOp>> {
	StringRef getArgument() const final {
		return "merlin-qnn-inline-constant-util-globals";
	}
	StringRef getDescription() const final {
		return "Replace every `util.global.load` of a constant-initialized "
			   "`util.global` tensor with an inline `arith.constant`, then "
			   "erase "
			   "the global. Required for QNN HTA conv weights so "
			   "SerializeGraph "
			   "embeds them as STATIC tensors instead of APP_WRITE bindings.";
	}
	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<arith::ArithDialect, IREE::Util::UtilDialect>();
	}

	void runOnOperation() override {
		ModuleOp module = getOperation();
		bool dbg = ::getenv("MERLIN_INLINE_GLOBALS_DBG") != nullptr;

		// Step 1: collect candidate globals — those with a
		// constant-materializable `initial_value`. Index by symbol name for
		// fast lookup at load sites.
		llvm::DenseMap<StringRef, IREE::Util::GlobalOp> candidates;
		for (auto globalOp : module.getOps<IREE::Util::GlobalOp>()) {
			Attribute init = globalOp.getInitialValueAttr();
			if (!isInlinableConstantAttr(init))
				continue;
			candidates.try_emplace(globalOp.getSymName(), globalOp);
		}
		if (candidates.empty())
			return;

		if (dbg) {
			llvm::errs() << "[inline-globals-dbg] candidates: "
						 << candidates.size() << "\n";
		}

		// Step 2: walk every util.global.load. If it targets a candidate,
		// replace the result with a freshly-materialized arith.constant at the
		// load site, then erase the load.
		SmallVector<IREE::Util::GlobalLoadOp> stalLoads;
		module.walk([&](IREE::Util::GlobalLoadOp loadOp) {
			StringRef name = loadOp.getGlobalAttr().getValue();
			auto it = candidates.find(name);
			if (it == candidates.end())
				return;
			IREE::Util::GlobalOp globalOp = it->second;
			Attribute init = globalOp.getInitialValueAttr();
			if (!isInlinableConstantAttr(init))
				return;
			auto typedInit = dyn_cast<TypedAttr>(init);
			if (!typedInit)
				return;
			OpBuilder b(loadOp);
			auto constantOp = arith::ConstantOp::create(
				b, loadOp.getLoc(), loadOp.getType(), typedInit);
			loadOp.getResult().replaceAllUsesWith(constantOp.getResult());
			stalLoads.push_back(loadOp);
		});
		for (auto loadOp : stalLoads)
			loadOp.erase();

		if (dbg) {
			llvm::errs() << "[inline-globals-dbg] erased loads: "
						 << stalLoads.size() << "\n";
		}

		// Step 3: erase globals whose loads are all gone. Use SymbolTable
		// user iteration so we don't accidentally drop a global that still has
		// other references (initializers, address ops, etc.).
		SymbolTable symbolTable(module);
		SymbolTableCollection collection;
		SmallVector<IREE::Util::GlobalOp> toErase;
		for (auto &kv : candidates) {
			IREE::Util::GlobalOp globalOp = kv.second;
			auto uses = SymbolTable::getSymbolUses(globalOp, module);
			if (!uses)
				continue;
			if (uses->begin() == uses->end()) {
				toErase.push_back(globalOp);
			}
		}
		for (auto globalOp : toErase)
			globalOp.erase();

		if (dbg) {
			llvm::errs() << "[inline-globals-dbg] erased globals: "
						 << toErase.size() << "\n";
		}
	}
};

} // namespace

std::unique_ptr<Pass> createInlineConstantUtilGlobalsPass() {
	return std::make_unique<InlineConstantUtilGlobalsPass>();
}

} // namespace mlir::iree_compiler::QNN
