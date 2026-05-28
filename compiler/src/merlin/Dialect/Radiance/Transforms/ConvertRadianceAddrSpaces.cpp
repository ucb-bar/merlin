// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ConvertRadianceAddrSpacesPass — pre-conversion pass that walks every op
// in the module and rewrites memref types bearing a #radiance.global /
// #radiance.shared memorySpace attribute into a memref with a plain
// IntegerAttr memorySpace (matching what the standard memref-to-llvm
// conversion expects).
//
// Rationale: the standard MLIR LLVMTypeConverter handles memref types with
// integer memorySpace values (becoming !llvm.ptr<addrspace>) but does not
// know about our dialect-private AddrSpaceAttr. By rewriting eagerly here,
// we avoid having to fork the LLVMTypeConverter and can use the upstream
// conversion patterns unchanged.

#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.h"
#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceDialect.h"
#include "compiler/src/merlin/Dialect/Radiance/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::iree_compiler::Radiance;

namespace {

// Returns the standard memref equivalent of `mr` with the dialect-private
// memory space replaced by an integer addrspace. Returns null if no
// rewrite is needed.
static MemRefType rewriteMemRefType(MemRefType mr) {
	auto memSpace = mr.getMemorySpace();
	auto asAttr = dyn_cast_or_null<AddrSpaceAttr>(memSpace);
	if (!asAttr)
		return nullptr;

	// Map dialect-private addrspaces to LLVM integer addrspaces using the
	// Muon ABI convention (radiance-kernels/lib/include/shared_mem.h):
	//   __global -> addrspace(0)   (DRAM / ext_gpu_mem)
	//   __shared -> addrspace(1)   (cluster L0/L1 SMEM)
	// The enum's stored values (Global=1, Shared=3) are an internal
	// convention and must NOT leak into emitted LLVM IR — pointers in
	// LLVM IR get hardware-routed by the cluster's smem fanout based on
	// addrspace bits, so a wrong mapping silently sends global accesses
	// to the SMEM Get-less fanout (caught by TLMonitor at runtime).
	int64_t addrspace;
	switch (asAttr.getValue()) {
		case AddrSpace::Global:
			addrspace = 0;
			break;
		case AddrSpace::Shared:
			addrspace = 1;
			break;
	}
	auto i64 = IntegerType::get(mr.getContext(), 64);
	auto newSpace = IntegerAttr::get(i64, addrspace);
	return MemRefType::get(
		mr.getShape(), mr.getElementType(), mr.getLayout(), newSpace);
}

class ConvertRadianceAddrSpacesPass
	: public PassWrapper<ConvertRadianceAddrSpacesPass,
		  OperationPass<ModuleOp>> {
  public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertRadianceAddrSpacesPass)

	StringRef getArgument() const final {
		return "radiance-convert-addrspaces";
	}
	StringRef getDescription() const final {
		return "Rewrite #radiance.global/#radiance.shared memref attrs "
			   "to integer addrspaces consumable by MemRef-to-LLVM.";
	}

	void runOnOperation() override {
		ModuleOp module = getOperation();
		auto walkResult = module.walk([&](Operation *op) {
			// Operand types are propagated implicitly via use-def; we
			// only need to rewrite SSA value types and block argument
			// types. Walk function-like ops + blocks below.
			(void)op;
			return WalkResult::advance();
		});
		(void)walkResult;

		// 1. Rewrite block argument types.
		module.walk([&](Block *block) {
			for (BlockArgument arg : block->getArguments()) {
				auto mr = dyn_cast<MemRefType>(arg.getType());
				if (!mr)
					continue;
				MemRefType newTy = rewriteMemRefType(mr);
				if (newTy)
					arg.setType(newTy);
			}
		});

		// 2. Rewrite function signatures. Use FunctionOpInterface so we
		// catch both func.func and IREE's util.func (which wraps func.func
		// after early input conversion).
		module.walk([&](FunctionOpInterface fn) {
			Type fnType = fn.getFunctionType();
			auto ft = dyn_cast<FunctionType>(fnType);
			if (!ft)
				return;
			SmallVector<Type> inputs, results;
			bool changed = false;
			for (Type t : ft.getInputs()) {
				if (auto mr = dyn_cast<MemRefType>(t)) {
					if (auto nt = rewriteMemRefType(mr)) {
						inputs.push_back(nt);
						changed = true;
						continue;
					}
				}
				inputs.push_back(t);
			}
			for (Type t : ft.getResults()) {
				if (auto mr = dyn_cast<MemRefType>(t)) {
					if (auto nt = rewriteMemRefType(mr)) {
						results.push_back(nt);
						changed = true;
						continue;
					}
				}
				results.push_back(t);
			}
			if (changed) {
				fn.setType(FunctionType::get(fn.getContext(), inputs, results));
			}
		});

		// 3. Rewrite remaining op result types in-place.
		module.walk([&](Operation *op) {
			for (Value res : op->getResults()) {
				auto mr = dyn_cast<MemRefType>(res.getType());
				if (!mr)
					continue;
				if (auto nt = rewriteMemRefType(mr))
					res.setType(nt);
			}
		});
	}
};

} // namespace

namespace mlir::iree_compiler::Radiance {
std::unique_ptr<Pass> createConvertRadianceAddrSpacesPass() {
	return std::make_unique<ConvertRadianceAddrSpacesPass>();
}
} // namespace mlir::iree_compiler::Radiance
