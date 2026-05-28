// Copyright 2026 The Merlin Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ApplyPlacementRequantization: schedule-driven re-quantization (Phase A2).
//
// The XPU-RT heterogeneous loop (XPU-RT/scripts/heterogeneous_loop.py) emits
// a `placement_requant.json` sidecar each round describing which dispatches
// need a dtype shift to match their assigned target (e.g., HTA requires
// int8, GPU prefers fp16). This pass consumes that sidecar via the
// `--merlin-placement-requant-json=<path>` flag and inserts
// quant.qcast / quant.dcast round-trip pairs around the source-level ops
// whose corresponding dispatch is marked for re-quantization.
//
// Identity matching: at this phase (pre-dispatch-creation) we don't yet
// have dispatch_K names. We walk the function's "anchor" ops in order
// (linalg conv / linalg.generic with reduction) and assume the K-th
// anchor becomes the K-th dispatch. This holds for the IREE dispatch-
// creation default (each compute-heavy op becomes one dispatch) and for
// the fixtures we currently use; networks with fusion may need a
// finer-grained matcher later.
//
// Quantization parameters: until tools/calibrate_placements.py lands,
// we use a conservative default (scale=0.0625, zero_point=0 for i8;
// fp16 needs no scale). Real calibration data should override these via
// the sidecar JSON's optional "scale" / "zero_point" fields.

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include <fstream>
#include <sstream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "merlin-apply-placement-requant"

namespace mlir::iree_compiler::QNN {

namespace {

struct PlacementEntry {
	std::string from_dtype;
	std::string to_dtype;
	std::string machine;
	double scale = 0.0625; // default i8 scale; override via sidecar
	int64_t zero_point = 0;
};

struct Sidecar {
	std::string source_dtype;
	// Indexed by dispatch_K integer (parsed from name suffix).
	llvm::DenseMap<int, PlacementEntry> by_dispatch_id;
};

static FailureOr<Sidecar> ParseSidecar(const std::string &path) {
	std::ifstream f(path);
	if (!f.good())
		return failure();
	std::stringstream ss;
	ss << f.rdbuf();
	auto parsed = llvm::json::parse(ss.str());
	if (!parsed)
		return failure();
	auto *obj = parsed->getAsObject();
	if (!obj)
		return failure();
	Sidecar out;
	if (auto sd = obj->getString("source_dtype"))
		out.source_dtype = sd->str();
	auto *ops = obj->getObject("ops");
	if (!ops)
		return out;
	for (auto &kv : *ops) {
		// Extract dispatch_K id from the canonical name: ".._dispatch_<N>".
		llvm::StringRef name = kv.first;
		auto pos = name.rfind("_dispatch_");
		if (pos == llvm::StringRef::npos)
			continue;
		int id = 0;
		if (name.drop_front(pos + 10).getAsInteger(10, id))
			continue;
		auto *e = kv.second.getAsObject();
		if (!e)
			continue;
		PlacementEntry pe;
		if (auto s = e->getString("from_dtype"))
			pe.from_dtype = s->str();
		if (auto s = e->getString("to_dtype"))
			pe.to_dtype = s->str();
		if (auto s = e->getString("machine"))
			pe.machine = s->str();
		if (auto v = e->getNumber("scale"))
			pe.scale = *v;
		if (auto v = e->getInteger("zero_point"))
			pe.zero_point = *v;
		out.by_dispatch_id[id] = pe;
	}
	return out;
}

// Returns the quant.uniform expressed type for a given to_dtype.
static Type GetQuantizedExpressedType(
	MLIRContext *ctx, const PlacementEntry &pe, Type expressed) {
	if (pe.to_dtype == "i8") {
		return quant::UniformQuantizedType::get(
			/*flags=*/quant::QuantizationFlags::Signed,
			/*storageType=*/IntegerType::get(ctx, 8),
			/*expressedType=*/expressed,
			/*scale=*/pe.scale, /*zeroPoint=*/pe.zero_point,
			/*storageTypeMin=*/-128, /*storageTypeMax=*/127);
	}
	if (pe.to_dtype == "u8") {
		return quant::UniformQuantizedType::get(
			/*flags=*/0, /*storageType=*/IntegerType::get(ctx, 8),
			/*expressedType=*/expressed, /*scale=*/pe.scale,
			/*zeroPoint=*/pe.zero_point, /*storageTypeMin=*/0,
			/*storageTypeMax=*/255);
	}
	return Type();
}

// Wrap one tensor operand with a quant.qcast → quant.dcast round-trip.
// On failure (e.g., non-tensor operand or unsupported dtype), leaves the
// operand unchanged and returns false.
static bool WrapTensorOperand(
	OpBuilder &b, Location loc, Value &operand, const PlacementEntry &pe) {
	auto rt = dyn_cast<RankedTensorType>(operand.getType());
	if (!rt)
		return false;

	// Already in the target dtype? skip.
	auto elt = rt.getElementType();
	if (pe.to_dtype == "i8" && elt.isInteger(8))
		return false;
	if (pe.to_dtype == "u8" && elt.isInteger(8))
		return false;

	// fp16 case: simple arith.truncf+extf round-trip via tensor.empty +
	// linalg.generic would be ideal but is heavyweight. For the first cut
	// we only emit quant.uniform round-trips for int8; fp16 placements
	// fall through unchanged (CPU/GPU both accept fp32, the placement is
	// satisfied without re-quant when source is fp32).
	if (pe.to_dtype == "f16")
		return false;

	if (pe.to_dtype != "i8" && pe.to_dtype != "u8")
		return false;

	auto quant_t = GetQuantizedExpressedType(b.getContext(), pe, elt);
	if (!quant_t)
		return false;
	auto quant_tensor = RankedTensorType::get(rt.getShape(), quant_t);

	auto qcast = quant::QuantizeCastOp::create(b, loc, quant_tensor, operand);
	auto dcast = quant::DequantizeCastOp::create(b, loc, rt, qcast.getResult());
	operand = dcast.getResult();
	return true;
}

// Walk the function's anchor ops in order (linalg conv + linalg.generic
// with at least one reduction). Returns them as a vector in source order.
static SmallVector<Operation *> CollectAnchorOps(FunctionOpInterface fn) {
	SmallVector<Operation *> out;
	fn->walk([&](Operation *op) {
		if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
			// Anchor only the ones that look like compute (have a reduction
			// iterator). Pure-pointwise generics are usually not their own
			// dispatch.
			bool has_reduction = false;
			for (auto it : generic.getIteratorTypesArray()) {
				if (it == utils::IteratorType::reduction) {
					has_reduction = true;
					break;
				}
			}
			if (has_reduction)
				out.push_back(op);
			return WalkResult::advance();
		}
		// Named convs and matmuls.
		if (isa<linalg::Conv2DNhwcHwcfOp>(op) ||
			isa<linalg::Conv2DNchwFchwOp>(op) ||
			isa<linalg::DepthwiseConv2DNhwcHwcOp>(op) ||
			isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op)) {
			out.push_back(op);
		}
		return WalkResult::advance();
	});
	return out;
}

class ApplyPlacementRequantizationPass
	: public PassWrapper<ApplyPlacementRequantizationPass,
		  OperationPass<ModuleOp>> {
  public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
		ApplyPlacementRequantizationPass)

	ApplyPlacementRequantizationPass() = default;
	ApplyPlacementRequantizationPass(
		const ApplyPlacementRequantizationPass &other)
		: PassWrapper(other) {
		sidecar_path_ = other.sidecar_path_.getValue();
	}

	StringRef getArgument() const final {
		return "merlin-apply-placement-requant";
	}
	StringRef getDescription() const final {
		return "Insert quant boundaries at dispatches whose schedule "
			   "placement requires a different dtype than the source IR.";
	}

	void getDependentDialects(DialectRegistry &registry) const final {
		registry.insert<arith::ArithDialect, linalg::LinalgDialect,
			quant::QuantDialect, tensor::TensorDialect, func::FuncDialect>();
	}

	void runOnOperation() final {
		ModuleOp mod = getOperation();
		const std::string &sp = sidecar_path_.getValue();
		if (sp.empty()) {
			// Allow the pass to be a no-op when no sidecar is provided —
			// useful for first-round runs where placements are unknown.
			return;
		}
		auto parsed = ParseSidecar(sp);
		if (failed(parsed)) {
			mod.emitWarning() << "ApplyPlacementRequantization: failed to "
								 "parse sidecar JSON "
							  << sp;
			return;
		}
		const Sidecar &sc = *parsed;
		if (sc.by_dispatch_id.empty()) {
			// No-op when no ops require re-quant.
			return;
		}

		mod.walk([&](FunctionOpInterface fn) {
			if (fn.isDeclaration() || fn.isExternal())
				return WalkResult::advance();
			auto anchors = CollectAnchorOps(fn);
			for (auto [idx, op] : llvm::enumerate(anchors)) {
				auto it = sc.by_dispatch_id.find((int)idx);
				if (it == sc.by_dispatch_id.end())
					continue;
				const PlacementEntry &pe = it->second;
				// Annotate for downstream tooling visibility.
				op->setAttr("merlin.requant.to_dtype",
					StringAttr::get(op->getContext(), pe.to_dtype));
				op->setAttr("merlin.requant.machine",
					StringAttr::get(op->getContext(), pe.machine));
				if (pe.from_dtype == pe.to_dtype)
					continue;

				OpBuilder b(op);
				bool wrapped_any = false;
				for (auto &operand : op->getOpOperands()) {
					Value v = operand.get();
					if (WrapTensorOperand(b, op->getLoc(), v, pe)) {
						operand.set(v);
						wrapped_any = true;
					}
				}
				if (wrapped_any) {
					LLVM_DEBUG(llvm::dbgs()
						<< "applied requant to anchor[" << idx << "] for "
						<< pe.from_dtype << "->" << pe.to_dtype << "\n");
				}
			}
			return WalkResult::advance();
		});
	}

	void setSidecarPath(StringRef p) {
		sidecar_path_.setValue(p.str());
	}

  private:
	Option<std::string> sidecar_path_{*this, "sidecar",
		llvm::cl::desc("Path to placement_requant.json sidecar from "
					   "XPU-RT/scripts/heterogeneous_loop.py"),
		llvm::cl::init("")};
};

} // namespace

std::unique_ptr<Pass> createApplyPlacementRequantizationPass(
	StringRef sidecar_path) {
	auto pass = std::make_unique<ApplyPlacementRequantizationPass>();
	if (!sidecar_path.empty())
		pass->setSidecarPath(sidecar_path);
	return pass;
}

} // namespace mlir::iree_compiler::QNN
