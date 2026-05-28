// QDQ → quant.uniform fold pass.
//
// This pass walks `linalg.generic` ops that match the dequantize/quantize
// body shapes (extsi · subi · sitofp · mulf for dequant; fptosi(round(
// clamp(divf(x, scale)))) for quantize), extracts the per-tensor scale +
// zero_point constants from the body, and rewrites the producer/consumer
// tensor types to wrap with `!quant.uniform<i8:f32, scale, zp>`. The
// dequant/quant generic itself is replaced by an
// `unrealized_conversion_cast` from i8-storage to quant.uniform-typed
// (or vice versa). SerializeGraph (compiler/plugins/target/QNN/Codegen/
// SerializeGraph.cpp:325) reads quant.uniform from element types directly.
//
// Why this is the universal answer: every quantization frontend (PyTorch
// pt2e, TFLite int8, ONNX QDQ) lowers to QDQ-decomposed linalg.generic
// chains — not to quant.uniform-typed tensors. Recognizing these chains
// in the compiler keeps the codegen path frontend-agnostic and lets ANY
// quantized network compile to QNN-validator-compliant graphs without
// per-network adaptation.
//
// SCAFFOLD ONLY: this file establishes the pass shell, registration, and
// documents the intended rewrite. The body-matcher integration with
// matchDequantBody / matchQuantizeBody (already in
// compiler/src/merlin/Dialect/QNN/Transforms/GenericMatchUtils.cpp) and
// the type-rewrite + cast-bridging logic are the next-session work.

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/QNN/Transforms/GenericMatchUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::QNN {
namespace {

// Type-based fold (survives DispatchCreation): replace the dequant/quant
// generic with an `unrealized_conversion_cast` whose result type wraps the
// i8 storage element with `quant.uniform<i8:f32, scale, zp>`. The downstream
// consumer (which expected the original storage type) gets a SECOND
// unrealized_conversion_cast back to that storage type — so consumer SSA
// type-checks still work, and any later pass that examines the producer's
// quant.uniform-typed result can read the scale/zp from the type.
//
// SerializeGraph (compiler/plugins/target/QNN/Codegen/SerializeGraph.cpp:325)
// reads quant.uniform from element types, so once a qnn op's input/output
// is wrapped, SerializeGraph emits qk=1 with the matched per-tensor scale/zp.
//
// Why type-based and not attr-based: dispatch creation generalizes ops +
// creates new linalg.generic in dispatch regions, dropping op attributes.
// Tensor types DO ride through dispatch creation unchanged (verified
// empirically: phase 4 attrs survive, phase 5 doesn't; phase 4 types do).

static quant::UniformQuantizedType makeQuantUniformI8(
	MLIRContext *ctx, double scale, int64_t zp) {
	Builder b(ctx);
	return quant::UniformQuantizedType::get(
		/*flags=*/quant::QuantizationFlags::Signed,
		/*storageType=*/b.getI8Type(),
		/*expressedType=*/b.getF32Type(),
		/*scale=*/scale,
		/*zeroPoint=*/zp,
		/*storageMin=*/-128, /*storageMax=*/127);
}

struct FoldDequantizeIntoType : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isF32())
			return failure();
		auto inTy = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		if (!inTy.getElementType().isInteger(8))
			return failure();

		double scale = 1.0;
		int64_t zp = 0;
		if (!matchDequantBody(op.getRegion().front(), &scale, &zp))
			return failure();

		// No-op for now. Per-variant integration in ConvertLinalgToQNN
		// (LowerConv2dQGeneric::findUpstreamDequant) handles the actual
		// discovery + propagation. This pass is left in place as a hook
		// for future module-level QDQ optimization.
		(void)scale;
		(void)zp;
		return failure();
	}
};

struct FoldQuantizeIntoType : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isInteger(8))
			return failure();
		auto inTy = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		if (!inTy.getElementType().isF32())
			return failure();

		double scale = 1.0;
		int64_t zp = 0;
		if (!matchQuantizeBody(op.getRegion().front(), &scale, &zp))
			return failure();
		(void)scale;
		(void)zp;
		return failure();
	}
};

class RewriteQDQToQuantUniformPass
	: public PassWrapper<RewriteQDQToQuantUniformPass,
		  OperationPass<ModuleOp>> {
  public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteQDQToQuantUniformPass)

	StringRef getArgument() const final {
		return "merlin-rewrite-qdq-to-quant-uniform";
	}
	StringRef getDescription() const final {
		return "Fold QDQ-decomposed linalg.generic chains into "
			   "quant.uniform-typed tensors so downstream linalg→qnn matches "
			   "cleanly. Universal across quant frontends (pt2e, TFLite, "
			   "ONNX).";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
			quant::QuantDialect, func::FuncDialect>();
	}

	void runOnOperation() override {
		// Discovery walk: build a map from i8-storage Value → (scale, zp) by
		// scanning every dequantize linalg.generic in the module.
		// Then for each conv-shape op (named linalg.conv_2d_*_q OR a
		// linalg.generic that matches matchQuantConvBody), look up its
		// input/weight in the map and stamp `merlin.qnn_input_scale` /
		// `merlin.qnn_weight_scale` attrs. ConvertLinalgToQNN copies these
		// attrs onto the resulting qnn.conv2d (per-variant), and
		// SerializeGraph emits qk=1 with the matched scale on the
		// corresponding tensor record.
		//
		// Mirror walk for quantize generics: scale/zp on the i8 PRODUCED
		// value goes onto the OUTPUT_scale of the producer conv/matmul.
		llvm::DenseMap<Value, std::pair<double, int64_t>> dequantOf;
		llvm::DenseMap<Value, std::pair<double, int64_t>> quantOf;
		getOperation().walk([&](linalg::GenericOp g) {
			if (!linalg::isElementwise(g) || g.getNumDpsInputs() != 1 ||
				g.getNumDpsInits() != 1)
				return;
			auto outTy = cast<RankedTensorType>(g.getResult(0).getType());
			auto inTy = cast<RankedTensorType>(g.getDpsInputs()[0].getType());
			double s = 1.0;
			int64_t z = 0;
			if (outTy.getElementType().isF32() &&
				inTy.getElementType().isInteger(8)) {
				if (matchDequantBody(g.getRegion().front(), &s, &z))
					dequantOf[g.getDpsInputs()[0]] = {s, z};
			} else if (outTy.getElementType().isInteger(8) &&
				inTy.getElementType().isF32()) {
				if (matchQuantizeBody(g.getRegion().front(), &s, &z))
					quantOf[g.getResult(0)] = {s, z};
			}
		});

		// Stamp on conv-shape ops. Handle both named-op
		// `linalg.conv_2d_nhwc_hwcf_q` (3 i8 input + 1 i8 weight + 2 i32
		// zps) and the generic body form.
		auto stampInput = [](Operation *op, double s, int64_t z) {
			Builder b(op->getContext());
			op->setAttr("merlin.qnn_input_scale",
				b.getF32FloatAttr(static_cast<float>(s)));
			op->setAttr("merlin.qnn_input_zero_point",
				b.getI32IntegerAttr(static_cast<int32_t>(z)));
		};
		auto stampWeight = [](Operation *op, double s, int64_t z) {
			Builder b(op->getContext());
			op->setAttr("merlin.qnn_weight_scale",
				b.getF32FloatAttr(static_cast<float>(s)));
			op->setAttr("merlin.qnn_weight_zero_point",
				b.getI32IntegerAttr(static_cast<int32_t>(z)));
		};
		auto stampOutput = [](Operation *op, double s, int64_t z) {
			Builder b(op->getContext());
			op->setAttr("merlin.qnn_output_scale",
				b.getF32FloatAttr(static_cast<float>(s)));
			op->setAttr("merlin.qnn_output_zero_point",
				b.getI32IntegerAttr(static_cast<int32_t>(z)));
		};

		getOperation().walk([&](Operation *op) {
			// Named convs.
			if (auto convQ = dyn_cast<linalg::Conv2DNhwcHwcfQOp>(op)) {
				Value in = convQ.getDpsInputs()[0];
				Value w = convQ.getDpsInputs()[1];
				if (auto it = dequantOf.find(in); it != dequantOf.end())
					stampInput(op, it->second.first, it->second.second);
				if (auto it = dequantOf.find(w); it != dequantOf.end())
					stampWeight(op, it->second.first, it->second.second);
				if (auto it = quantOf.find(convQ.getResult(0));
					it != quantOf.end())
					stampOutput(op, it->second.first, it->second.second);
				return;
			}
			// Generic conv-shape (matmul-like, generalized convs).
			if (auto g = dyn_cast<linalg::GenericOp>(op)) {
				if (g.getNumDpsInputs() < 2)
					return;
				Value in0 = g.getDpsInputs()[0];
				Value in1 = g.getDpsInputs()[1];
				if (auto it = dequantOf.find(in0); it != dequantOf.end())
					stampInput(op, it->second.first, it->second.second);
				if (auto it = dequantOf.find(in1); it != dequantOf.end())
					stampWeight(op, it->second.first, it->second.second);
				if (g->getNumResults() == 1) {
					if (auto it = quantOf.find(g.getResult(0));
						it != quantOf.end())
						stampOutput(op, it->second.first, it->second.second);
				}
			}
		});

		// Patterns are no-ops (preserved as a future hook). The discovery
		// above is the actual work.
		RewritePatternSet patterns(&getContext());
		patterns.add<FoldDequantizeIntoType, FoldQuantizeIntoType>(
			&getContext());
		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createRewriteQDQToQuantUniformPass() {
	return std::make_unique<RewriteQDQToQuantUniformPass>();
}

} // namespace mlir::iree_compiler::QNN
