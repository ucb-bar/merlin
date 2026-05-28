//===- LowerTileToISA.cpp - Bridge gemmini.matmul_tile -> gemmini.tile_* --===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//
//
// Bridges the recovered tensor-domain Gemmini ops (`gemmini.matmul_tile`,
// `gemmini.conv2d`) onto the memref-domain ISA-tier ops (`gemmini.tile_matmul`,
// `gemmini.tile_conv`) that LegalizeForLLVMExport.cpp consumes.
//
// IMPORTANT — bufferization gap:
// `gemmini.matmul_tile` produces tensors; `gemmini.tile_matmul` operates on
// memrefs. Without a separate bufferization pass running first, the patterns
// in this file will only fire on already-bufferized inputs (i.e., when
// `op.getLhs().getType()` is a `MemRefType`). The pass is structured so that
// pure-tensor inputs are simply left alone, and a downstream lit fixture that
// authors `gemmini.matmul_tile` directly with memref operands exercises the
// bridge end-to-end. See docs/dev_blog/2026-03-11-gemmini-workstream-log.md
// for the bufferization roadmap.
//
//===----------------------------------------------------------------------===//

#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINILOWERTILETOISAPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

// Result of matching the canonical quantized-matmul tail that IREE emits for
// QDQ models. The chain (post-bufferization, inside a single dispatch) is:
//
//   linalg.fill   ins(0)             outs(%mm_acc : i32)
//   linalg.matmul ins(A, B)          outs(%mm_acc : i32)             // A·B
//   linalg.generic ins(%mm_acc, %bias) outs(%biased : i32) { addi }  //
//   optional linalg.generic ins(%biased) outs(%c_i8 : i8) {
//     %a = arith.sitofp %in : i32 to f32
//     %b = arith.mulf  %a, %scale_in : f32     // = 1 / quantize_step
//     %c = arith.divf  %b, %scale_out : f32    // = output quant step
//     %d = math.roundeven %c : f32
//     %e = arith.addf  %d, %zp_out : f32       // we require zp_out == 0
//     %f = arith.maximumf %e, -128.0 : f32
//     %g = arith.minimumf %f,  127.0 : f32
//     %h = arith.fptosi  %g : f32 to i8
//     (optional ReLU tail: cmpi sgt 0 / select / sitofp / roundeven / addf 0 /
//      maximumf -128 / minimumf 127 / fptosi — semantically equivalent to
//      passing act=RELU to Gemmini's CONFIG_EX.)
//   }
//
// The net scale applied to the int32 accumulator is `scale_in / scale_out`,
// which is exactly Gemmini's CONFIG_ST acc_scale semantics. Folding this
// rescale (and the optional bias-add and ReLU) into the matmul lets us emit
// `tile_matmul` with `fullC=false`, `accScale=net`, `dArray=bias`, `act=RELU?`
// — shrinking the MVOUT data path from i32→i8 and eliminating the separate
// CPU/RVV rescale dispatch. See the 2026-05-26 investigation log for the
// dronet `matmul_3136x32x27` case, which the canonical chipyard equivalent
// completes in 28,800 cycles vs our (pre-fold) 16.3M.
struct RescaleFoldMatch {
	linalg::GenericOp biasAddOp; // optional bias-add op (may be null)
	linalg::GenericOp rescaleOp;
	Value outI8; // destination memref for the fused matmul (i8)
	Value biasMemRef; // the i32 bias buffer (only valid if biasAddOp set)
	float accScale; // scale_in / scale_out
	bool relu; // true → emit Gemmini act=RELU
	bool biasIsBroadcast; // true → bias memref is rank-1 / broadcast indexed,
						  //        emit TileMatMul with repeatingBias=true
};

// Match the rescale linalg.generic body. `bbIn` is the bb-arg that the
// matmul's int32 acc flows into.  Returns the extracted scale and a flag
// indicating whether the body included the post-rescale ReLU tail.
static bool matchRescaleBody(
	linalg::GenericOp gen, Value bbIn, float &netScale, bool &relu) {
	Block &body = gen.getRegion().front();
	auto term = dyn_cast<linalg::YieldOp>(body.getTerminator());
	if (!term || term.getNumOperands() != 1)
		return false;

	auto fptosi = term->getOperand(0).getDefiningOp<arith::FPToSIOp>();
	if (!fptosi)
		return false;

	// Look for the optional ReLU tail: the final fptosi is preceded by
	// minimumf(127) ← maximumf(-128) ← addf(0) ← roundeven ← sitofp ← select.
	// If we see that select(cmpi sgt 0, ...) pattern, an inner fptosi
	// produces the input — keep walking through to the canonical chain.
	relu = false;
	Value cursor = fptosi.getIn();
	auto minF = cursor.getDefiningOp<arith::MinimumFOp>();
	if (!minF)
		return false;
	if (auto clipHiCst = minF.getRhs().getDefiningOp<arith::ConstantOp>()) {
		auto attr = dyn_cast<FloatAttr>(clipHiCst.getValue());
		if (!attr || attr.getValueAsDouble() != 127.0)
			return false;
	} else
		return false;
	auto maxF = minF.getLhs().getDefiningOp<arith::MaximumFOp>();
	if (!maxF)
		return false;
	if (auto clipLoCst = maxF.getRhs().getDefiningOp<arith::ConstantOp>()) {
		auto attr = dyn_cast<FloatAttr>(clipLoCst.getValue());
		if (!attr || attr.getValueAsDouble() != -128.0)
			return false;
	} else
		return false;
	auto addZp = maxF.getLhs().getDefiningOp<arith::AddFOp>();
	if (!addZp)
		return false;
	if (auto zpCst = addZp.getRhs().getDefiningOp<arith::ConstantOp>()) {
		auto attr = dyn_cast<FloatAttr>(zpCst.getValue());
		if (!attr || attr.getValueAsDouble() != 0.0)
			return false;
	} else
		return false;
	auto round = addZp.getLhs().getDefiningOp<math::RoundEvenOp>();
	if (!round)
		return false;
	cursor = round.getOperand();

	// At this point `cursor` is the f32 value being rounded.  Two shapes:
	//  - no-ReLU:   cursor ← arith.divf (mulf bbIn scale_in) scale_out
	//  - with-ReLU: cursor ← arith.sitofp (select (cmpi sgt 0, x, 0)) where
	//               x is the post-clip i8 from an *inner* fptosi rooted at
	//               another minimumf/maximumf/addf/roundeven/divf/mulf chain
	//               on bbIn.
	if (auto innerSitofp = cursor.getDefiningOp<arith::SIToFPOp>()) {
		// ReLU tail.
		relu = true;
		auto sel = innerSitofp.getIn().getDefiningOp<arith::SelectOp>();
		if (!sel)
			return false;
		auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
		if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::sgt)
			return false;
		auto innerFptosi = sel.getTrueValue().getDefiningOp<arith::FPToSIOp>();
		if (!innerFptosi)
			return false;
		// Walk the inner saturate chain.
		auto innerMin = innerFptosi.getIn().getDefiningOp<arith::MinimumFOp>();
		if (!innerMin)
			return false;
		auto innerMax = innerMin.getLhs().getDefiningOp<arith::MaximumFOp>();
		if (!innerMax)
			return false;
		auto innerAdd = innerMax.getLhs().getDefiningOp<arith::AddFOp>();
		if (!innerAdd)
			return false;
		auto innerRound = innerAdd.getLhs().getDefiningOp<math::RoundEvenOp>();
		if (!innerRound)
			return false;
		cursor = innerRound.getOperand();
	}

	auto divf = cursor.getDefiningOp<arith::DivFOp>();
	if (!divf)
		return false;
	auto outScaleCst = divf.getRhs().getDefiningOp<arith::ConstantOp>();
	if (!outScaleCst)
		return false;
	auto outScaleAttr = dyn_cast<FloatAttr>(outScaleCst.getValue());
	if (!outScaleAttr)
		return false;
	auto mulf = divf.getLhs().getDefiningOp<arith::MulFOp>();
	if (!mulf)
		return false;
	auto inScaleCst = mulf.getRhs().getDefiningOp<arith::ConstantOp>();
	if (!inScaleCst)
		return false;
	auto inScaleAttr = dyn_cast<FloatAttr>(inScaleCst.getValue());
	if (!inScaleAttr)
		return false;
	auto inSitofp = mulf.getLhs().getDefiningOp<arith::SIToFPOp>();
	if (!inSitofp || inSitofp.getIn() != bbIn)
		return false;

	double outScale = outScaleAttr.getValueAsDouble();
	if (outScale == 0.0)
		return false;
	double net = inScaleAttr.getValueAsDouble() / outScale;
	if (!std::isfinite(net) || net <= 0.0)
		return false;
	netScale = static_cast<float>(net);
	return true;
}

// Find the linalg.generic that consumes `producer`-with-element-type-i32 and
// produces an i8 element-type memref via the canonical rescale body.
// Returns nullptr on failure.  `producerVal` must be one of the matmul's
// `outs` or a bias-add's `outs` (both i32 memrefs).
static linalg::GenericOp findRescaleConsumer(
	Value producerVal, Operation *afterOp) {
	for (Operation *user : producerVal.getUsers()) {
		auto gen = dyn_cast<linalg::GenericOp>(user);
		if (!gen)
			continue;
		if (gen->getBlock() != afterOp->getBlock())
			continue;
		if (!afterOp->isBeforeInBlock(gen))
			continue;
		if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1)
			continue;
		if (gen.getDpsInputs()[0] != producerVal)
			continue;
		auto outTy = dyn_cast<MemRefType>(gen.getDpsInits()[0].getType());
		if (!outTy || !outTy.getElementType().isInteger(8))
			continue;
		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 2)
			continue;
		if (!maps[0].isIdentity() || !maps[1].isIdentity())
			continue;
		return gen;
	}
	return nullptr;
}

// Result of matching a bias-add `linalg.generic` of the form
// `out = matmul_dest + bias`. `biasIsBroadcast` is true when the bias
// operand uses indexing `(d0, d1) -> (d1)` (rank-1 N-element bias replayed
// across all M output rows). In that case
// `LowerBufferizedLinalgMatmulToTileMatmul` forwards the bias to TileMatMul's
// `dArray` as a 1×N view and sets `repeatingBias=true`, saving the per-dispatch
// MVIN-D cost of a fully materialized MxN bias buffer.
struct BiasAddMatch {
	linalg::GenericOp op;
	Value biasVal;
	bool biasIsBroadcast;
};

static std::optional<BiasAddMatch> findBiasAddConsumerImpl(
	Value matmulDest, Operation *matmulOp) {
	MLIRContext *ctx = matmulOp->getContext();
	AffineExpr d0 = getAffineDimExpr(0, ctx);
	AffineExpr d1 = getAffineDimExpr(1, ctx);
	AffineMap broadcastMap = AffineMap::get(2, 0, {d1}, ctx);

	for (Operation *user : matmulDest.getUsers()) {
		auto gen = dyn_cast<linalg::GenericOp>(user);
		if (!gen)
			continue;
		if (gen->getBlock() != matmulOp->getBlock())
			continue;
		if (!matmulOp->isBeforeInBlock(gen))
			continue;
		if (gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1)
			continue;
		auto outTy = dyn_cast<MemRefType>(gen.getDpsInits()[0].getType());
		if (!outTy || !outTy.getElementType().isInteger(32))
			continue;
		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 3)
			continue;
		if (!maps[2].isIdentity())
			continue;
		// Identify which input is the matmul's dest, which is the bias.
		int matmulIdx, biasIdx;
		if (gen.getDpsInputs()[0] == matmulDest) {
			matmulIdx = 0;
			biasIdx = 1;
		} else if (gen.getDpsInputs()[1] == matmulDest) {
			matmulIdx = 1;
			biasIdx = 0;
		} else {
			continue;
		}
		if (!maps[matmulIdx].isIdentity())
			continue;
		// Bias indexing must be either identity (full MxN bias) OR
		// `(d0,d1)->(d1)` (broadcast — repeatingBias=true).
		bool biasIsBroadcast = false;
		if (maps[biasIdx].isIdentity()) {
			biasIsBroadcast = false;
		} else if (maps[biasIdx] == broadcastMap) {
			biasIsBroadcast = true;
		} else {
			continue;
		}
		// Body must be exactly `arith.addi(in0, in1)`.
		Block &body = gen.getRegion().front();
		if (body.getNumArguments() != 3)
			continue;
		auto term = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!term || term.getNumOperands() != 1)
			continue;
		auto addi = term->getOperand(0).getDefiningOp<arith::AddIOp>();
		if (!addi)
			continue;
		Value a = body.getArgument(0), b = body.getArgument(1);
		Value lhs = addi.getLhs(), rhs = addi.getRhs();
		if (!((lhs == a && rhs == b) || (lhs == b && rhs == a)))
			continue;
		return BiasAddMatch{gen, gen.getDpsInputs()[biasIdx], biasIsBroadcast};
	}
	return std::nullopt;
}

// Legacy pair-returning wrapper used by tryMatchI32ToI8Rescale and similar
// helpers that don't yet care about the broadcast bit. New call sites
// should use `findBiasAddConsumerImpl` directly.
static std::pair<linalg::GenericOp, Value> findBiasAddConsumer(
	Value matmulDest, Operation *matmulOp) {
	auto m = findBiasAddConsumerImpl(matmulDest, matmulOp);
	if (!m.has_value())
		return {nullptr, nullptr};
	return {m->op, m->biasVal};
}

static std::optional<RescaleFoldMatch> tryMatchI32ToI8Rescale(
	linalg::MatmulOp matmul) {
	Value matmulDest = matmul.getDpsInits()[0];
	auto destMemRef = dyn_cast<MemRefType>(matmulDest.getType());
	if (!destMemRef || !destMemRef.getElementType().isInteger(32))
		return std::nullopt;

	// Look for an optional bias-add between matmul and rescale.
	auto biasMatch = findBiasAddConsumerImpl(matmulDest, matmul.getOperation());
	linalg::GenericOp biasAdd = biasMatch ? biasMatch->op : nullptr;
	Value biasVal = biasMatch ? biasMatch->biasVal : Value{};
	bool biasIsBroadcast = biasMatch ? biasMatch->biasIsBroadcast : false;

	Value rescaleSource = biasAdd ? biasAdd.getDpsInits()[0] : matmulDest;
	Operation *rescaleAfter =
		biasAdd ? biasAdd.getOperation() : matmul.getOperation();

	linalg::GenericOp rescale =
		findRescaleConsumer(rescaleSource, rescaleAfter);
	if (!rescale)
		return std::nullopt;

	// The intermediate i32 buffer(s) must only be read/written by ops that
	// are part of this fold (or a pre-matmul linalg.fill that initializes
	// the accumulator — it becomes dead once the matmul's destination is
	// rewritten and DCE will sweep it). Anything else (memref.load, a
	// second consumer, etc.) means folding would change semantics.
	for (Operation *user : matmulDest.getUsers()) {
		if (user == matmul.getOperation())
			continue;
		if (biasAdd && user == biasAdd.getOperation())
			continue;
		if (!biasAdd && user == rescale.getOperation())
			continue;
		if (isa<linalg::FillOp>(user) &&
			user->isBeforeInBlock(matmul.getOperation()))
			continue;
		return std::nullopt;
	}
	if (biasAdd) {
		Value biasOut = biasAdd.getDpsInits()[0];
		for (Operation *user : biasOut.getUsers()) {
			if (user == biasAdd.getOperation())
				continue;
			if (user == rescale.getOperation())
				continue;
			if (isa<linalg::FillOp>(user) &&
				user->isBeforeInBlock(biasAdd.getOperation()))
				continue;
			return std::nullopt;
		}
	}

	Block &body = rescale.getRegion().front();
	if (body.getNumArguments() != 2)
		return std::nullopt;
	Value bbIn = body.getArgument(0);
	float net = 0.0f;
	bool relu = false;
	if (!matchRescaleBody(rescale, bbIn, net, relu))
		return std::nullopt;

	return RescaleFoldMatch{biasAdd, rescale, rescale.getDpsInits()[0], biasVal,
		net, relu, biasIsBroadcast};
}

// Result of matching the chained residual rescale that follows the matmul
// in dronet's dispatch_11/20/29. See the 20-op body in
// `compiler/src/merlin/Dialect/Gemmini/Transforms/Preprocess.cpp` for the
// canonical chain. Bit-exact full fold is impossible because of the
// intermediate clip; this partial fold collapses just the matmul + first
// rescale into one TileMatMul.
struct ChainedResidualFoldMatch {
	linalg::GenericOp biasAddOp; // optional bias-add op (may be null)
	linalg::GenericOp rescaleOp; // the chained 20-op linalg.generic
	Value biasMemRef; // bias buffer (only valid if biasAddOp set)
	float accScale; // scale_in / scale_int (the FIRST rescale)
	// The constants from the second half of the chain. We have to clone the
	// body into a fresh 11-op linalg.generic that reads the new i8 matmul
	// output instead of the original i32 destination.
	Value scaleResVal; // residual scale (mulf)
	Value scaleIntVal; // intermediate scale, reused on the rebuild side
	Value scaleOutVal; // output scale (divf)
	Value zeroVal; // zero-point (0.0)
	Value clipLoVal; // -128.0
	Value clipHiVal; // 127.0
};

static std::optional<ChainedResidualFoldMatch> tryMatchChainedResidualRescale(
	linalg::MatmulOp matmul) {
	Value matmulDest = matmul.getDpsInits()[0];
	auto destMemRef = dyn_cast<MemRefType>(matmulDest.getType());
	if (!destMemRef || !destMemRef.getElementType().isInteger(32))
		return std::nullopt;

	auto [biasAdd, biasVal] =
		findBiasAddConsumer(matmulDest, matmul.getOperation());

	Value rescaleSource = biasAdd ? biasAdd.getDpsInits()[0] : matmulDest;
	Operation *rescaleAfter =
		biasAdd ? biasAdd.getOperation() : matmul.getOperation();

	// Find the 2-input chained rescale (NOT the 1-input one which
	// tryMatchI32ToI8Rescale handles).
	linalg::GenericOp chained = nullptr;
	for (Operation *user : rescaleSource.getUsers()) {
		auto gen = dyn_cast<linalg::GenericOp>(user);
		if (!gen)
			continue;
		if (gen->getBlock() != rescaleAfter->getBlock())
			continue;
		if (!rescaleAfter->isBeforeInBlock(gen))
			continue;
		if (gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1)
			continue;
		if (gen.getDpsInputs()[0] != rescaleSource)
			continue;
		auto in1Ty = dyn_cast<MemRefType>(gen.getDpsInputs()[1].getType());
		if (!in1Ty || !in1Ty.getElementType().isInteger(8))
			continue;
		auto outTy = dyn_cast<MemRefType>(gen.getDpsInits()[0].getType());
		if (!outTy || !outTy.getElementType().isInteger(8))
			continue;
		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 3)
			continue;
		if (!maps[0].isIdentity() || !maps[1].isIdentity() ||
			!maps[2].isIdentity())
			continue;
		chained = gen;
		break;
	}
	if (!chained)
		return std::nullopt;

	// Same single-use safety checks as the clean fold.
	for (Operation *user : matmulDest.getUsers()) {
		if (user == matmul.getOperation())
			continue;
		if (biasAdd && user == biasAdd.getOperation())
			continue;
		if (!biasAdd && user == chained.getOperation())
			continue;
		if (isa<linalg::FillOp>(user) &&
			user->isBeforeInBlock(matmul.getOperation()))
			continue;
		return std::nullopt;
	}
	if (biasAdd) {
		Value biasOut = biasAdd.getDpsInits()[0];
		for (Operation *user : biasOut.getUsers()) {
			if (user == biasAdd.getOperation())
				continue;
			if (user == chained.getOperation())
				continue;
			if (isa<linalg::FillOp>(user) &&
				user->isBeforeInBlock(biasAdd.getOperation()))
				continue;
			return std::nullopt;
		}
	}

	Block &body = chained.getRegion().front();
	if (body.getNumArguments() != 3)
		return std::nullopt;
	auto term = dyn_cast<linalg::YieldOp>(body.getTerminator());
	if (!term || term.getNumOperands() != 1)
		return std::nullopt;

	SmallVector<Operation *> ops;
	for (Operation &nested : body.without_terminator())
		ops.push_back(&nested);
	// 19 non-terminator ops: 8 first-rescale + 4 residual+intermediate
	// mulf chain + 1 addf sum + 6 second-rescale. The terminator
	// (linalg.yield) is excluded from `without_terminator()`.
	if (ops.size() != 19)
		return std::nullopt;

	auto sf0 = dyn_cast<arith::SIToFPOp>(ops[0]);
	auto mf0 = dyn_cast<arith::MulFOp>(ops[1]);
	auto df0 = dyn_cast<arith::DivFOp>(ops[2]);
	auto rd0 = dyn_cast<math::RoundEvenOp>(ops[3]);
	auto af0 = dyn_cast<arith::AddFOp>(ops[4]);
	auto mx0 = dyn_cast<arith::MaximumFOp>(ops[5]);
	auto mn0 = dyn_cast<arith::MinimumFOp>(ops[6]);
	auto fp0 = dyn_cast<arith::FPToSIOp>(ops[7]);
	auto sf_r = dyn_cast<arith::SIToFPOp>(ops[8]);
	auto mf_r = dyn_cast<arith::MulFOp>(ops[9]);
	auto sf_i = dyn_cast<arith::SIToFPOp>(ops[10]);
	auto mf_i = dyn_cast<arith::MulFOp>(ops[11]);
	auto sum = dyn_cast<arith::AddFOp>(ops[12]);
	auto df1 = dyn_cast<arith::DivFOp>(ops[13]);
	auto rd1 = dyn_cast<math::RoundEvenOp>(ops[14]);
	auto af1 = dyn_cast<arith::AddFOp>(ops[15]);
	auto mx1 = dyn_cast<arith::MaximumFOp>(ops[16]);
	auto mn1 = dyn_cast<arith::MinimumFOp>(ops[17]);
	auto fp1 = dyn_cast<arith::FPToSIOp>(ops[18]);
	if (!sf0 || !mf0 || !df0 || !rd0 || !af0 || !mx0 || !mn0 || !fp0)
		return std::nullopt;
	if (!sf_r || !mf_r || !sf_i || !mf_i || !sum)
		return std::nullopt;
	if (!df1 || !rd1 || !af1 || !mx1 || !mn1 || !fp1)
		return std::nullopt;

	Value mmIn = body.getArgument(0);
	Value resIn = body.getArgument(1);
	if (sf0.getIn() != mmIn)
		return std::nullopt;
	if (mf0.getLhs() != sf0.getResult())
		return std::nullopt;
	if (df0.getLhs() != mf0.getResult())
		return std::nullopt;
	if (rd0.getOperand() != df0.getResult())
		return std::nullopt;
	if (af0.getLhs() != rd0.getResult())
		return std::nullopt;
	if (mx0.getLhs() != af0.getResult())
		return std::nullopt;
	if (mn0.getLhs() != mx0.getResult())
		return std::nullopt;
	if (fp0.getIn() != mn0.getResult())
		return std::nullopt;
	if (sf_r.getIn() != resIn)
		return std::nullopt;
	if (mf_r.getLhs() != sf_r.getResult())
		return std::nullopt;
	if (sf_i.getIn() != fp0.getResult())
		return std::nullopt;
	if (mf_i.getLhs() != sf_i.getResult())
		return std::nullopt;
	if (sum.getLhs() != mf_i.getResult() || sum.getRhs() != mf_r.getResult())
		return std::nullopt;
	if (df1.getLhs() != sum.getResult())
		return std::nullopt;
	if (rd1.getOperand() != df1.getResult())
		return std::nullopt;
	if (af1.getLhs() != rd1.getResult())
		return std::nullopt;
	if (mx1.getLhs() != af1.getResult())
		return std::nullopt;
	if (mn1.getLhs() != mx1.getResult())
		return std::nullopt;
	if (fp1.getIn() != mn1.getResult())
		return std::nullopt;
	if (term->getOperand(0) != fp1.getResult())
		return std::nullopt;

	// Pull scale_in / scale_int from defining constants — these are
	// f32 splat constants outside the region.
	auto getF32 = [](Value v) -> std::optional<double> {
		auto cst = v.getDefiningOp<arith::ConstantOp>();
		if (!cst)
			return std::nullopt;
		auto attr = dyn_cast<FloatAttr>(cst.getValue());
		if (!attr)
			return std::nullopt;
		return attr.getValueAsDouble();
	};
	auto scaleIn = getF32(mf0.getRhs());
	auto scaleInt = getF32(df0.getRhs());
	if (!scaleIn || !scaleInt || *scaleInt == 0.0)
		return std::nullopt;
	double net = *scaleIn / *scaleInt;
	if (!std::isfinite(net) || net <= 0.0)
		return std::nullopt;

	ChainedResidualFoldMatch m;
	m.biasAddOp = biasAdd;
	m.rescaleOp = chained;
	m.biasMemRef = biasVal;
	m.accScale = static_cast<float>(net);
	m.scaleResVal = mf_r.getRhs();
	m.scaleIntVal = mf_i.getRhs();
	m.scaleOutVal = df1.getRhs();
	m.zeroVal = af1.getRhs();
	m.clipLoVal = mx1.getRhs();
	m.clipHiVal = mn1.getRhs();
	return m;
}

// Result of matching a matmul → 1-input chained generic whose body STARTS
// with the canonical 8-op QDQ rescale and CONTINUES with non-rescale
// computation (yolov8n's matmul+SiLU+rescale chains). Bit-exact full fold
// is impossible because the post-rescale logic can't be expressed inside
// Gemmini's MVOUT. Partial fold: emit TileMatMul with accScale=in/int
// writing i8 to a fresh alloca; rewrite the chained generic to read that
// alloca and DROP its first 8 ops (the new bb-arg IS the i8 intermediate).
struct HeadRescaleSplitMatch {
	linalg::GenericOp biasAddOp; // optional bias-add op (may be null)
	linalg::GenericOp chainedOp; // the long-body 1-input rescale chain
	Value biasMemRef; // (only valid if biasAddOp set)
	float accScale;
	int64_t headOpCount; // 8 (the canonical rescale prefix)
};

static std::optional<HeadRescaleSplitMatch> tryMatchHeadRescaleAndSplit(
	linalg::MatmulOp matmul) {
	Value matmulDest = matmul.getDpsInits()[0];
	auto destMemRef = dyn_cast<MemRefType>(matmulDest.getType());
	if (!destMemRef || !destMemRef.getElementType().isInteger(32))
		return std::nullopt;

	auto [biasAdd, biasVal] =
		findBiasAddConsumer(matmulDest, matmul.getOperation());
	Value chainedSource = biasAdd ? biasAdd.getDpsInits()[0] : matmulDest;
	Operation *chainedAfter =
		biasAdd ? biasAdd.getOperation() : matmul.getOperation();

	// 1-input generic consuming chainedSource, identity-indexed, writing i8.
	linalg::GenericOp chained = nullptr;
	for (Operation *user : chainedSource.getUsers()) {
		auto gen = dyn_cast<linalg::GenericOp>(user);
		if (!gen)
			continue;
		if (gen->getBlock() != chainedAfter->getBlock())
			continue;
		if (!chainedAfter->isBeforeInBlock(gen))
			continue;
		if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1)
			continue;
		if (gen.getDpsInputs()[0] != chainedSource)
			continue;
		auto outTy = dyn_cast<MemRefType>(gen.getDpsInits()[0].getType());
		if (!outTy || !outTy.getElementType().isInteger(8))
			continue;
		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 2)
			continue;
		if (!maps[0].isIdentity() || !maps[1].isIdentity())
			continue;
		chained = gen;
		break;
	}
	if (!chained)
		return std::nullopt;

	for (Operation *user : matmulDest.getUsers()) {
		if (user == matmul.getOperation())
			continue;
		if (biasAdd && user == biasAdd.getOperation())
			continue;
		if (!biasAdd && user == chained.getOperation())
			continue;
		if (isa<linalg::FillOp>(user) &&
			user->isBeforeInBlock(matmul.getOperation()))
			continue;
		return std::nullopt;
	}
	if (biasAdd) {
		Value biasOut = biasAdd.getDpsInits()[0];
		for (Operation *user : biasOut.getUsers()) {
			if (user == biasAdd.getOperation())
				continue;
			if (user == chained.getOperation())
				continue;
			if (isa<linalg::FillOp>(user) &&
				user->isBeforeInBlock(biasAdd.getOperation()))
				continue;
			return std::nullopt;
		}
	}

	Block &body = chained.getRegion().front();
	SmallVector<Operation *> ops;
	for (Operation &nested : body.without_terminator())
		ops.push_back(&nested);
	// Need MORE than the trivial 8-op rescale.
	if (ops.size() <= 8)
		return std::nullopt;

	auto sf = dyn_cast<arith::SIToFPOp>(ops[0]);
	auto mf = dyn_cast<arith::MulFOp>(ops[1]);
	auto df = dyn_cast<arith::DivFOp>(ops[2]);
	auto rd = dyn_cast<math::RoundEvenOp>(ops[3]);
	auto af = dyn_cast<arith::AddFOp>(ops[4]);
	auto mx = dyn_cast<arith::MaximumFOp>(ops[5]);
	auto mn = dyn_cast<arith::MinimumFOp>(ops[6]);
	auto fp = dyn_cast<arith::FPToSIOp>(ops[7]);
	if (!sf || !mf || !df || !rd || !af || !mx || !mn || !fp)
		return std::nullopt;
	Value mmIn = body.getArgument(0);
	if (sf.getIn() != mmIn)
		return std::nullopt;
	if (mf.getLhs() != sf.getResult())
		return std::nullopt;
	if (df.getLhs() != mf.getResult())
		return std::nullopt;
	if (rd.getOperand() != df.getResult())
		return std::nullopt;
	if (af.getLhs() != rd.getResult())
		return std::nullopt;
	if (mx.getLhs() != af.getResult())
		return std::nullopt;
	if (mn.getLhs() != mx.getResult())
		return std::nullopt;
	if (fp.getIn() != mn.getResult())
		return std::nullopt;

	auto checkConst = [&](Value v, double expected) {
		auto cst = v.getDefiningOp<arith::ConstantOp>();
		if (!cst)
			return false;
		auto attr = dyn_cast<FloatAttr>(cst.getValue());
		return attr && attr.getValueAsDouble() == expected;
	};
	if (!checkConst(af.getRhs(), 0.0))
		return std::nullopt;
	if (!checkConst(mx.getRhs(), -128.0))
		return std::nullopt;
	if (!checkConst(mn.getRhs(), 127.0))
		return std::nullopt;

	auto getF32 = [](Value v) -> std::optional<double> {
		auto cst = v.getDefiningOp<arith::ConstantOp>();
		if (!cst)
			return std::nullopt;
		auto attr = dyn_cast<FloatAttr>(cst.getValue());
		if (!attr)
			return std::nullopt;
		return attr.getValueAsDouble();
	};
	auto scaleIn = getF32(mf.getRhs());
	auto scaleInt = getF32(df.getRhs());
	if (!scaleIn || !scaleInt || *scaleInt == 0.0)
		return std::nullopt;
	double net = *scaleIn / *scaleInt;
	if (!std::isfinite(net) || net <= 0.0)
		return std::nullopt;

	HeadRescaleSplitMatch m;
	m.biasAddOp = biasAdd;
	m.chainedOp = chained;
	m.biasMemRef = biasVal;
	m.accScale = static_cast<float>(net);
	m.headOpCount = 8;
	return m;
}

// Helper: emit a flush(skip=0) at the end of `func` if not already present.
static void appendFlushEpilogue(func::FuncOp func, OpBuilder &builder) {
	auto &block = func.getBody().front();
	for (auto &op : block) {
		if (isa<FlushOp>(op))
			return;
	}
	builder.setInsertionPoint(block.getTerminator());
	Value zero = builder.create<arith::ConstantOp>(
		func.getLoc(), builder.getI64IntegerAttr(0));
	builder.create<FlushOp>(func.getLoc(), zero);
}

struct LowerMatmulTileToISA final : OpRewritePattern<MatmulTileOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		MatmulTileOp op, PatternRewriter &rewriter) const override {
		// We only fire on memref inputs (post-bufferization). On pure-tensor
		// inputs the upstream `gemmini-canonicalize` / IREE bufferization is
		// expected to convert tensors to memrefs first.
		auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
		auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
		auto outType = dyn_cast<MemRefType>(op.getResult().getType());
		if (!lhsType || !rhsType || !outType)
			return rewriter.notifyMatchFailure(op,
				"lower-tile-to-isa requires memref operands; bufferize "
				"first");

		// `gemmini.tile_matmul` has 4 memref operands (A, B, C, D).
		// 2026-05-19 BIAS-PATH REVERT: previously aliased D = op.getResult()
		// to expose pre-fills as bias. Real models (dronet's FC matmul
		// dispatches 31/32) have an alloca output that's stack-garbage
		// uninitialized — MVIN-D pulled that garbage in as "bias", adding
		// random i32 to the matmul result. Confirmed on FireSim: the
		// fixture (fresh alloca, often zero) passes; dronet (alloca after
		// many prior dispatches) fails. Revert to emptyD so the matmul
		// emits OVERWRITE-mode PRELOAD = pure A·B with no spurious bias.
		// Models that need a real bias inject it via a separate
		// downstream linalg.generic, not via the matmul's outs buffer.
		Location loc = op.getLoc();
		// 2026-05-22 BUG FIX (multi-N-tile MVIN-D OOB): the earlier 16x16
		// hardcode worked for FC matmul (16x16 padded) but on multi-N-tile
		// shapes (M=16, N=32+), MVIN-D for j-tile >= 1 reads at offset
		// j0*dim*sizeOfAccT = 64 with row-stride strideD*sizeOfAccT = 16*4
		// = 64 bytes; that walks rows 1..16 of the 16x16 buffer instead of
		// cols 16..31 of row 0, and row 15 reads past the 1024-byte alloca,
		// pulling stack garbage in as pseudo-bias. Symptom on Spike:
		// matmul_16x32x16 row 15 cols 16, 18 = -938966256 / -2147409298
		// (= 16 + the OOB i32 values).
		// Fix: allocate D as 1×N with repeatingBias=true. dStride is then
		// forced to 0 by spTiledMatmulOs (replicate same row 16x in spad);
		// per-j-tile offset is j0*dim*sizeOfAccT and stays inside the
		// buffer. Stack footprint is N*4 bytes regardless of M, so this
		// works for large-M shapes (e.g. 3136x32x27) that would overflow
		// the 32KiB max_stack_allocation if D was sized M×N.
		int64_t outN = outType.getShape().back();
		auto i32MemRef = MemRefType::get({1, outN}, rewriter.getI32Type());
		Value emptyD = rewriter.create<memref::AllocaOp>(loc, i32MemRef);
		Value zeroI32 = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI32IntegerAttr(0));
		rewriter.create<linalg::FillOp>(
			loc, ValueRange{zeroI32}, ValueRange{emptyD});

		// 2026-05-18 B-TRANSPOSE FIX: Gemmini::MatmulOp models rhs as N×K
		// (see ConvertNamedMatmulPattern's `// Gemmini::MatmulOp currently
		// models rhs as NxK.`). But `spTiledMatmulOs` in
		// LegalizeForLLVMExport treats B as K×N (offsets `k0 * strideB +
		// j0`). For shapes where K == N (e.g. matmul_196x32x32) the two
		// interpretations are indistinguishable, which is why the Spike
		// fixtures of those shapes passed even though dronet's larger
		// shapes failed. For shapes where K != N (e.g. dronet's
		// matmul_16x128x64 with K=64, N=128 — B arrives as memref<128x64>),
		// we must tell `spTiledMatmulOs` that B is laid out N×K by setting
		// `bTranspose=true`.
		//
		// Detection: `B.dim0` should equal K (= A.dim1) for the K×N case
		// and equal N (= C.dim1) for the N×K case.
		bool bTranspose = false;
		auto lhsTy = dyn_cast<MemRefType>(op.getLhs().getType());
		auto rhsTy = dyn_cast<MemRefType>(op.getRhs().getType());
		auto outTy = dyn_cast<MemRefType>(op.getResult().getType());
		if (lhsTy && rhsTy && outTy && lhsTy.getRank() == 2 &&
			rhsTy.getRank() == 2 && outTy.getRank() == 2) {
			int64_t K = lhsTy.getDimSize(1);
			int64_t N = outTy.getDimSize(1);
			int64_t bDim0 = rhsTy.getDimSize(0);
			if (!ShapedType::isDynamic(K) && !ShapedType::isDynamic(N) &&
				!ShapedType::isDynamic(bDim0) && K != N) {
				if (bDim0 == N) {
					bTranspose = true;
				}
			}
		}

		const float identity = 1.0f;
		rewriter.create<TileMatMulOp>(loc, op.getLhs(), op.getRhs(),
			op.getResult(), emptyD,
			/*aScaleFactor=*/llvm::APFloat(identity),
			/*bScaleFactor=*/llvm::APFloat(identity),
			/*dScaleFactor=*/llvm::APFloat(identity),
			/*act=*/0,
			/*accScale=*/llvm::APFloat(identity),
			/*bertScale=*/llvm::APFloat(0.0f),
			/*repeatingBias=*/true,
			/*aTranspose=*/false,
			/*bTranspose=*/bTranspose,
			/*fullC=*/false,
			/*lowD=*/false,
			/*weightA=*/0,
			/*dataflow=*/static_cast<int64_t>(op.getDataflow()));
		rewriter.eraseOp(op);
		return success();
	}
};

// Inside-dispatch pattern: match a memref-domain `linalg.matmul` (post-
// bufferization) and emit `gemmini.tile_matmul` directly, skipping the
// gemmini.* tensor-tier. The gemmini tensor ops still exist for the host-
// IR debug path (lowerBackToIREE=true), but inside the dispatch we go
// straight from linalg to ISA.
struct LowerBufferizedLinalgMatmulToTileMatmul final
	: OpRewritePattern<linalg::MatmulOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::MatmulOp op, PatternRewriter &rewriter) const override {
		if (op.hasPureTensorSemantics())
			return rewriter.notifyMatchFailure(
				op, "requires bufferized (memref-domain) linalg.matmul");
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return rewriter.notifyMatchFailure(op, "expected 2 inputs, 1 init");

		Value lhs = op.getDpsInputs()[0];
		Value rhs = op.getDpsInputs()[1];
		Value out = op.getDpsInits()[0];
		if (!isa<MemRefType>(lhs.getType()) ||
			!isa<MemRefType>(rhs.getType()) || !isa<MemRefType>(out.getType()))
			return rewriter.notifyMatchFailure(op, "non-memref operand");

		Location loc = op.getLoc();

		// 2026-05-19 BIAS-PATH REVERT: previously aliased D = out to expose
		// pre-fills as bias. Real models (dronet's FC matmul dispatches
		// 31/32) have an alloca output that's stack-garbage uninitialized
		// — MVIN-D pulled that garbage in as "bias", adding random i32
		// to the matmul result. Confirmed on FireSim: the fixture
		// matmul_1x1x2048 (fresh alloca, often zero) passed; dronet's
		// same shape failed because its alloca had prior dispatches'
		// stack residue. Revert to emptyD so noBias=true is inferred,
		// PRELOAD emits OVERWRITE mode, accumulator = pure A·B.
		// Models that genuinely need an int32 bias should inject it via
		// a downstream linalg.generic, not via the matmul's outs buffer.
		// 2026-05-22 BUG FIX (multi-N-tile MVIN-D OOB): see
		// ConvertNamedMatmulPattern above for the full rationale.
		// Allocate D as 1×N with repeatingBias=true so the buffer stays
		// small (works for large-M shapes that would overflow the 32 KiB
		// max_stack_allocation budget) while MVIN-D's per-j-tile offset
		// stays inside the buffer.
		// Opt #C (2026-05-26): allocate D as a zero-shape memref so the
		// LegalizeForLLVMExport path detects `noBias=true` and libgemmini
		// substitutes a dummy address — MVIN-D is skipped entirely (saves
		// the per-tile MVIN-D cost AND the stack alloca + linalg.fill that
		// would otherwise need to run on the worker CPU). The downstream
		// `tile_matmul` lowering already special-cases shape-with-zero D
		// (see LegalizeForLLVMExport.cpp:986-994).
		auto i32MemRef = MemRefType::get({0, 0}, rewriter.getI32Type());
		Value emptyD = rewriter.create<memref::AllocaOp>(loc, i32MemRef);

		const float identity = 1.0f;
		// fullC = true: MVOUT writes back the full i32 accumulator. With
		// fullC=false the lowering would shrink-store i8, which is wrong when
		// the destination memref is i32 (libgemmini packs 4 i8 bytes into one
		// i32 slot in that case — symptom: each output cell looks like
		// 0x08080808 = packed [8,8,8,8] instead of plain 8).
		bool fullC = false;
		if (auto outElemBits =
				cast<MemRefType>(out.getType()).getElementTypeBitWidth();
			outElemBits >= 32) {
			fullC = true;
		}

		// 2026-05-18 B-TRANSPOSE FIX: inspect the linalg.matmul's indexing
		// maps to detect when B is laid out N×K (transposed) instead of
		// K×N. IREE's preprocessing pipeline folds `linalg.transpose` of
		// B into the matmul by rewriting the rhs indexing map from the
		// default `(d0,d1,d2) -> (d2,d1)` (K,N) to `(d0,d1,d2) ->
		// (d1,d2)` (N,K). The named `linalg.matmul` op accepts both
		// forms via the `indexing_maps` attribute, but `spTiledMatmulOs`
		// in LegalizeForLLVMExport assumes B is K×N. Without setting
		// bTranspose=true on the resulting tile_matmul, dronet's
		// matmul_16x128x64 and matmul_49x64x32 dispatches (K<N) read B
		// with wrong row-stride and produce deterministic garbage —
		// exactly the bit-stable wrong hash we've been chasing.
		bool aTranspose = false;
		bool bTranspose = false;
		{
			auto maps = op.getIndexingMapsArray();
			MLIRContext *ctx = op.getContext();
			AffineExpr d0 = getAffineDimExpr(0, ctx);
			AffineExpr d1 = getAffineDimExpr(1, ctx);
			AffineExpr d2 = getAffineDimExpr(2, ctx);
			AffineMap lhsKxNMap = AffineMap::get(3, 0, {d0, d2}, ctx);
			AffineMap lhsNxKMap = AffineMap::get(3, 0, {d2, d0}, ctx);
			AffineMap rhsKxNMap = AffineMap::get(3, 0, {d2, d1}, ctx);
			AffineMap rhsNxKMap = AffineMap::get(3, 0, {d1, d2}, ctx);
			if (maps.size() == 3) {
				if (maps[0] == lhsNxKMap)
					aTranspose = true;
				else if (maps[0] != lhsKxNMap) {
					return rewriter.notifyMatchFailure(
						op, "unrecognized lhs indexing map");
				}
				if (maps[1] == rhsNxKMap)
					bTranspose = true;
				else if (maps[1] != rhsKxNMap) {
					return rewriter.notifyMatchFailure(
						op, "unrecognized rhs indexing map");
				}
			}
		}

		// 2026-05-18 DATAFLOW + B-TRANSPOSE FIX:
		//  - OS dataflow + b_transpose-alone hangs FireSim Shuttle
		//    Gemmini (verified 2026-05-25 by walking both WS+LOOP_WS
		//    and OS+CONFIG_EX.bTranspose=1 paths — both hang).
		//  - WS / gemminiLoopWs path also hangs (Load-misaligned fault).
		// Decline to lower b-only-transposed matmuls here; instead the
		// frontend should run --iree-preprocessing-transpose-matmul-pass
		// (or equivalent) to fold the bTranspose into an explicit
		// linalg.transpose, leaving a regular K×N matmul we can lower.
		// dronet's b-transposed dispatches that aren't pre-transposed
		// will run on CPU (correct, not gemmini-accelerated).
		if (bTranspose && !aTranspose) {
			return rewriter.notifyMatchFailure(op,
				"b-only-transposed matmul: gemmini OS path hangs and "
				"WS / LOOP_WS path also hangs on FireSim Shuttle. "
				"Use --iree-preprocessing-transpose-matmul-pass to "
				"fold bTranspose into an explicit linalg.transpose.");
		}

		Dataflow dataflow = Dataflow::OutputStationary;

		// MATMUL+RESCALE FUSION (2026-05-26): if the matmul's i32 result is
		// consumed by the canonical sitofp→mulf→divf→roundeven→addf→
		// maximumf→minimumf→fptosi rescale chain producing i8, lower both as
		// one TileMatMul with `fullC=false` + `accScale=in_scale/out_scale`.
		// Gemmini applies that scale during MVOUT, so we cut the i32 MVOUT
		// (4× the bandwidth of i8) and drop the separate CPU/RVV rescale
		// dispatch entirely. Without the fold, IREE keeps the matmul writing
		// i32 to a transient buffer and emits a follow-up dispatch — the
		// gap that made dronet's matmul_3136x32x27 take 16.3M cycles vs the
		// canonical chipyard equivalent's 28K.
		Value cArray = out;
		Value dArray = emptyD;
		llvm::APFloat accScale(identity);
		bool fullCFinal = fullC;
		bool repeatingBias = true;
		int64_t actAttr = 0; // NO_ACTIVATION
		std::optional<RescaleFoldMatch> fold = tryMatchI32ToI8Rescale(op);
		std::optional<ChainedResidualFoldMatch> chainedFold;
		std::optional<HeadRescaleSplitMatch> headSplit;
		if (fold.has_value()) {
			cArray = fold->outI8;
			accScale = llvm::APFloat(fold->accScale);
			fullCFinal = false;
			if (fold->relu)
				actAttr = 1; // RELU
			if (fold->biasAddOp) {
				if (fold->biasIsBroadcast) {
					// Bias is rank-1 N-element; reshape it to a 1×N row
					// view and let Gemmini's repeatingBias=true replay it
					// across all M output rows (stride-0 MVIN-D).
					auto bTy = cast<MemRefType>(fold->biasMemRef.getType());
					assert(bTy.getRank() == 1);
					auto reshaped = MemRefType::get(
						{1, bTy.getShape()[0]}, bTy.getElementType());
					SmallVector<ReassociationIndices> reassoc = {{0, 1}};
					dArray = rewriter.create<memref::ExpandShapeOp>(
						loc, reshaped, fold->biasMemRef, reassoc);
					repeatingBias = true;
				} else {
					// Full MxN bias — forward as-is with strideD = N elems.
					dArray = fold->biasMemRef;
					repeatingBias = false;
				}
			}
		} else {
			// CHAINED-RESIDUAL PARTIAL FOLD (2026-05-26 Opt#3): a residual-
			// block matmul (dronet's dispatch_11/20/29) is followed by a
			// 20-op 2-input rescale generic that combines (matmul → first
			// rescale to i8 intermediate) and (intermediate-i8 → f32 +
			// residual-i8 → f32, sum, second rescale to i8). Bit-exact
			// full fold is blocked by the intermediate clip, but we can
			// collapse the matmul side: emit a tile_matmul with `accScale
			// = scale_in/scale_int` writing i8 to a fresh alloca, then
			// rewrite the chained generic to read that i8 alloca instead
			// of the original i32 destination (the new body's first 8 ops
			// are gone — the input bb-arg IS already the intermediate i8).
			chainedFold = tryMatchChainedResidualRescale(op);
			if (chainedFold.has_value()) {
				auto destTy = cast<MemRefType>(out.getType());
				auto i8MemTy = MemRefType::get(
					destTy.getShape(), rewriter.getIntegerType(8));
				Value i8Alloca =
					rewriter.create<memref::AllocaOp>(loc, i8MemTy);
				cArray = i8Alloca;
				accScale = llvm::APFloat(chainedFold->accScale);
				fullCFinal = false;
				if (chainedFold->biasAddOp) {
					dArray = chainedFold->biasMemRef;
					repeatingBias = false;
				}
			} else {
				// HEAD-RESCALE SPLIT (2026-05-26): yolov8n's matmul+SiLU
				// dispatches put a 28-op chain in one linalg.generic — the
				// first 8 ops are a canonical QDQ rescale, the rest is
				// SiLU + final rescale. Split off the head: emit
				// tile_matmul writing i8 to a fresh alloca with
				// accScale=in/int, then rewrite the chained generic to
				// read the alloca with the first 8 ops dropped (bb-arg
				// becomes the i8 intermediate directly).
				headSplit = tryMatchHeadRescaleAndSplit(op);
				if (headSplit.has_value()) {
					auto destTy = cast<MemRefType>(out.getType());
					auto i8MemTy = MemRefType::get(
						destTy.getShape(), rewriter.getIntegerType(8));
					Value i8Alloca =
						rewriter.create<memref::AllocaOp>(loc, i8MemTy);
					cArray = i8Alloca;
					accScale = llvm::APFloat(headSplit->accScale);
					fullCFinal = false;
					if (headSplit->biasAddOp) {
						dArray = headSplit->biasMemRef;
						repeatingBias = false;
					}
				}
			}
		}

		rewriter.create<TileMatMulOp>(loc, lhs, rhs, cArray, dArray,
			/*aScaleFactor=*/llvm::APFloat(identity),
			/*bScaleFactor=*/llvm::APFloat(identity),
			/*dScaleFactor=*/llvm::APFloat(identity),
			/*act=*/actAttr,
			/*accScale=*/accScale,
			/*bertScale=*/llvm::APFloat(0.0f),
			/*repeatingBias=*/repeatingBias,
			/*aTranspose=*/aTranspose,
			/*bTranspose=*/bTranspose,
			/*fullC=*/fullCFinal,
			/*lowD=*/false,
			/*weightA=*/0,
			/*dataflow=*/static_cast<int64_t>(dataflow));
		if (fold.has_value()) {
			rewriter.eraseOp(fold->rescaleOp);
			if (fold->biasAddOp)
				rewriter.eraseOp(fold->biasAddOp);
		}
		if (chainedFold.has_value()) {
			// Rebuild the residual-add + final rescale generic on top of
			// the new i8 matmul output. Same indexing maps / iter types as
			// the original chained generic, but the first input is the i8
			// alloca (cArray) instead of the i32 matmul destination, and
			// the body's first 8 ops (the i32→i8 rescale) are dropped —
			// the bb arg is already the intermediate i8.
			OpBuilder::InsertionGuard g(rewriter);
			rewriter.setInsertionPoint(chainedFold->rescaleOp);
			Value resVal = chainedFold->rescaleOp.getDpsInputs()[1];
			Value outBuf = chainedFold->rescaleOp.getDpsInits()[0];
			SmallVector<AffineMap> maps =
				chainedFold->rescaleOp.getIndexingMapsArray();
			SmallVector<utils::IteratorType> iter =
				chainedFold->rescaleOp.getIteratorTypesArray();
			Value scaleResVal = chainedFold->scaleResVal;
			Value scaleIntVal = chainedFold->scaleIntVal;
			Value scaleOutVal = chainedFold->scaleOutVal;
			Value zeroVal = chainedFold->zeroVal;
			Value cMinVal = chainedFold->clipLoVal;
			Value cMaxVal = chainedFold->clipHiVal;
			rewriter.create<linalg::GenericOp>(chainedFold->rescaleOp.getLoc(),
				/*resultTensorTypes=*/TypeRange{},
				/*inputs=*/ValueRange{cArray, resVal},
				/*outputs=*/ValueRange{outBuf},
				/*indexingMaps=*/maps,
				/*iteratorTypes=*/iter,
				[&](OpBuilder &b, Location nl, ValueRange args) {
					Value mm = args[0]; // i8 (intermediate)
					Value res = args[1]; // i8 (residual)
					Value rf =
						b.create<arith::SIToFPOp>(nl, b.getF32Type(), res);
					Value rs = b.create<arith::MulFOp>(nl, rf, scaleResVal);
					Value mi =
						b.create<arith::SIToFPOp>(nl, b.getF32Type(), mm);
					Value is = b.create<arith::MulFOp>(nl, mi, scaleIntVal);
					Value sumv = b.create<arith::AddFOp>(nl, is, rs);
					Value j = b.create<arith::DivFOp>(nl, sumv, scaleOutVal);
					Value k = b.create<math::RoundEvenOp>(nl, j);
					Value l = b.create<arith::AddFOp>(nl, k, zeroVal);
					Value m = b.create<arith::MaximumFOp>(nl, l, cMinVal);
					Value n = b.create<arith::MinimumFOp>(nl, m, cMaxVal);
					Value o =
						b.create<arith::FPToSIOp>(nl, b.getIntegerType(8), n);
					b.create<linalg::YieldOp>(nl, o);
				});
			rewriter.eraseOp(chainedFold->rescaleOp);
			if (chainedFold->biasAddOp)
				rewriter.eraseOp(chainedFold->biasAddOp);
		}
		if (headSplit.has_value()) {
			// Replace the chained generic with a new generic that reads
			// the i8 alloca and skips the first 8 ops of the original
			// body. The bb-arg of the new generic IS the post-rescale i8
			// intermediate, so wherever the original body referenced the
			// fptosi result (ops[7]), the new body uses the bb-arg.
			linalg::GenericOp old = headSplit->chainedOp;
			Block &oldBody = old.getRegion().front();
			SmallVector<Operation *> oldOps;
			for (Operation &nested : oldBody.without_terminator())
				oldOps.push_back(&nested);
			auto oldYield = cast<linalg::YieldOp>(oldBody.getTerminator());
			Value oldFptosiResult = oldOps[7]->getResult(0);
			Value outBuf = old.getDpsInits()[0];
			SmallVector<AffineMap> maps = old.getIndexingMapsArray();
			SmallVector<utils::IteratorType> iter = old.getIteratorTypesArray();
			auto outTy = cast<MemRefType>(outBuf.getType());
			OpBuilder::InsertionGuard g(rewriter);
			rewriter.setInsertionPoint(old);
			linalg::GenericOp tail =
				rewriter.create<linalg::GenericOp>(old.getLoc(),
					/*resultTensorTypes=*/TypeRange{},
					/*inputs=*/ValueRange{cArray},
					/*outputs=*/ValueRange{outBuf},
					/*indexingMaps=*/maps,
					/*iteratorTypes=*/iter,
					/*bodyBuild=*/nullptr);
			Block *newBlock = rewriter.createBlock(&tail.getRegion(),
				tail.getRegion().end(),
				TypeRange{rewriter.getIntegerType(8), outTy.getElementType()},
				SmallVector<Location>{old.getLoc(), old.getLoc()});
			rewriter.setInsertionPointToStart(newBlock);
			IRMapping mapping;
			mapping.map(oldFptosiResult, newBlock->getArgument(0));
			for (size_t i = headSplit->headOpCount; i < oldOps.size(); ++i)
				rewriter.clone(*oldOps[i], mapping);
			Value yieldedVal = oldYield.getOperand(0);
			rewriter.create<linalg::YieldOp>(
				old.getLoc(), mapping.lookupOrDefault(yieldedVal));
			rewriter.eraseOp(old);
			if (headSplit->biasAddOp)
				rewriter.eraseOp(headSplit->biasAddOp);
		}
		rewriter.eraseOp(op);
		return success();
	}
};

struct LowerConv2DToISA final : OpRewritePattern<Conv2DOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Conv2DOp op, PatternRewriter &rewriter) const override {
		// Conv2D requires a 4-rank memref input (NHWC) plus weights (FHWC) +
		// bias (F). Without bufferization there's nothing useful to do.
		auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
		auto filterType = dyn_cast<MemRefType>(op.getFilter().getType());
		auto outType = dyn_cast<MemRefType>(op.getResult().getType());
		if (!inputType || !filterType || !outType)
			return rewriter.notifyMatchFailure(op,
				"lower-tile-to-isa requires memref operands; bufferize "
				"first");
		// Conv2DOp has stride/dilation/zp attrs; map them onto TileConvOp.
		Location loc = op.getLoc();
		auto i32MemRef = MemRefType::get({0}, rewriter.getI32Type());
		Value emptyBias = rewriter.create<memref::AllocaOp>(loc, i32MemRef);

		// We don't have all the metadata `tile_conv` needs (out dims, kernel
		// dims) as constants on Conv2DOp. Pull them from filter/result types.
		ArrayRef<int64_t> outShape = outType.getShape();
		ArrayRef<int64_t> filterShape = filterType.getShape();
		Value outRowDim = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(outShape[1]));
		Value outColDim = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(outShape[2]));
		Value kernelDim = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(filterShape[1]));

		rewriter.create<TileConvOp>(loc, op.getInput(), op.getResult(),
			op.getFilter(), emptyBias, outRowDim, outColDim, kernelDim,
			/*stride=*/op.getStrideH(),
			/*inputDilation=*/static_cast<int64_t>(1),
			/*kernelDilation=*/op.getDilationH(),
			/*padding=*/static_cast<int64_t>(0),
			/*act=*/static_cast<int64_t>(0),
			/*scale=*/llvm::APFloat(1.0f),
			/*poolSize=*/static_cast<int64_t>(0),
			/*poolStride=*/static_cast<int64_t>(0),
			/*poolPadding=*/static_cast<int64_t>(0),
			/*wrot180=*/false,
			/*transOutput1203=*/false,
			/*transInput3120=*/false,
			/*transWeight1203=*/false,
			/*transWeight0132=*/false);
		rewriter.eraseOp(op);
		return success();
	}
};

// Opt #E (2026-05-26): native tile_conv lowering — inside-dispatch pattern.
// Matches a bufferized 6-loop `linalg.generic` int8 conv2d (HWC input, HWCF
// filter, HWC i32 output) and consumes its downstream bias-add + i32→i8
// rescale + optional HWC→CHW transpose `linalg.generic` chain. Emits
// `gemmini.tile_conv` with bias + scale, then either:
//   - if the rescale's output indexing is identity (HWC i8 result): erases
//     the rescale generic and lets tile_conv write directly to its outs.
//   - if there's a transpose (HWC → CHW): rewrites the rescale generic
//     in-place to be an i8-only transpose (body becomes `linalg.yield
//     %in`), reading from a fresh HWC i8 alloca that tile_conv writes.
//
// This eliminates the CPU-side im2col + bias-add + rescale dispatches that
// the matmul-via-img2col path needed (dronet dispatch_0/1/5/etc. plus
// hundreds of similar in yolov8n).
struct LowerBufferizedLinalgConvToTileConv final
	: OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern::OpRewritePattern;

	// Recognize a bufferized 6-loop int8 conv2d body:
	//   ^bb0(%in: i8, %fil: i8, %out: i32):
	//     %a = arith.extsi %in : i8 to i32
	//     %b = arith.extsi %fil : i8 to i32
	//     %c = arith.muli %a, %b : i32
	//     %d = arith.addi %out, %c : i32
	//     linalg.yield %d : i32
	static bool matchConvBody(linalg::GenericOp op) {
		Block &body = op.getRegion().front();
		SmallVector<Operation *> ops;
		for (Operation &nested : body.without_terminator())
			ops.push_back(&nested);
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return false;
		if (ops.size() != 4)
			return false;
		auto sf0 = dyn_cast<arith::ExtSIOp>(ops[0]);
		auto sf1 = dyn_cast<arith::ExtSIOp>(ops[1]);
		auto mul = dyn_cast<arith::MulIOp>(ops[2]);
		auto add = dyn_cast<arith::AddIOp>(ops[3]);
		if (!sf0 || !sf1 || !mul || !add)
			return false;
		Value in = body.getArgument(0);
		Value fil = body.getArgument(1);
		Value out = body.getArgument(2);
		if (sf0.getIn() != in)
			return false;
		if (sf1.getIn() != fil)
			return false;
		if (mul.getLhs() != sf0.getResult() || mul.getRhs() != sf1.getResult())
			return false;
		if ((add.getLhs() != out || add.getRhs() != mul.getResult()) &&
			(add.getRhs() != out || add.getLhs() != mul.getResult()))
			return false;
		return yield.getOperand(0) == add.getResult();
	}

	// Match the indexing maps of a 6-loop HWC conv:
	//   in:     (oh, ow, oc, fh, fw, ic) -> (oh*sH + fh, ow*sW + fw, ic)
	//   filter: (oh, ow, oc, fh, fw, ic) -> (fh, fw, ic, oc)
	//   out:    (oh, ow, oc, fh, fw, ic) -> (oh, ow, oc)
	// Returns the stride (assumed sH == sW).
	static std::optional<int64_t> matchConvMaps(linalg::GenericOp op) {
		auto maps = op.getIndexingMapsArray();
		if (maps.size() != 3)
			return std::nullopt;
		// Output map: (d0..d5) -> (d0, d1, d2). Identity on the 3 parallel
		// dims, drops the 3 reduction dims.
		AffineMap outMap = maps[2];
		if (outMap.getNumResults() != 3)
			return std::nullopt;
		MLIRContext *ctx = op.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineExpr d2 = getAffineDimExpr(2, ctx);
		AffineExpr d3 = getAffineDimExpr(3, ctx);
		AffineExpr d4 = getAffineDimExpr(4, ctx);
		AffineExpr d5 = getAffineDimExpr(5, ctx);
		if (outMap.getResult(0) != d0 || outMap.getResult(1) != d1 ||
			outMap.getResult(2) != d2)
			return std::nullopt;
		// Filter map: (fh, fw, ic, oc) = (d3, d4, d5, d2).
		AffineMap filMap = maps[1];
		if (filMap.getNumResults() != 4)
			return std::nullopt;
		if (filMap.getResult(0) != d3 || filMap.getResult(1) != d4 ||
			filMap.getResult(2) != d5 || filMap.getResult(3) != d2)
			return std::nullopt;
		// Input map: (oh*S + fh, ow*S + fw, ic). Extract S from coefficient.
		AffineMap inMap = maps[0];
		if (inMap.getNumResults() != 3)
			return std::nullopt;
		if (inMap.getResult(2) != d5)
			return std::nullopt;
		auto extractStride = [&](AffineExpr e, AffineExpr base,
								 AffineExpr offset) -> std::optional<int64_t> {
			auto add = dyn_cast<AffineBinaryOpExpr>(e);
			if (!add || add.getKind() != AffineExprKind::Add)
				return std::nullopt;
			AffineExpr lhs = add.getLHS();
			AffineExpr rhs = add.getRHS();
			if (rhs != offset && lhs != offset)
				return std::nullopt;
			AffineExpr scaled = (rhs == offset) ? lhs : rhs;
			// scaled == S * base
			auto mul = dyn_cast<AffineBinaryOpExpr>(scaled);
			if (!mul || mul.getKind() != AffineExprKind::Mul)
				return std::nullopt;
			auto cst = dyn_cast<AffineConstantExpr>(mul.getLHS());
			AffineExpr otherSide = mul.getRHS();
			if (!cst) {
				cst = dyn_cast<AffineConstantExpr>(mul.getRHS());
				otherSide = mul.getLHS();
			}
			if (!cst)
				return std::nullopt;
			if (otherSide != base)
				return std::nullopt;
			return cst.getValue();
		};
		auto sH = extractStride(inMap.getResult(0), d0, d3);
		auto sW = extractStride(inMap.getResult(1), d1, d4);
		if (!sH || !sW || *sH != *sW)
			return std::nullopt;
		return *sH;
	}

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.hasPureTensorSemantics())
			return rewriter.notifyMatchFailure(op, "needs bufferized form");
		if (op.getNumLoops() != 6)
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto iters = op.getIteratorTypesArray();
		int nPar = 0, nRed = 0;
		for (auto k : iters) {
			if (k == utils::IteratorType::parallel)
				++nPar;
			else if (k == utils::IteratorType::reduction)
				++nRed;
		}
		if (nPar != 3 || nRed != 3)
			return failure();

		auto inTy = dyn_cast<MemRefType>(op.getDpsInputs()[0].getType());
		auto filTy = dyn_cast<MemRefType>(op.getDpsInputs()[1].getType());
		auto outTy = dyn_cast<MemRefType>(op.getDpsInits()[0].getType());
		if (!inTy || !inTy.getElementType().isSignlessInteger(8))
			return failure();
		if (!filTy || !filTy.getElementType().isSignlessInteger(8))
			return failure();
		if (!outTy || !outTy.getElementType().isSignlessInteger(32))
			return failure();
		if (inTy.getRank() != 3 || filTy.getRank() != 4 || outTy.getRank() != 3)
			return failure();
		if (!matchConvBody(op))
			return failure();
		auto strideOpt = matchConvMaps(op);
		if (!strideOpt)
			return failure();
		int64_t stride = *strideOpt;

		// Single-use safety on the conv's destination buffer (the pre-fill
		// linalg.fill is OK; the bias-add+rescale consumer must be the only
		// other reader/writer).
		Value convDest = op.getDpsInits()[0];
		linalg::FillOp preFill;
		linalg::GenericOp tail;
		for (Operation *user : convDest.getUsers()) {
			if (user == op.getOperation())
				continue;
			if (auto f = dyn_cast<linalg::FillOp>(user)) {
				if (preFill || !user->isBeforeInBlock(op.getOperation()))
					return failure();
				preFill = f;
				continue;
			}
			if (auto g = dyn_cast<linalg::GenericOp>(user)) {
				if (tail || !op->isBeforeInBlock(g.getOperation()))
					return failure();
				tail = g;
				continue;
			}
			return failure();
		}
		if (!tail)
			return failure();

		// The tail must take (conv_i32, bias_i32) and write i8.
		if (tail.getNumDpsInputs() != 2 || tail.getNumDpsInits() != 1)
			return failure();
		auto tailOutTy = dyn_cast<MemRefType>(tail.getDpsInits()[0].getType());
		if (!tailOutTy || !tailOutTy.getElementType().isSignlessInteger(8))
			return failure();
		int convIdx = -1, biasIdx = -1;
		if (tail.getDpsInputs()[0] == convDest) {
			convIdx = 0;
			biasIdx = 1;
		} else if (tail.getDpsInputs()[1] == convDest) {
			convIdx = 1;
			biasIdx = 0;
		} else {
			return failure();
		}
		auto biasTy =
			dyn_cast<MemRefType>(tail.getDpsInputs()[biasIdx].getType());
		if (!biasTy || !biasTy.getElementType().isSignlessInteger(32))
			return failure();
		// Bias must be either rank-1 (Fxi32) with broadcast indexing, OR
		// rank-3 (HxWxF) with identity indexing (we'll prefer rank-1).
		auto tailMaps = tail.getIndexingMapsArray();
		if (tailMaps.size() != 3)
			return failure();
		if (!tailMaps[convIdx].isIdentity())
			return failure();
		// Verify tail body matches one of two shapes:
		//  (A) 9 ops, no ReLU: addi(conv, bias) → sitofp → mulf scale_in →
		//      divf scale_out → roundeven → addf 0 → max -128 → min 127 →
		//      fptosi.
		//  (B) 17 ops, with ReLU: case (A) + cmpi sgt 0 → select → sitofp →
		//      roundeven → addf 0 → max -128 → min 127 → fptosi.
		Block &tbody = tail.getRegion().front();
		SmallVector<Operation *> tops;
		for (Operation &nested : tbody.without_terminator())
			tops.push_back(&nested);
		auto tyield = dyn_cast<linalg::YieldOp>(tbody.getTerminator());
		if (!tyield)
			return failure();
		// Opt#E coverage gate: ONLY accept the 9-op (no-ReLU) rescale chain.
		// The 17-op (with-ReLU) chain runs a SECOND rescale after the ReLU
		// (clamp → cmpi sgt 0 → select → sitofp → roundeven → addf → max →
		// min → fptosi). libgemmini's tile_conv applies activation AFTER its
		// single output scale, not BEFORE a second rescale — so emitting
		// `act = RELU` here drops the second-rescale step and silently
		// produces wrong output (hash drift confirmed on dronet 2026-05-26
		// FireSim run). Leave those convs on the img2col→matmul path until
		// we can correctly fuse the second rescale into the matmul output.
		bool tailRelu = false;
		if (tops.size() != 9) {
			return failure();
		}
		auto addi = dyn_cast<arith::AddIOp>(tops[0]);
		auto sf2 = dyn_cast<arith::SIToFPOp>(tops[1]);
		auto mf = dyn_cast<arith::MulFOp>(tops[2]);
		auto df = dyn_cast<arith::DivFOp>(tops[3]);
		auto rd = dyn_cast<math::RoundEvenOp>(tops[4]);
		auto af = dyn_cast<arith::AddFOp>(tops[5]);
		auto mx = dyn_cast<arith::MaximumFOp>(tops[6]);
		auto mn = dyn_cast<arith::MinimumFOp>(tops[7]);
		auto fp = dyn_cast<arith::FPToSIOp>(tops[8]);
		if (!addi || !sf2 || !mf || !df || !rd || !af || !mx || !mn || !fp)
			return failure();
		Value convBB = tbody.getArgument(convIdx);
		Value biasBB = tbody.getArgument(biasIdx);
		Value lhsA = addi.getLhs(), rhsA = addi.getRhs();
		if (!((lhsA == convBB && rhsA == biasBB) ||
				(lhsA == biasBB && rhsA == convBB)))
			return failure();
		if (sf2.getIn() != addi.getResult())
			return failure();
		if (mf.getLhs() != sf2.getResult())
			return failure();
		if (df.getLhs() != mf.getResult())
			return failure();
		if (rd.getOperand() != df.getResult())
			return failure();
		if (af.getLhs() != rd.getResult())
			return failure();
		if (mx.getLhs() != af.getResult())
			return failure();
		if (mn.getLhs() != mx.getResult())
			return failure();
		if (fp.getIn() != mn.getResult())
			return failure();
		if (!tailRelu) {
			if (tyield.getOperand(0) != fp.getResult())
				return failure();
		} else {
			// Walk the ReLU + second-rescale tail (ops 9-16).
			auto cmpi = dyn_cast<arith::CmpIOp>(tops[9]);
			auto sel = dyn_cast<arith::SelectOp>(tops[10]);
			auto sfR = dyn_cast<arith::SIToFPOp>(tops[11]);
			auto rdR = dyn_cast<math::RoundEvenOp>(tops[12]);
			auto afR = dyn_cast<arith::AddFOp>(tops[13]);
			auto mxR = dyn_cast<arith::MaximumFOp>(tops[14]);
			auto mnR = dyn_cast<arith::MinimumFOp>(tops[15]);
			auto fpR = dyn_cast<arith::FPToSIOp>(tops[16]);
			if (!cmpi || !sel || !sfR || !rdR || !afR || !mxR || !mnR || !fpR)
				return failure();
			if (cmpi.getPredicate() != arith::CmpIPredicate::sgt)
				return failure();
			if (cmpi.getLhs() != fp.getResult())
				return failure();
			if (sel.getTrueValue() != fp.getResult())
				return failure();
			if (sfR.getIn() != sel.getResult())
				return failure();
			if (rdR.getOperand() != sfR.getResult())
				return failure();
			if (afR.getLhs() != rdR.getResult())
				return failure();
			if (mxR.getLhs() != afR.getResult())
				return failure();
			if (mnR.getLhs() != mxR.getResult())
				return failure();
			if (fpR.getIn() != mnR.getResult())
				return failure();
			if (tyield.getOperand(0) != fpR.getResult())
				return failure();
		}

		auto fconst = [&](Value v) -> std::optional<double> {
			auto cst = v.getDefiningOp<arith::ConstantOp>();
			if (!cst)
				return std::nullopt;
			auto attr = dyn_cast<FloatAttr>(cst.getValue());
			return attr ? std::optional<double>(attr.getValueAsDouble())
						: std::nullopt;
		};
		auto zp = fconst(af.getRhs());
		auto qmin = fconst(mx.getRhs());
		auto qmax = fconst(mn.getRhs());
		if (!zp || !qmin || !qmax)
			return failure();
		if (*zp != 0.0 || *qmin != -128.0 || *qmax != 127.0)
			return failure();
		auto scaleIn = fconst(mf.getRhs());
		auto scaleOut = fconst(df.getRhs());
		if (!scaleIn || !scaleOut || *scaleOut == 0.0)
			return failure();
		double net = *scaleIn / *scaleOut;

		// Output layout: tail's output indexing map determines whether the
		// output is HWC (identity) or transposed (e.g., HWC→CHW). Currently
		// only handle identity output; if transposed, emit tile_conv into a
		// fresh HWC alloca and rewrite the tail to a pure i8 transpose.
		bool tailIsTranspose = !tailMaps[2].isIdentity();

		Location loc = op.getLoc();
		// GemminiTileConvLowering expects rank-4 NHWC input/output (see
		// LegalizeForLLVMExport.cpp:3172-3175 — accesses inputShape[0..3]).
		// dronet/yolov8n's bufferized conv body uses rank-3 HWC (no batch
		// dim). Wrap with `memref.expand_shape` adding N=1.
		auto expandToNHWC = [&](Value v) -> Value {
			auto t = cast<MemRefType>(v.getType());
			if (t.getRank() == 4)
				return v;
			SmallVector<int64_t> newShape;
			newShape.push_back(1);
			for (int64_t d : t.getShape())
				newShape.push_back(d);
			SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
			// Use computeExpandedType so the strided layout (when reading
			// from a subspan with offset) and memory-space attribute
			// propagate correctly.
			auto computed = memref::ExpandShapeOp::computeExpandedType(
				t, newShape, reassoc);
			MemRefType newTy = succeeded(computed)
				? *computed
				: MemRefType::get(newShape, t.getElementType(),
					  MemRefLayoutAttrInterface{}, t.getMemorySpace());
			return rewriter.create<memref::ExpandShapeOp>(
				loc, newTy, v, reassoc);
		};
		Value inputNHWC = expandToNHWC(op.getDpsInputs()[0]);
		Value tileConvOut;
		Value hwcAlloca;
		if (tailIsTranspose) {
			auto hwcTy =
				MemRefType::get(outTy.getShape(), rewriter.getIntegerType(8));
			hwcAlloca = rewriter.create<memref::AllocaOp>(loc, hwcTy);
			tileConvOut = expandToNHWC(hwcAlloca);
		} else {
			tileConvOut = expandToNHWC(tail.getDpsInits()[0]);
		}

		// Bias plumbing. TileConvOp expects a memref bias (rank-1 F).
		Value biasMemRef = tail.getDpsInputs()[biasIdx];

		// Emit the tile_conv. Out dims pulled from the output type; kernel
		// dim from the filter.
		IntegerType i64Ty = rewriter.getI64Type();
		Value outRowDim = rewriter.create<arith::ConstantOp>(
			loc, i64Ty, rewriter.getI64IntegerAttr(outTy.getShape()[0]));
		Value outColDim = rewriter.create<arith::ConstantOp>(
			loc, i64Ty, rewriter.getI64IntegerAttr(outTy.getShape()[1]));
		Value kernelDim = rewriter.create<arith::ConstantOp>(
			loc, i64Ty, rewriter.getI64IntegerAttr(filTy.getShape()[0]));
		rewriter.create<TileConvOp>(loc,
			inputNHWC, // input (NHWC)
			tileConvOut, // output (NHWC i8)
			op.getDpsInputs()[1], // weights (HWCF)
			biasMemRef, // bias (F)
			outRowDim, outColDim, kernelDim,
			/*stride=*/stride,
			/*inputDilation=*/(int64_t)1,
			/*kernelDilation=*/(int64_t)1,
			/*padding=*/(int64_t)0,
			/*act=*/(int64_t)(tailRelu ? 1 : 0),
			/*scale=*/llvm::APFloat((float)net),
			/*poolSize=*/(int64_t)0,
			/*poolStride=*/(int64_t)0,
			/*poolPadding=*/(int64_t)0,
			/*wrot180=*/false,
			/*transOutput1203=*/false,
			/*transInput3120=*/false,
			/*transWeight1203=*/false,
			/*transWeight0132=*/false);

		if (tailIsTranspose) {
			// Replace the tail with a pure i8 transpose: same indexing maps,
			// ins(hwc_alloca), outs(original_outs), body = yield %in.
			OpBuilder::InsertionGuard g(rewriter);
			rewriter.setInsertionPoint(tail);
			SmallVector<AffineMap> newMaps = {tailMaps[convIdx], tailMaps[2]};
			SmallVector<utils::IteratorType> newIter;
			for (auto t : tail.getIteratorTypesArray())
				newIter.push_back(t);
			rewriter.create<linalg::GenericOp>(tail.getLoc(),
				/*resultTensorTypes=*/TypeRange{},
				/*inputs=*/ValueRange{hwcAlloca},
				/*outputs=*/ValueRange{tail.getDpsInits()[0]},
				/*indexingMaps=*/newMaps,
				/*iteratorTypes=*/newIter,
				[&](OpBuilder &b, Location nl, ValueRange args) {
					b.create<linalg::YieldOp>(nl, args[0]);
				});
		}
		// Erase the tail (replaced) and the conv body.
		rewriter.eraseOp(tail);
		if (preFill)
			rewriter.eraseOp(preFill);
		rewriter.eraseOp(op);
		return success();
	}
};

// Opt#G — recognize yolov8n's `matmul_like_*` swapped-operand `linalg.generic`
// as an inside-dispatch matmul and emit a canonical `linalg.matmul` (memref)
// so the existing `LowerBufferizedLinalgMatmulToTileMatmul` + Opt#0 rescale
// fold can take over. The flow-level dispatch name `matmul_like_32x6400x32`
// (and ~20 similar shapes in yolov8n) hides this exact form:
//   linalg.generic {
//     indexing_maps = [(d0,d1,d2) -> (d2,d1),  // op0 = B in K×N layout
//                      (d0,d1,d2) -> (d0,d2),  // op1 = A in M×K layout
//                      (d0,d1,d2) -> (d0,d1)], // out = M×N
//     iterator_types = ["parallel", "parallel", "reduction"]
//   } ins(op0, op1) outs(out)  {
//     ^bb0(%in: i8, %in_: i8, %out: i32):
//       %a = arith.extsi %in
//       %b = arith.extsi %in_
//       %c = arith.muli %a, %b
//       %d = arith.addi %out, %c
//       linalg.yield %d
//   }
// We swap the operands and emit a `linalg.matmul` with custom maps
// [(d0,d2), (d2,d1), (d0,d1)] — same canonical "b-only-transposed" form
// that MaterializeBTransposeAsLinalgTransposePattern (Opt#3) produces.
// LowerBufferizedLinalgMatmulToTileMatmul's bTranspose auto-detection
// (lhsK == rhsDim0 ⇒ B is K×N, set bTranspose=true) handles the layout.
struct CanonicalizeSwappedMatmulLikeGeneric final
	: OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp gen, PatternRewriter &rewriter) const override {
		if (gen.hasPureTensorSemantics())
			return failure(); // memref only
		if (gen.getNumLoops() != 3)
			return failure();
		if (gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1)
			return failure();
		auto iters = gen.getIteratorTypesArray();
		if (iters.size() != 3)
			return failure();
		if (iters[0] != utils::IteratorType::parallel ||
			iters[1] != utils::IteratorType::parallel ||
			iters[2] != utils::IteratorType::reduction)
			return failure();

		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 3)
			return failure();
		MLIRContext *ctx = gen.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineExpr d2 = getAffineDimExpr(2, ctx);
		AffineMap swapLhsMap = AffineMap::get(3, 0, {d2, d1}, ctx);
		AffineMap swapRhsMap = AffineMap::get(3, 0, {d0, d2}, ctx);
		AffineMap outMap = AffineMap::get(3, 0, {d0, d1}, ctx);
		if (maps[0] != swapLhsMap || maps[1] != swapRhsMap || maps[2] != outMap)
			return failure();

		// Body: extsi/extsi/muli/addi/yield.
		Block &body = gen.getRegion().front();
		if (body.getNumArguments() != 3)
			return failure();
		SmallVector<Operation *> ops;
		for (Operation &nested : body.without_terminator())
			ops.push_back(&nested);
		if (ops.size() != 4)
			return failure();
		auto sf0 = dyn_cast<arith::ExtSIOp>(ops[0]);
		auto sf1 = dyn_cast<arith::ExtSIOp>(ops[1]);
		auto mul = dyn_cast<arith::MulIOp>(ops[2]);
		auto add = dyn_cast<arith::AddIOp>(ops[3]);
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!sf0 || !sf1 || !mul || !add || !yield)
			return failure();
		Value bb0 = body.getArgument(0);
		Value bb1 = body.getArgument(1);
		Value bb2 = body.getArgument(2);
		if (sf0.getIn() != bb0 || sf1.getIn() != bb1)
			return failure();
		if (mul.getLhs() != sf0.getResult() || mul.getRhs() != sf1.getResult())
			return failure();
		if (!((add.getLhs() == bb2 && add.getRhs() == mul.getResult()) ||
				(add.getRhs() == bb2 && add.getLhs() == mul.getResult())))
			return failure();
		if (yield.getOperand(0) != add.getResult())
			return failure();

		Value op0 = gen.getDpsInputs()[0];
		Value op1 = gen.getDpsInputs()[1];
		Value out = gen.getDpsInits()[0];
		auto op0Ty = dyn_cast<MemRefType>(op0.getType());
		auto op1Ty = dyn_cast<MemRefType>(op1.getType());
		auto outTy = dyn_cast<MemRefType>(out.getType());
		if (!op0Ty || !op1Ty || !outTy)
			return failure();
		if (op0Ty.getRank() != 2 || op1Ty.getRank() != 2 ||
			outTy.getRank() != 2)
			return failure();
		if (!op0Ty.getElementType().isSignlessInteger(8))
			return failure();
		if (!op1Ty.getElementType().isSignlessInteger(8))
			return failure();
		if (!outTy.getElementType().isSignlessInteger(32))
			return failure();

		// Emit canonical linalg.matmul with operands swapped (op1 = A as
		// lhs, op0 = B as rhs in K×N) and the b-only-transposed indexing
		// maps. Downstream LowerBufferizedLinalgMatmulToTileMatmul picks
		// it up; its existing bTranspose auto-detection (rhsDim0 == K)
		// handles the layout.
		Location loc = gen.getLoc();
		AffineMap canonLhs = AffineMap::get(3, 0, {d0, d2}, ctx);
		AffineMap canonRhs = AffineMap::get(3, 0, {d2, d1}, ctx);
		SmallVector<AffineMap> newMaps = {canonLhs, canonRhs, outMap};
		auto matmul = rewriter.create<linalg::MatmulOp>(loc,
			TypeRange{}, // memref form returns no results
			ValueRange{op1, op0}, ValueRange{out});
		matmul.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
		rewriter.eraseOp(gen);
		return success();
	}
};

struct GemminiLowerTileToISAPass final
	: public impl::GemminiLowerTileToISAPassBase<GemminiLowerTileToISAPass> {
	GemminiLowerTileToISAPass() = default;
	explicit GemminiLowerTileToISAPass(const GemminiTransformOptions &options)
		: options(options) {}

	void runOnOperation() override {
		auto func = getOperation();
		RewritePatternSet patterns(&getContext());
		patterns
			.add<LowerMatmulTileToISA, LowerBufferizedLinalgMatmulToTileMatmul,
				LowerConv2DToISA, LowerBufferizedLinalgConvToTileConv,
				CanonicalizeSwappedMatmulLikeGeneric>(&getContext());
		if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
			signalPassFailure();
		}

		// If the function now contains any TileMatMul/TileConv op, ensure a
		// final flush is emitted so the LegalizeForLLVMExport's per-op fence
		// is paired with a hardware-level FLUSH_SKIP=0.
		if (auto fn = dyn_cast<func::FuncOp>(func.getOperation())) {
			bool hasTileOp = false;
			fn.walk([&](Operation *op) {
				if (isa<TileMatMulOp, TileConvOp>(op))
					hasTileOp = true;
			});
			if (hasTileOp && !fn.getBody().empty()) {
				OpBuilder builder(&getContext());
				appendFlushEpilogue(fn, builder);
			}
		}
	}

	GemminiTransformOptions options;
};

} // namespace

// createGemminiLowerTileToISAPass() is auto-generated as a friend of
// GemminiLowerTileToISAPassBase by GEN_PASS_DEF_*; do not redeclare.

std::unique_ptr<Pass> createGemminiLowerTileToISAPassWithOptions(
	const GemminiTransformOptions &options) {
	return std::make_unique<GemminiLowerTileToISAPass>(options);
}

} // namespace mlir::iree_compiler::Gemmini
