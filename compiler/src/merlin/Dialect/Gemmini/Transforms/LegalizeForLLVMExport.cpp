//===- LegalizeForLLVMExport.cpp - Prepare Gemmini for LLVM translation ---===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Original File sourced and modified from
// https://github.com/buddy-compiler/buddy-mlir
//
//===----------------------------------------------------------------------===//

#include <cstdlib>

#include "llvm/Support/Format.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.h"
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Transforms.h"
#include "third_party/iree_bar/runtime/src/iree/hal/local/loaders/merlin_debug_addresses.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree_compiler::Gemmini;

namespace {

int64_t getNumberFromValue(Value &value) {
	return dyn_cast<IntegerAttr>(value.getDefiningOp()->getAttr("value"))
		.getInt();
}

acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
	union {
		acc_scale_t_bits b;
		acc_scale_t f;
	} un;

	un.f = x;
	return un.b;
}

scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
	union {
		scale_t_bits b;
		scale_t f;
	} un;

	un.f = x;
	return un.b;
}

// This function is used to insert a fence operation to ensure that the
// NPU and CPU memory operations are executed in the correct order.
// Use sequentially consistent ordering for strongest memory guarantee.
void insertFence(Location loc, ConversionPatternRewriter &rewriter) {
	auto ordering = LLVM::AtomicOrdering::seq_cst;
	rewriter.create<LLVM::FenceOp>(loc, ordering);
}

// FIX (2026-05-19) — gemmini-subspan-offset-dropped: when the memref
// operand of `gemmini.tile_matmul` comes from
// `hal.interface.binding.subspan offset(%c<N>)`, IREE's LLVMCPU lowering
// uses `MemRefDescriptor::fromStaticShape` whenever the bufferized
// memref's StridedLayoutAttr is fully static. That helper stores the
// raw binding base pointer in both `allocated_ptr` and `aligned_ptr`
// fields, and parks the static offset in the descriptor's `offset`
// slot. The Gemmini lowering reads `aligned_ptr` via
// `memref::ExtractAlignedPointerAsIndexOp` — so it never sees the
// subspan's `byte_offset` operand, and the resulting `mvin` reads
// bytes from rodata + 0 instead of rodata + N. Bindings that carry the
// `Indirect` flag are unaffected because the runtime pre-applies the
// offset to the binding pointer, but bare `ReadOnly` bindings reach
// Gemmini's mvin path with the offset still owed.
//
// `walkBackToSubspanByteOffset` walks the def-use chain backward from
// `operand` through view/cast/reinterpret/extract_strided_metadata ops
// to the originating `IREE::HAL::InterfaceBindingSubspanOp`, and
// returns its `byte_offset` SSA value (or {} if the chain doesn't lead
// to a subspan cleanly). The caller adds it to the pointer extracted
// from the memref descriptor.
static std::optional<Value> walkBackToSubspanByteOffset(Value operand) {
	Value cur = operand;
	// Bound iterations to avoid cycles in pathological IR.
	for (int hop = 0; hop < 16; ++hop) {
		if (!cur) return std::nullopt;
		Operation *def = cur.getDefiningOp();
		if (!def) return std::nullopt;
		if (auto subspan =
				dyn_cast<mlir::iree_compiler::IREE::HAL::
					InterfaceBindingSubspanOp>(def)) {
			// FIX (2026-05-21): always apply byte_offset, regardless of
			// the Indirect flag. Empirical evidence (bprobe traces) shows
			// the IREE local-task runtime does NOT pre-resolve the
			// byte_offset for Indirect bindings (or if it does, the
			// downstream codegen still applies its own GEP-based offset).
			// The standard memref->LLVM lowering ALWAYS applies the
			// byte_offset via the memref descriptor's offset field. For
			// the Gemmini MVIN/MVOUT lowering to be consistent with the
			// readers/writers in adjacent CPU dispatches, we MUST apply
			// the same offset here. The previous 2026-05-20 patch (skip
			// for Indirect) caused MVOUT to write at base+0 while the
			// next dispatch's standard codegen read at base+byte_offset,
			// silently dropping the matmul output — producing the long-
			// observed "constant" wrong i32 for dronet d16/d18.
			return subspan.getByteOffset();
		}
		// Common view-like ops that preserve the underlying allocation.
		if (auto vli = dyn_cast<ViewLikeOpInterface>(def)) {
			cur = vli.getViewSource();
			continue;
		}
		// Fall back to the first operand for unrecognised single-operand
		// passthrough ops (UnrealizedConversionCastOp, builtin.cast, etc.).
		if (def->getNumOperands() == 1) {
			cur = def->getOperand(0);
			continue;
		}
		return std::nullopt;
	}
	return std::nullopt;
}

template <typename IntrOp = MvinIntrOp>
void gemminiMvinOffset(const Value &mem, const size_t offset,
	const uint32_t SpAddr, const size_t cols, const size_t rows,
	int64_t addrLen, ConversionPatternRewriter &rewriter) {
	Location loc = mem.getLoc();
	Value offsetOp = rewriter.create<arith::ConstantOp>(
		loc, rewriter.getI64IntegerAttr(offset));
	IntegerType i64Type = rewriter.getI64Type();
	Value configPtr =
		rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
	uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
		(uint64_t)cols << addrLen | (uint64_t)SpAddr;
	Value spad = rewriter.create<arith::ConstantOp>(
		loc, rewriter.getI64IntegerAttr(spadAddrInt));
	rewriter.create<IntrOp>(loc, configPtr, spad);
}

void gemminiMvoutOffset(const Value &mem, const size_t offset,
	const uint32_t SpAddr, const size_t cols, const size_t rows,
	int64_t addrLen, ConversionPatternRewriter &rewriter) {
	Location loc = mem.getLoc();
	Value offsetOp = rewriter.create<arith::ConstantOp>(
		loc, rewriter.getI64IntegerAttr(offset));
	IntegerType i64Type = rewriter.getI64Type();
	Value configPtr =
		rewriter.create<arith::AddIOp>(loc, i64Type, mem, offsetOp);
	uint64_t spadAddrInt = (uint64_t)rows << (addrLen + 16) |
		(uint64_t)cols << addrLen | (uint64_t)SpAddr;
	Value spad = rewriter.create<arith::ConstantOp>(
		loc, rewriter.getI64IntegerAttr(spadAddrInt));
	rewriter.create<MvoutIntrOp>(loc, configPtr, spad);
}

} // namespace

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
	using OpConversionPattern<OpTy>::OpConversionPattern;

	LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
		ConversionPatternRewriter &rewriter) const final {
		if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
			return rewriter.notifyMatchFailure(
				op, "operand types already match");
		rewriter.modifyOpInPlace(
			op, [&]() { op->setOperands(adaptor.getOperands()); });
		return success();
	}
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
  public:
	using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const final {
		rewriter.modifyOpInPlace(
			op, [&]() { op->setOperands(adaptor.getOperands()); });
		return success();
	}
};

struct GemminiFlushLowering : public ConvertOpToLLVMPattern<FlushOp> {
	using ConvertOpToLLVMPattern<FlushOp>::ConvertOpToLLVMPattern;
	LogicalResult matchAndRewrite(FlushOp flushOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Location loc = flushOp.getLoc();
		Value skip = flushOp.getSkip();
		IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(0);
		Value rs2 = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64Type(), rs2Attr);
		rewriter.replaceOpWithNewOp<FlushIntrOp>(flushOp, skip, rs2);
		return success();
	}
};

struct GemminiConfigStLowering : public ConvertOpToLLVMPattern<ConfigStOp> {
	using ConvertOpToLLVMPattern<ConfigStOp>::ConvertOpToLLVMPattern;
	LogicalResult matchAndRewrite(ConfigStOp configStOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value strideValue = configStOp.getStride();
		int stride = getNumberFromValue(strideValue);
		float scale = configStOp.getScale().convertToFloat();
		Location loc = configStOp.getLoc();
		uint64_t rs1 = ((uint64_t)configStOp.getActivation() << 2) | CONFIG_ST;
		uint64_t arg =
			(uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)scale)
				<< 32 |
			(uint32_t)stride;
		Value value1 = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value value2 = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(arg));
		rewriter.replaceOpWithNewOp<ConfigIntrOp>(configStOp, value1, value2);
		return success();
	}
};

struct GemminiConfigLdLowering : public ConvertOpToLLVMPattern<ConfigLdOp> {
	using ConvertOpToLLVMPattern<ConfigLdOp>::ConvertOpToLLVMPattern;
	explicit GemminiConfigLdLowering(
		LLVMTypeConverter &typeConverter, int64_t dim)
		: ConvertOpToLLVMPattern(typeConverter), dim(dim) {}
	LogicalResult matchAndRewrite(ConfigLdOp configLdOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value rs2Value = configLdOp.getStride();
		float scale = configLdOp.getScale().convertToFloat();
		uint64_t blockMvinStride = configLdOp.getBlockMvinStride();
		if (blockMvinStride == (uint64_t)-1)
			blockMvinStride = dim;
		uint64_t pixelRepeats = configLdOp.getPixelRepeats();
		uint64_t rs1 = (uint64_t)scale_t_to_scale_t_bits(scale) << 32 |
			(blockMvinStride << 16) | pixelRepeats << 8 |
			configLdOp.getId() << 3 | configLdOp.getShrunk() << 2 | CONFIG_LD;
		Location loc = configLdOp.getLoc();
		Value rs1value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		rewriter.replaceOpWithNewOp<ConfigIntrOp>(
			configLdOp, rs1value, rs2Value);
		return success();
	}

  private:
	int64_t dim;
};

struct GemminiConfigExLowering : public ConvertOpToLLVMPattern<ConfigExOp> {
	GemminiConfigExLowering(LLVMTypeConverter &converter, int64_t mxFormat)
		: ConvertOpToLLVMPattern<ConfigExOp>(converter), mxFormat(mxFormat) {}

	LogicalResult matchAndRewrite(ConfigExOp configExOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		IntegerType i64Type = rewriter.getI64Type();
		Location loc = configExOp.getLoc();
		float scale = configExOp.getSysAccScale().convertToFloat();
		// mxGemmini CONFIG_EX bits (gemmini.h:269-284,
		// MxParameters.scala:124-130). When mxFormat==-1 (Disabled) all
		// MX bits are zero — vanilla-Gemmini behavior preserved
		// byte-identically.
		uint64_t mxBits = 0;
		if (mxFormat >= 0) {
			// 2026-05-08: empirically derived from matmul_tiled_fp8.c
			// (gemmini-mx@69a1c038, software/gemmini-rocc-tests/bareMetalC),
			// which is the only working FP8 matmul reference and does PASS
			// on RadianceGemminiOnlyConfig in 17 s wallclock. Its CONFIG_EX
			// is `gemmini_extended3_config_ex(WS, 0, 0, ACC_SCALE_IDENTITY,
			// 1, 1, 0, 0, false, 0, 0, 3, 0)` — i.e.
			// act_mx_fmt=0, wgt_mx_fmt=0, out_mx_fmt=3, uselut=0.
			// Despite both inputs being 8-bit FP8 (e4m3) data, the
			// hardware expects act=wgt=0 (the "8-bit-input" PE-mode keyed
			// by code=0 in MxFormats.scala when the spatial array runs
			// in narrow_type=true mode driven by the input width 8). The
			// out_mx_fmt=3 selects the BF16 dequant output mode used by
			// the requantizer's mvout path. matmul_ws_mx_generic.c uses
			// (1,1,1,1) for FP6 and (1) here would also be valid, but
			// (0,0,3,0) is the path the upstream FP8 test exercises.
			//
			// Bundle layout per gemmini.h's gemmini_extended3_config_ex
			// macro comment + GemminiISA.scala::ConfigExRs1 (LSB→MSB):
			//   [1:0] cmd_type, [2] dataflow, [4:3] activation,
			//   [5] uselut, [6] _spacer0, [7] set_only_strides,
			//   [8] a_transpose, [9] b_transpose,
			//   [11:10] act_mx_fmt, [13:12] wgt_mx_fmt,
			//   [15:14] out_mx_fmt.
			//
			// We treat any non-Disabled mxFormat as "FP8 mode" for now;
			// FP6 / FP4 would need different (act,wgt,out,uselut) tuples
			// + the LUT load/CONFIG_SCALE_MEM scaffolding from
			// matmul_ws_mx_generic.c.
			const uint64_t actMxFmt = 0;
			const uint64_t wgtMxFmt = 0;
			const uint64_t outMxFmt = 3;
			mxBits = (actMxFmt << 10)
				| (wgtMxFmt << 12)
				| (outMxFmt << 14);
		}
		uint64_t rs1 = (uint64_t)acc_scale_t_to_acc_scale_t_bits(scale) << 32 |
			configExOp.getAStride() << 16 | mxBits |
			configExOp.getBTranspose() << 9 |
			configExOp.getATranspose() << 8 |
			configExOp.getSetOnlyStrides() << 7 | configExOp.getSysAct() << 3 |
			configExOp.getDataflow() << 2 | CONFIG_EX;

		uint64_t rs2 = configExOp.getCStride() << 48 | configExOp.getSysShift();
		IntegerAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
		IntegerAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
		Value rs1Value =
			rewriter.create<arith::ConstantOp>(loc, i64Type, rs1Attr);
		Value rs2Value =
			rewriter.create<arith::ConstantOp>(loc, i64Type, rs2Attr);
		rewriter.replaceOpWithNewOp<ConfigIntrOp>(
			configExOp, rs1Value, rs2Value);
		return success();
	}

  private:
	int64_t mxFormat;
};

struct GemminiConfigNormLowering : public ConvertOpToLLVMPattern<ConfigNormOp> {
	using ConvertOpToLLVMPattern<ConfigNormOp>::ConvertOpToLLVMPattern;
	LogicalResult matchAndRewrite(ConfigNormOp configNormOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Location loc = configNormOp.getLoc();
		uint64_t rs1 =
			(((uint64_t)((uint32_t)configNormOp.getQConst())) << 32) |
			(configNormOp.getQConstType() & 1) << 18 |
			(configNormOp.getSetStatsIdOnly() & 1) << 17 |
			(configNormOp.getActMsb() & 1) << 16 |
			configNormOp.getStatsId() << 8 | CONFIG_BERT;
		uint64_t rs2 =
			(((uint64_t)((uint32_t)configNormOp.getIgeluQc())) << 32) |
			((uint64_t)((uint32_t)configNormOp.getIgeluQb()));
		Value rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs2));
		rewriter.replaceOpWithNewOp<ConfigIntrOp>(
			configNormOp, rs1Value, rs2Value);
		return success();
	}
};

struct GemminiMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
	using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;
	explicit GemminiMvinLowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(MvinOp mvinOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value input = mvinOp.getInput();
		Location loc = input.getLoc();
		MemRefType memRefType =
			dyn_cast<MemRefType>(mvinOp.getOperandTypes().front());
		llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
		TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
		Value extractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, resultType, input);
		IntegerType i64Type = rewriter.getI64Type();
		Value indexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
		Value spadAddrValue = mvinOp.getAddr();
		uint64_t number = getNumberFromValue(spadAddrValue);
		uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
			(uint64_t)memRefShape[1] << addrLen | number;
		Value spad = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(spadAddrInt));
		rewriter.replaceOpWithNewOp<MvinIntrOp>(mvinOp, indexCastOp, spad);
		return success();
	}

  private:
	int64_t addrLen;
};

struct GemminiMvin2Lowering : public ConvertOpToLLVMPattern<Mvin2Op> {
	using ConvertOpToLLVMPattern<Mvin2Op>::ConvertOpToLLVMPattern;
	explicit GemminiMvin2Lowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(Mvin2Op mvin2Op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value input = mvin2Op.getInput();
		Location loc = input.getLoc();
		MemRefType memRefType =
			dyn_cast<MemRefType>(mvin2Op.getOperandTypes().front());
		llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
		TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
		Value extractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, resultType, input);
		IntegerType i64Type = rewriter.getI64Type();
		Value indexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
		Value spadAddrValue = mvin2Op.getAddr();
		uint64_t number = getNumberFromValue(spadAddrValue);
		uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
			(uint64_t)memRefShape[1] << addrLen | number;
		Value spad = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(spadAddrInt));
		rewriter.replaceOpWithNewOp<Mvin2IntrOp>(mvin2Op, indexCastOp, spad);
		return success();
	}

  private:
	int64_t addrLen;
};

struct GemminiMvin3Lowering : public ConvertOpToLLVMPattern<Mvin3Op> {
	using ConvertOpToLLVMPattern<Mvin3Op>::ConvertOpToLLVMPattern;
	explicit GemminiMvin3Lowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(Mvin3Op mvin3Op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value input = mvin3Op.getInput();
		Location loc = input.getLoc();
		MemRefType memRefType =
			dyn_cast<MemRefType>(mvin3Op.getOperandTypes().front());
		llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
		TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
		Value extractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, resultType, input);
		IntegerType i64Type = rewriter.getI64Type();
		Value indexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
		Value spadAddrValue = mvin3Op.getAddr();
		uint64_t number = getNumberFromValue(spadAddrValue);
		uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
			(uint64_t)memRefShape[1] << addrLen | number;
		Value spad = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(spadAddrInt));
		rewriter.replaceOpWithNewOp<Mvin3IntrOp>(mvin3Op, indexCastOp, spad);
		return success();
	}

  private:
	int64_t addrLen;
};

struct GemminiMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
	using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;
	explicit GemminiMvoutLowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(MvoutOp mvoutOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value output = mvoutOp.getOutput();
		TypeRange resultType = mlir::TypeRange(rewriter.getIndexType());
		Location loc = mvoutOp.getLoc();
		Value extractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, resultType, output);
		IntegerType i64Type = rewriter.getI64Type();
		Value indexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, extractOp);
		Value spadAddr = mvoutOp.getAddr();
		uint64_t number = getNumberFromValue(spadAddr);
		MemRefType memRefType =
			dyn_cast<MemRefType>(mvoutOp.getOperandTypes().front());
		llvm::ArrayRef<int64_t> memRefShape = memRefType.getShape();
		uint64_t spadAddrInt = (uint64_t)memRefShape[0] << (addrLen + 16) |
			(uint64_t)memRefShape[1] << addrLen | number;
		Value newSpad = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(spadAddrInt));
		rewriter.replaceOpWithNewOp<MvoutIntrOp>(
			mvoutOp, indexCastOp, newSpad);
		return success();
	}

  private:
	int64_t addrLen;
};

struct GemminiPreloadZerosLowering
	: public ConvertOpToLLVMPattern<PreloadZerosOp> {
	using ConvertOpToLLVMPattern<PreloadZerosOp>::ConvertOpToLLVMPattern;
	explicit GemminiPreloadZerosLowering(
		LLVMTypeConverter &typeConverter, int64_t dim, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(PreloadZerosOp preloadZerosOp,
		OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
		Value addr = preloadZerosOp.getAddr();
		Value cRows = preloadZerosOp.getCRows();
		Value cCols = preloadZerosOp.getCCols();
		Location loc = preloadZerosOp.getLoc();
		uint64_t addrInt = getNumberFromValue(addr);
		uint64_t cRowsInt = getNumberFromValue(cRows);
		uint64_t cColsInt = getNumberFromValue(cCols);
		uint64_t rs1 = (uint64_t)dim << (addrLen + 16) |
			(uint64_t)dim << addrLen | (uint64_t)-1;
		uint64_t rs2 =
			cRowsInt << (addrLen + 16) | cColsInt << (addrLen) | addrInt;
		Value rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs2));
		rewriter.replaceOpWithNewOp<PreloadIntrOp>(
			preloadZerosOp, rs1Value, rs2Value);
		return success();
	}

  private:
	int64_t dim;
	int64_t addrLen;
};

struct GemminiPreloadLowering : public ConvertOpToLLVMPattern<PreloadOp> {
	using ConvertOpToLLVMPattern<PreloadOp>::ConvertOpToLLVMPattern;
	explicit GemminiPreloadLowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(PreloadOp preloadOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value bdAddr = preloadOp.getBdAddr();
		Value cAddr = preloadOp.getCAddr();
		Value bdCols = preloadOp.getBdCols();
		Value bdRows = preloadOp.getBdRows();
		Value cCols = preloadOp.getCCols();
		Value cRows = preloadOp.getCRows();
		Location loc = preloadOp.getLoc();
		uint64_t bdAddrInt = getNumberFromValue(bdAddr);
		uint64_t cAddrInt = getNumberFromValue(cAddr);
		uint64_t bdColsInt = getNumberFromValue(bdCols);
		uint64_t bdRowsInt = getNumberFromValue(bdRows);
		uint64_t cColsInt = getNumberFromValue(cCols);
		uint64_t cRowsInt = getNumberFromValue(cRows);
		uint64_t rs1 = bdRowsInt << (addrLen + 16) | bdColsInt << addrLen |
			(uint64_t)bdAddrInt;
		uint64_t rs2 = cRowsInt << (addrLen + 16) | cColsInt << addrLen |
			(uint64_t)cAddrInt;
		Value rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs2));
		rewriter.replaceOpWithNewOp<PreloadIntrOp>(
			preloadOp, rs1Value, rs2Value);
		return success();
	}

  private:
	int64_t addrLen;
};

struct GemminiComputePreloadedLowering
	: public ConvertOpToLLVMPattern<ComputePreloadedOp> {
	using ConvertOpToLLVMPattern<ComputePreloadedOp>::ConvertOpToLLVMPattern;
	explicit GemminiComputePreloadedLowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(ComputePreloadedOp computePreloadedOp,
		OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
		Value aAddr = computePreloadedOp.getAAddr();
		Value bdAddr = computePreloadedOp.getBdAddr();
		Value aRows = computePreloadedOp.getARows();
		Value aCols = computePreloadedOp.getACols();
		Value bdRows = computePreloadedOp.getBdRows();
		Value bdCols = computePreloadedOp.getBdCols();
		Location loc = computePreloadedOp.getLoc();
		uint64_t aAddrInt = getNumberFromValue(aAddr);
		uint64_t bdAddrInt = getNumberFromValue(bdAddr);
		uint64_t aRowsInt = getNumberFromValue(aRows);
		uint64_t aColsInt = getNumberFromValue(aCols);
		uint64_t bdRowsInt = getNumberFromValue(bdRows);
		uint64_t bdColsInt = getNumberFromValue(bdCols);
		uint64_t rs1 =
			aRowsInt << (addrLen + 16) | aColsInt << addrLen | aAddrInt;
		uint64_t rs2 =
			bdRowsInt << (addrLen + 16) | bdColsInt << addrLen | bdAddrInt;
		Value rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs2));
		rewriter.replaceOpWithNewOp<ComputePreloadedIntrOp>(
			computePreloadedOp, rs1Value, rs2Value);
		return success();
	}

  private:
	int64_t addrLen;
};

struct GemminiComputeAccumulatedLowering
	: public ConvertOpToLLVMPattern<ComputeAccumulatedOp> {
	using ConvertOpToLLVMPattern<ComputeAccumulatedOp>::ConvertOpToLLVMPattern;
	explicit GemminiComputeAccumulatedLowering(
		LLVMTypeConverter &typeConverter, int64_t addrLen)
		: ConvertOpToLLVMPattern(typeConverter), addrLen(addrLen) {}
	LogicalResult matchAndRewrite(ComputeAccumulatedOp computeAccumulatedOp,
		OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
		Value aAddr = computeAccumulatedOp.getAAddr();
		Value bdAddr = computeAccumulatedOp.getBdAddr();
		Value aRows = computeAccumulatedOp.getARows();
		Value aCols = computeAccumulatedOp.getACols();
		Value bdRows = computeAccumulatedOp.getBdRows();
		Value bdCols = computeAccumulatedOp.getBdCols();
		Location loc = computeAccumulatedOp.getLoc();
		uint64_t aAddrInt = getNumberFromValue(aAddr);
		uint64_t bdAddrInt = getNumberFromValue(bdAddr);
		uint64_t aRowsInt = getNumberFromValue(aRows);
		uint64_t aColsInt = getNumberFromValue(aCols);
		uint64_t bdRowsInt = getNumberFromValue(bdRows);
		uint64_t bdColsInt = getNumberFromValue(bdCols);
		uint64_t rs1 =
			aRowsInt << (addrLen + 16) | aColsInt << addrLen | aAddrInt;
		uint64_t rs2 =
			bdRowsInt << (addrLen + 16) | bdColsInt << addrLen | bdAddrInt;
		Value rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs2));
		rewriter.replaceOpWithNewOp<ComputeAccumulatedIntrOp>(
			computeAccumulatedOp, rs1Value, rs2Value);

		return success();
	}

  private:
	int64_t addrLen;
};

class GemminiTileMatMulLowering : public ConvertOpToLLVMPattern<TileMatMulOp> {
	void gemminiLoopWs(size_t i, size_t j, size_t k, size_t padI, size_t padJ,
		size_t padK, Value &a, Value &b, Value &d, Value &c, size_t aRowStride,
		size_t bRowStride, size_t dRowStride, size_t cRowStride,
		bool aTranspose, bool bTranspose, bool fullC, bool lowD,
		bool exAccumulate, int act, TileMatMulOp &tileMatMulOp,
		ConversionPatternRewriter &rewriter) const {
		// loopWsConfigBounds instruction.
		uint64_t rs1 =
			(uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
		uint64_t rs2 = (uint64_t)k << 32 | (uint64_t)j << 16 | (uint64_t)i;
		IntegerType i64Type = rewriter.getI64Type();
		Location loc = a.getLoc();
		Value rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs1));
		Value rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(rs2));
		rewriter.create<LoopWsConfigBoundsIntrOp>(loc, rs1Value, rs2Value);
		// loopWsConfigAddrsAB instruction.
		rewriter.create<LoopWsConfigAddrsABIntrOp>(loc, a, b);
		// loopWsConfigAddrsDC instruction
		rewriter.create<LoopWsConfigAddrsDCIntrOp>(loc, d, c);
		// loopWsConfigStridesAB instruction
		rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(aRowStride));
		rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(bRowStride));
		rewriter.create<LoopWsConfigStridesABIntrOp>(loc, rs1Value, rs2Value);
		// loopWsConfigStrideDC instruction
		rs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(dRowStride));
		rs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(cRowStride));
		rewriter.create<LoopWsConfigStridesDCIntrOp>(loc, rs1Value, rs2Value);
		const int aSpadId = 0;
		const int bSpadId = 0;
		const int isResadd = 0;
		rs1 = (uint64_t)aSpadId << 18 | (uint64_t)bSpadId << 16 |
			(uint64_t)act << 8 | lowD << 2 | (fullC) << 1 | exAccumulate;
		rs2 = isResadd << 2 | bTranspose << 1 | aTranspose;
		rs1Value = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(rs1));
		rs2Value = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(rs2));
		rewriter.create<LoopWsIntrOp>(loc, rs1Value, rs2Value);
	}

	void spTiledMatmulWs(Value &a, Value &b, Value &d, Value &c,
		scale_t aScaleFactor, scale_t bScaleFactor, scale_acc_t dScaleFactor,
		size_t i, size_t j, size_t k, size_t padI, size_t padJ, size_t padK,
		size_t strideA, size_t strideB, size_t strideD, size_t strideC,
		bool aTranspose, bool bTranspose, bool fullC, bool lowD, bool noBias,
		bool repeatingBias, int act, TileMatMulOp &tileMatMulOp,
		ConversionPatternRewriter &rewriter) const {

		gemminiLoopWs(i, j, k, padI, padJ, padK, a, b, d, c, strideA, strideB,
			repeatingBias ? 0 : strideD, strideC, aTranspose, bTranspose, fullC,
			lowD, !noBias, act, tileMatMulOp, rewriter);
	}

	// Tiling functions
	void spTiledMatmulOs(Value &a, Value &b, Value &d, Value &c,
		scale_t aScaleFactor, scale_t bScaleFactor, scale_acc_t dScaleFactor,
		size_t i, size_t j, size_t k, size_t padI, size_t padJ, size_t padK,
		size_t strideA, size_t strideB, size_t strideD, size_t strideC,
		bool aTranspose, bool bTranspose, bool fullC, bool lowD, bool noBias,
		bool repeatingBias, int act, TileMatMulOp &tileMatMulOp,
		ConversionPatternRewriter &rewriter) const {
		const uint32_t aSpAddrStart = 0;
		const uint32_t bSpAddrStart = BANK_NUM * bankRows - k * j * dim;
		const uint32_t dSpAddrStart = 1 << (addrLen - 1);
		const uint32_t cSpAddrStart =
			(3 << (addrLen - 2)) | (fullC << (addrLen - 3));

		// On hardware with a narrow MvinRs2.num_cols field
		// (mxGemmini-MMIO via RadianceGemminiOnlyConfig: only 6 bits, so
		// cols max is 63), `blocks * dim` must stay below the field cap.
		// Clamp to 1 so each MVIN issues with cols == dim (= 16 for our
		// configs). See `clampSingleBlockMvin` member comment for full
		// context.
		const size_t maxBlockLen =
			clampSingleBlockMvin ? 1u : (MAX_BYTES / (dim * 1));
		const size_t maxBlockLenAcc =
			clampSingleBlockMvin ? 1u : (MAX_BYTES / (dim * 4));

		const int aBlocks = k <= maxBlockLen ? k : maxBlockLen;
		const int bBlocks = j <= maxBlockLen ? j : maxBlockLen;
		const int dBlocks = j <= maxBlockLenAcc ? j : maxBlockLenAcc;

		Location loc = a.getLoc();
		bool dAddrNull = llvm::dyn_cast<arith::ConstantOp>(d.getDefiningOp()) &&
			getNumberFromValue(d) == 0;
		bool cAddrNull = llvm::dyn_cast<arith::ConstantOp>(c.getDefiningOp()) &&
			getNumberFromValue(c) == 0;

		// 2026-05-21 BUG A FIX (Phase 2 — coherency fence): emit fence
		// rw,rw BEFORE the first MVIN. This guarantees the worker CPU's
		// preceding stores (including LowerTileToISA's zero-init of the
		// D buffer on the stack) are visible to Gemmini's DMA at the
		// L2/DRAM coherency point. Without this fence, the CPU's L1
		// store buffer can still hold those zero-init values while
		// MVIN-D fires; the DMA reads stale (uninitialized) stack
		// memory and the accumulator picks up garbage as bias →
		// consistent wrong matmul output for dronet (the conv stack
		// reuses the same stack region across dispatches so the bytes
		// happen to be "consistent" garbage).
		// embedded_elf_loader.c also issues a `fence rw,rw` BEFORE
		// the dispatch ELF call, but that's separate — it covers the
		// runtime's binding-ptrs setup, not the dispatch's internal
		// stack stores.
		insertFence(loc, rewriter);

		// Move-in D — use port 0 (default MVIN / k_MVIN), CONFIG_LD id=0.
		// 2026-05-18: REVERTED the earlier mvin3/id=2 split. Upstream
		// `sp_tiled_matmul_os` in gemmini.h:447-467 uses
		// `gemmini_extended_mvin` (port 0, id=0 default) for D, B, AND A.
		// FireSim Shuttle's bitstream load controller does NOT distinguish
		// per-port stride state for the OS dataflow: issuing mvin2/mvin3
		// reads from uninitialised state_id slots, producing garbage
		// (-532341 observed for matmul_1x1x2048 with all-ones inputs vs
		// expected 2048). Match upstream: single port 0 + single
		// CONFIG_LD slot 0 reconfigured before each operand burst.
		if (!dAddrNull && !noBias) {
			const size_t dStride = repeatingBias ? 0 : strideD * sizeOfAccT;
			Value strideValue = rewriter.create<arith::ConstantOp>(
				loc, rewriter.getI64IntegerAttr(dStride));
			rewriter.create<ConfigLdOp>(
				loc, strideValue, llvm::APFloat((float)dScaleFactor));

			for (size_t i0 = 0; i0 < i; i0++) {
				for (size_t j0 = 0; j0 < j; j0 += dBlocks) {
					const size_t biasRow = repeatingBias ? 0 : i0;
					const size_t offset =
						(biasRow * strideD + j0) * dim * sizeOfAccT;
					const uint32_t dSpAddrAcc =
						dSpAddrStart + (i0 * j + j0) * dim;
					const size_t blocks = j0 + dBlocks <= j ? dBlocks : j - j0;
					const size_t cols =
						blocks * dim - (j0 + blocks >= j ? padJ : 0);
					const size_t rows = dim - (i0 == i - 1 ? padI : 0);
					gemminiMvinOffset(
						d, offset, dSpAddrAcc, cols, rows, addrLen, rewriter);
				}
			}
		}


		// Move-in B — same port 0, id=0 (see comment on Move-in D above).
		Value strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideB));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, llvm::APFloat((float)bScaleFactor));
		for (size_t j0 = 0; j0 < j; j0 += bBlocks) {
			for (size_t k0 = 0; k0 < k; k0++) {
				const size_t offset = (k0 * strideB + j0) * dim * sizeOfElemT;
				const uint32_t bSpAddr = bSpAddrStart + (k0 * j + j0) * dim;
				const size_t blocks = j0 + bBlocks <= j ? bBlocks : j - j0;
				const size_t cols =
					blocks * dim - (j0 + blocks >= j ? padJ : 0);
				const size_t rows = dim - (k0 == k - 1 ? padK : 0);
				gemminiMvinOffset(
					b, offset, bSpAddr, cols, rows, addrLen, rewriter);
			}
		}


		// Move-in A
		strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideA));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, llvm::APFloat((float)aScaleFactor));

		for (size_t i0 = 0; i0 < i; i0++) {
			for (size_t k0 = 0; k0 < k; k0 += aBlocks) {
				const size_t offset = (i0 * strideA + k0) * dim * sizeOfElemT;
				const uint32_t aSpAddr = aSpAddrStart + (i0 * k + k0) * dim;
				const size_t blocks = k0 + aBlocks <= k ? aBlocks : k - k0;
				const size_t cols =
					blocks * dim - (k0 + blocks >= k ? padK : 0);
				const size_t rows = dim - (i0 == i - 1 ? padI : 0);
				gemminiMvinOffset(
					a, offset, aSpAddr, cols, rows, addrLen, rewriter);
			}
		}


		for (size_t i0 = 0; i0 < i; i0++) {
			for (size_t j0 = 0; j0 < j; j0++) {
				const uint32_t cSpAddr = cSpAddrStart + (i0 * j + j0) * dim;
				for (size_t k0 = 0; k0 < k; k0++) {

					const uint32_t aSpAddr = aSpAddrStart + (i0 * k + k0) * dim;
					const uint32_t bSpAddr = bSpAddrStart + (k0 * j + j0) * dim;

					uint32_t outSpAddr = k0 == k - 1 ? cSpAddr : GARBAGE_ADDR;

					// 2026-05-21: previously tried k0 == 0 here (per agent
					// hypothesis), produced identical wrong hashes on FireSim.
					// Reverted to k0 == k - 1 which matches chipyard's
					// sp_tiled_matmul_os at gemmini.h:508. (WS path at
					// gemmini.h:638 uses k == 0, but OS path uses K-1.)
					// The bug is elsewhere — see tmp/bug_a_*.md.
					int noBiasNewMatrix = noBias && !dAddrNull && k0 == k - 1;
					if (noBiasNewMatrix) {
						outSpAddr &= ~(1 << (addrLen - 2));
					}

					const size_t aCols = dim - (k0 == k - 1 ? padK : 0);
					const size_t aRows = dim - (i0 == i - 1 ? padI : 0);
					const size_t bCols = dim - (j0 == j - 1 ? padJ : 0);
					const size_t bRows = dim - (k0 == k - 1 ? padK : 0);
					const size_t cCols = dim - (j0 == j - 1 ? padJ : 0);
					const size_t cRows = dim - (i0 == i - 1 ? padI : 0);

					Value aColsOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(aCols));
					Value aRowsOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(aRows));
					Value bColsOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(bCols));
					Value bRowsOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(bRows));
					Value cColsOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(cCols));
					Value cRowsOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(cRows));

					Value aSpAddrOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(aSpAddr));
					Value bSpAddrOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(bSpAddr));
					Value outSpAddrOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(outSpAddr));

					Value garbageAddrOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(GARBAGE_ADDR));
					Value dimOp = rewriter.create<arith::ConstantOp>(
						loc, rewriter.getI64IntegerAttr(dim));

					rewriter.create<PreloadOp>(loc, garbageAddrOp, outSpAddrOp,
						dimOp, dimOp, cRowsOp, cColsOp);

					if (k0 == 0) { // First iteration
						rewriter.create<ComputePreloadedOp>(loc, aSpAddrOp,
							bSpAddrOp, aRowsOp, aColsOp, bRowsOp, bColsOp);

					} else { // All other iterations
						rewriter.create<ComputeAccumulatedOp>(loc, aSpAddrOp,
							bSpAddrOp, aRowsOp, aColsOp, bRowsOp, bColsOp);
					}
				}
			}
		}

		// Move-out C
		if (!cAddrNull) {
			const size_t sizeof_C = fullC ? sizeOfAccT : sizeOfElemT;

			for (size_t i0 = 0; i0 < i; i0++) {
				for (size_t j0 = 0; j0 < j; j0++) {
					const size_t offset = (i0 * strideC + j0) * dim * sizeof_C;
					const uint32_t cSpAddr = cSpAddrStart + (i0 * j + j0) * dim;

					const size_t cCols = dim - (j0 == j - 1 ? padJ : 0);
					const size_t cRows = dim - (i0 == i - 1 ? padI : 0);

					gemminiMvoutOffset(
						c, offset, cSpAddr, cCols, cRows, addrLen, rewriter);
				}
			}
		}
	}
	void tiledMatmulOuter(size_t dimI, size_t dimJ, size_t dimK, Value &A,
		Value &B, Value &D, Value &C, size_t strideA, size_t strideB,
		size_t strideD, size_t strideC, scale_t aScaleFactor,
		scale_t bScaleFactor, scale_acc_t dScaleFactor, size_t tileI,
		size_t tileJ, size_t tileK, int act, acc_scale_t scale,
		acc_scale_t bertScale, bool repeatingBias, bool aTranspose,
		bool bTranspose, bool fullC, bool lowD, uint8_t weightA, int dataflow,
		TileMatMulOp &tileMatMulOp, ConversionPatternRewriter &rewriter) const {
		const size_t dimIPadded = (dimI / dim + (dimI % dim != 0)) * dim;
		const size_t dimJPadded = (dimJ / dim + (dimJ % dim != 0)) * dim;
		const size_t dimKPadded = (dimK / dim + (dimK % dim != 0)) * dim;
		const size_t I0 =
			dimIPadded / (tileI * dim) + (dimIPadded % (tileI * dim) != 0);
		const size_t J0 =
			dimJPadded / (tileJ * dim) + (dimJPadded % (tileJ * dim) != 0);
		const size_t K0 =
			dimKPadded / (tileK * dim) + (dimKPadded % (tileK * dim) != 0);
		const size_t lastI = dimIPadded % (tileI * dim) == 0
			? tileI
			: (dimIPadded / dim) % tileI;
		const size_t lastJ = dimJPadded % (tileJ * dim) == 0
			? tileJ
			: (dimJPadded / dim) % tileJ;
		const size_t lastK = dimKPadded % (tileK * dim) == 0
			? tileK
			: (dimKPadded / dim) % tileK;
		const size_t paddingI = dimIPadded - dimI;
		const size_t paddingJ = dimJPadded - dimJ;
		const size_t paddingK = dimKPadded - dimK;
		// noBias = true when the D (bias) operand is an empty memref (shape
		// contains a 0). LowerBufferizedLinalgMatmulToTileMatmul synthesizes
		// a memref<0x0xi32> alloca for D when the source linalg.matmul has no
		// bias to thread through; without this check, spTiledMatmulOs would
		// MVIN 8x8xi32 of stack garbage from that alloca into the accumulator
		// and COMPUTE_PRELOADED would produce A·B + garbage instead of A·B.
		bool noBias = false;
		if (auto dMemRefType = llvm::dyn_cast<MemRefType>(
				tileMatMulOp.getDArray().getType())) {
			for (int64_t d : dMemRefType.getShape()) {
				if (d == 0) {
					noBias = true;
					break;
				}
			}
		}
		const size_t sizeofD = lowD ? sizeOfElemT : sizeOfAccT;
		const size_t sizeofC = fullC ? sizeOfAccT : sizeOfElemT;
		Location loc = tileMatMulOp.getLoc();
		llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);
		// cStride=1 spreads compute PE row i to accumulator row
		// (base_sp_addr + i). Without this, every PE row overwrites the same
		// accumulator slot — symptom: only row 0 of an NxM output ends up
		// populated post-MVOUT (libgemmini's compute() at gemmini.cc:642
		// computes `accumulator.at(base_sp_addr + c_stride * i).at(j)`, so
		// c_stride=0 collapses all i to row 0).
		// 2026-05-18 B-TRANSPOSE FIX: propagate the tile_matmul's
		// aTranspose/bTranspose attributes into CONFIG_EX bits 8/9.
		// Without this, the gemmini RoCC pipeline always decodes B in
		// K×N layout even when LowerTileToISA emits bTranspose=true.
		// (LowerBufferizedLinalgMatmulToTileMatmul inspects the
		// linalg.matmul indexing maps and sets bTranspose=true for
		// dronet's `(d1,d2)` rhs indexing, but the attribute was
		// dropped on the floor here.)
		rewriter.create<ConfigExOp>(loc, /*dataflow = */ dataflow,
			/*sysAct = */ act & 3,
			/*sysShift = */ 0, accScaleIdentity, /*cStride = */ 1,
			/*aStride = */ 1,
			/*aTranspose = */ aTranspose,
			/*bTranspose = */ bTranspose);
		Value strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideC * sizeofC));
		rewriter.create<ConfigStOp>(
			loc, strideValue, act & 3, llvm::APFloat(scale));
		// Outer tiled_matmul_outer config — match upstream gemmini.h:784-786:
		//   gemmini_extended3_config_ld(stride_A * sizeof(elem_t), ..., 0);
		//   gemmini_extended3_config_ld(stride_B * sizeof(elem_t), ..., 1);
		//   gemmini_extended3_config_ld(D_stride, ..., 2);
		// Outer sets all three id slots; inner sp_tiled_matmul_os reconfigures
		// slot 0 before each MVIN burst (all MVINs go through port 0 / id=0).
		strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideA * sizeOfElemT));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, llvm::APFloat(aScaleFactor), false, 0);
		strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideB * sizeOfElemT));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, llvm::APFloat(bScaleFactor), false, 1);
		strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideD * sizeofD));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, llvm::APFloat((float)dScaleFactor), lowD, 2);

		/*
		  Add config norm op
		*/
		if (act == IGELU) {
			const float sqrt_2 = 1.41421356237;
			const float S = bertScale;
			const float S_erf = (-0.2888 * ((S * S) / 2));

			const uint32_t qb = -1.769 / (S / sqrt_2);
			const uint32_t qc = 1.0 / S_erf;
			rewriter.create<ConfigNormOp>(loc, 0, 0, 0, 0, 0, qb, qc);
		}

		if (act == SOFTMAX) {
			const float a = 0.3585;
			const float b = 1.353;
			const float c = 0.344;

			const uint32_t qln2 = (int)(0.693147 / bertScale);
			const uint32_t qln2_inv = 65536 / qln2;
			const uint32_t qb = b / bertScale;
			const uint32_t qc = c / (a * bertScale * bertScale);
			rewriter.create<ConfigNormOp>(loc, qln2, 0, 0, 1, 0, qb, qc);
			rewriter.create<ConfigNormOp>(loc, qln2_inv, 1, 0, 1, 0, qb, qc);
		}

		for (size_t i0 = 0; i0 < I0; i0++)
			for (size_t j0 = 0; j0 < J0; j0++)
				for (size_t k0 = 0; k0 < K0; k0++) {
					Value pre;
					Location loc = A.getLoc();
					// 2026-05-18 ROOT-CAUSE FIX: match upstream
					// tiled_matmul_outer (gemmini.h:842-849) — pre is
					// NULL only on k0 != 0 (intermediate outer K
					// tiles); on k0 == 0 we always pass D + offset,
					// even when noBias is true.
					//
					// The earlier 2026-05-17 patch added "|| noBias"
					// to this condition. That made spTiledMatmulOs see
					// dAddrNull = true for the noBias case, which made
					// noBiasNewMatrix = (noBias && !dAddrNull) = false,
					// which left the cSpAddr OVERWRITE bit (bit 30)
					// SET. In libgemmini's compute() at gemmini.cc:631,
					//   bool acc_accum = (output_sp_addr >> 30) & 0x1;
					// bit 30 SET means ACCUMULATE: the PE drain ADDS
					// to the existing accumulator slot instead of
					// REPLACING it. Across outer-i0 iterations the
					// same cSpAddr is re-used, so each new tile's
					// A·B was being added on top of the previous
					// tile's result. Symptom verified on Spike with
					// matmul_196x32x32, all-ones inputs: rows 0-15
					// produced K, rows 16-31 produced 2K, ... up
					// to rows 192-195 producing 13K (one outer M-tile
					// per K added). Reverting to upstream semantics
					// (pre = D + offset on k0=0 regardless of noBias)
					// makes noBiasNewMatrix evaluate true on the last
					// inner k iteration, which clears bit 30 (OVERWRITE
					// mode) so the PE drain REPLACES the accumulator.
					//
					// The MVIN-D guard inside spTiledMatmulOs already
					// uses `!dAddrNull && !noBias`, so passing a non-
					// null D here does NOT cause spurious bias loads
					// when noBias is true.
					if (k0 != 0) {
						IntegerAttr preAttr = rewriter.getI64IntegerAttr(0);
						pre = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), preAttr);
					} else {
						size_t biasRow = repeatingBias ? 0 : i0 * tileI * dim;
						size_t offset =
							(biasRow * strideD + j0 * tileJ * dim) * sizeofD;
						IntegerAttr offsetAttr =
							rewriter.getI64IntegerAttr(offset);
						Value offsetValue = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), offsetAttr);
						pre = rewriter.create<arith::AddIOp>(
							loc, rewriter.getI64Type(), D, offsetValue);
					}

					Value out;
					if (k0 == K0 - 1) {
						size_t offset =
							(i0 * tileI * dim * strideC + j0 * tileJ * dim) *
							sizeofC;
						IntegerAttr offsetAttr =
							rewriter.getI64IntegerAttr(offset);
						Value offsetValue = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), offsetAttr);
						out = rewriter.create<arith::AddIOp>(
							loc, rewriter.getI64Type(), C, offsetValue);
					} else {
						IntegerAttr outAttr = rewriter.getI64IntegerAttr(0);
						out = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), outAttr);
					}
					const size_t i = i0 < I0 - 1 ? tileI : lastI;
					const size_t j = j0 < J0 - 1 ? tileJ : lastJ;
					const size_t k = k0 < K0 - 1 ? tileK : lastK;
					const size_t padI = i0 == I0 - 1 ? paddingI : 0;
					const size_t padJ = j0 == J0 - 1 ? paddingJ : 0;
					const size_t padK = k0 == K0 - 1 ? paddingK : 0;
					Value a;
					if (aTranspose) {
						size_t offset =
							(k0 * tileK * dim * strideA + i0 * tileI * dim) *
							sizeOfElemT;
						IntegerAttr offsetAttr =
							rewriter.getI64IntegerAttr(offset);
						Value offsetValue = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), offsetAttr);
						a = rewriter.create<arith::AddIOp>(
							loc, rewriter.getI64Type(), A, offsetValue);
					} else {
						size_t offset =
							(i0 * tileI * dim * strideA + k0 * tileK * dim) *
							sizeOfElemT;
						IntegerAttr offsetAttr =
							rewriter.getI64IntegerAttr(offset);
						Value offsetValue = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), offsetAttr);
						a = rewriter.create<arith::AddIOp>(
							loc, rewriter.getI64Type(), A, offsetValue);
					}
					Value b;
					if (bTranspose) {
						size_t offset =
							(j0 * tileJ * dim * strideB + k0 * tileK * dim) *
							sizeOfElemT;
						IntegerAttr offsetAttr =
							rewriter.getI64IntegerAttr(offset);
						Value offsetValue = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), offsetAttr);
						b = rewriter.create<arith::AddIOp>(
							loc, rewriter.getI64Type(), B, offsetValue);
					} else {
						size_t offset =
							(k0 * tileK * dim * strideB + j0 * tileJ * dim) *
							sizeOfElemT;
						IntegerAttr offsetAttr =
							rewriter.getI64IntegerAttr(offset);
						Value offsetValue = rewriter.create<arith::ConstantOp>(
							loc, rewriter.getI64Type(), offsetAttr);
						b = rewriter.create<arith::AddIOp>(
							loc, rewriter.getI64Type(), B, offsetValue);
					}
					if (dataflow == OUTPUT_STATIONARY) {
						spTiledMatmulOs(a, b, pre, out, aScaleFactor,
							bScaleFactor, dScaleFactor, i, j, k, padI, padJ,
							padK, strideA, strideB, strideD, strideC,
							aTranspose, bTranspose, fullC, lowD, noBias,
							repeatingBias, act, tileMatMulOp, rewriter);
					} else { // WS
						spTiledMatmulWs(a, b, pre, out, aScaleFactor,
							bScaleFactor, dScaleFactor, i, j, k, padI, padJ,
							padK, strideA, strideB, strideD, strideC,
							aTranspose, bTranspose, fullC, lowD, noBias,
							repeatingBias, act, tileMatMulOp, rewriter);
					}
				}
		IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
		Value flushValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64Type(), flushAttr);
		rewriter.replaceOpWithNewOp<FlushIntrOp>(
			tileMatMulOp, flushValue, flushValue);
		return;
	}

	// Phase 8 LOOP_WS lowering — host outer-loop variant.
	//
	// 2026-05-25 ROOT-CAUSE FIX (see tmp/loop_ws_debug/INVESTIGATION_LOG.md):
	// The previous single-shot variant passed the whole matmul dims to ONE
	// LOOP_WS RoCC opcode. libgemmini's loop_ws() (gemmini.cc:720) and the
	// FireSim RTL enforce a strict double-buffering contract:
	//   (I*K + K*J)*DIM <= BANK_NUM*bankRows/2 AND I*J <= ACC_ROWS/2
	// dronet's matmul_16x128x1152 (I=1, J=8, K=72) violates this with
	// SPAD footprint 10368 > 8192. The libgemmini handler aborts with
	// "LOOP_WS bounds were too large for double-buffering". The hardware
	// likewise cannot maintain double-buffering and produces wrong output.
	//
	// Canonical chipyard `tiled_matmul_outer` (gemmini.h:692) handles
	// this by wrapping `gemmini_loop_ws` in a host I0/J0/K0 triple loop
	// where each tile satisfies the contract; the caller picks
	// `tile_I/tile_J/tile_K` accordingly. We do the same here, reusing
	// the per-call-site greedy heuristic that already enforces SPAD/acc
	// budgets (see LegalizeForLLVMExport.cpp:~1894 — same heuristic
	// feeds the per-tile OS path).
	//
	// Per tile we emit the 6 LOOP_WS RoCC instructions (BOUNDS + ADDRS_AB
	// + ADDRS_DC + STRIDES_AB + STRIDES_DC + LOOP_WS opcode). Config
	// (CONFIG_EX, CONFIG_ST, CONFIG_LD×3, fence) is hoisted ONCE before
	// the loop because Gemmini retains those settings between LOOP_WS
	// invocations (matches gemmini.h:734-738).
	void tiledMatmulOuterLoopWs(size_t dimI, size_t dimJ, size_t dimK, Value &A,
		Value &B, Value &D, Value &C, size_t strideA, size_t strideB,
		size_t strideD, size_t strideC, scale_t aScaleFactor,
		scale_t bScaleFactor, scale_acc_t dScaleFactor,
		size_t tileI, size_t tileJ, size_t tileK,
		int act, acc_scale_t scale, bool repeatingBias, bool aTranspose,
		bool bTranspose, bool fullC, bool lowD, TileMatMulOp &tileMatMulOp,
		ConversionPatternRewriter &rewriter) const {
		const size_t dimIPadded = (dimI / dim + (dimI % dim != 0)) * dim;
		const size_t dimJPadded = (dimJ / dim + (dimJ % dim != 0)) * dim;
		const size_t dimKPadded = (dimK / dim + (dimK % dim != 0)) * dim;
		const size_t I0 =
			dimIPadded / (tileI * dim) + (dimIPadded % (tileI * dim) != 0);
		const size_t J0 =
			dimJPadded / (tileJ * dim) + (dimJPadded % (tileJ * dim) != 0);
		const size_t K0 =
			dimKPadded / (tileK * dim) + (dimKPadded % (tileK * dim) != 0);
		const size_t lastI = dimIPadded % (tileI * dim) == 0
			? tileI
			: (dimIPadded / dim) % tileI;
		const size_t lastJ = dimJPadded % (tileJ * dim) == 0
			? tileJ
			: (dimJPadded / dim) % tileJ;
		const size_t lastK = dimKPadded % (tileK * dim) == 0
			? tileK
			: (dimKPadded / dim) % tileK;
		const size_t paddingI = dimIPadded - dimI;
		const size_t paddingJ = dimJPadded - dimJ;
		const size_t paddingK = dimKPadded - dimK;

		// noBias = true when D's static shape contains a 0 (empty memref).
		bool noBias = false;
		if (auto dMemRefType = llvm::dyn_cast<MemRefType>(
				tileMatMulOp.getDArray().getType())) {
			for (int64_t d : dMemRefType.getShape()) {
				if (d == 0) {
					noBias = true;
					break;
				}
			}
		}

		const size_t sizeofD = lowD ? sizeOfElemT : sizeOfAccT;
		const size_t sizeofC = fullC ? sizeOfAccT : sizeOfElemT;
		Location loc = tileMatMulOp.getLoc();
		IntegerType i64Type = rewriter.getI64Type();
		llvm::APFloat accScaleIdentity((float)ACC_SCALE_IDENTITY);

		// COHERENCY FENCE (kept from prior fix): per-tile OS issues a
		// `fence rw,rw` BEFORE the first MVIN. The fence guarantees the
		// worker CPU's preceding stores (linalg.fill's zero-init of the
		// stack-allocated D bias buffer in LowerTileToISA) are visible
		// at the L2/DRAM coherency point before Gemmini's DMA reads.
		insertFence(loc, rewriter);

		// Hoisted config — matches canonical tiled_matmul_outer
		// (gemmini.h:734-738). Gemmini retains these between LOOP_WS
		// invocations, so they need to be issued once per dispatch.
		rewriter.create<ConfigExOp>(loc, /*dataflow=*/WEIGHT_STATIONARY,
			/*sysAct=*/act & 3, /*sysShift=*/0, accScaleIdentity,
			/*cStride=*/1, /*aStride=*/1,
			/*aTranspose=*/aTranspose, /*bTranspose=*/bTranspose);

		const size_t dStrideBytes =
			(repeatingBias ? 0 : strideD) * sizeofD;

		// CONFIG_LD/CONFIG_ST want strides in BYTES.
		Value cStrideBytesV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(strideC * sizeofC));
		rewriter.create<ConfigStOp>(
			loc, cStrideBytesV, act & 3, llvm::APFloat(scale));

		Value aStrideBytesV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(strideA * sizeOfElemT));
		rewriter.create<ConfigLdOp>(
			loc, aStrideBytesV, llvm::APFloat(aScaleFactor), false, 0);
		Value bStrideBytesV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(strideB * sizeOfElemT));
		rewriter.create<ConfigLdOp>(
			loc, bStrideBytesV, llvm::APFloat(bScaleFactor), false, 1);
		Value dStrideBytesV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(dStrideBytes));
		rewriter.create<ConfigLdOp>(
			loc, dStrideBytesV, llvm::APFloat((float)dScaleFactor), lowD, 2);

		// 2026-05-26 ROOT-CAUSE FIX: LOOP_WS_CONFIG_STRIDES_AB/DC takes
		// strides in ELEMENTS, not bytes. libgemmini (gemmini.cc:740,
		// 752, 769, 800-801, 815-820, 856-857) shows the inner FSM
		// formula `loop_ws_X + (i*loop_ws_X_stride + j)*DIM*sizeof_X`,
		// which means X_stride is in element units; the FSM multiplies
		// by sizeof itself. Canonical chipyard's sp_tiled_matmul_ws line
		// 684 passes A_row_stride / B_row_stride / D_row_stride /
		// C_row_stride directly (they came in as element counts from
		// tiled_matmul_outer line 822-826).
		// Previously we passed BYTE strides — same as CONFIG_LD/ST —
		// which silently worked for any matmul where the internal I or
		// K outer-index was 1 (the bogus stride term multiplied by 0),
		// but crashed for unaligned dronet matmuls (matmul_3136x32x27,
		// matmul_196x32x288) where I_per_tile > 1: the MVOUT formula
		// `C + (i * loop_ws_C_stride) * DIM * sizeof_C` then resolves to
		// 4× the intended row stride and writes past the C buffer into
		// the Zephyr worker stack → return-address corruption →
		// mcause=1 mepc=corrupt.
		Value aStrideElemsV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(strideA));
		Value bStrideElemsV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(strideB));
		Value cStrideElemsV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(strideC));
		Value dStrideElemsV = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(
				repeatingBias ? 0 : strideD));

		// Operand-reuse formula per canonical gemmini.h:783-792.
		// Only kicks in when the outer-tile count along one operand axis
		// is <= 2; alternation between spad-id 1 and 2 then keeps the
		// reused operand resident and skips redundant MVINs.
		const bool aReuse = (I0 * K0 <= 2);
		const bool bReuse = (J0 * K0 <= 2);

		for (size_t i0 = 0; i0 < I0; i0++) {
		for (size_t j0 = 0; j0 < J0; j0++) {
		for (size_t k0 = 0; k0 < K0; k0++) {
			// Per-tile bounds and padding.
			const size_t Ii = i0 < I0 - 1 ? tileI : lastI;
			const size_t Jj = j0 < J0 - 1 ? tileJ : lastJ;
			const size_t Kk = k0 < K0 - 1 ? tileK : lastK;
			const size_t padI = i0 == I0 - 1 ? paddingI : 0;
			const size_t padJ = j0 == J0 - 1 ? paddingJ : 0;
			const size_t padK = k0 == K0 - 1 ? paddingK : 0;

			// Per-tile A/B addresses (DRAM byte offsets).
			// Canonical operand-reuse via NULL address (gemmini.h:819-820):
			// when aReuse is active, pass A=NULL for any j0 >= 1 so the
			// hardware reuses the already-MVIN'd A copy in SPAD. Same
			// for bReuse with i0 >= 1.
			Value aAddr;
			if (aReuse && j0 >= 1) {
				aAddr = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(0));
			} else {
				size_t offset = aTranspose
					? (k0 * tileK * dim * strideA + i0 * tileI * dim) * sizeOfElemT
					: (i0 * tileI * dim * strideA + k0 * tileK * dim) * sizeOfElemT;
				Value offConst = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(offset));
				aAddr = rewriter.create<arith::AddIOp>(loc, i64Type, A, offConst);
			}
			Value bAddr;
			if (bReuse && i0 >= 1) {
				bAddr = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(0));
			} else {
				size_t offset = bTranspose
					? (j0 * tileJ * dim * strideB + k0 * tileK * dim) * sizeOfElemT
					: (k0 * tileK * dim * strideB + j0 * tileJ * dim) * sizeOfElemT;
				Value offConst = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(offset));
				bAddr = rewriter.create<arith::AddIOp>(loc, i64Type, B, offConst);
			}

			// D address: NULL on k0!=0 (intermediate K tile, no bias mvin)
			// and on noBias (no bias at all). Else D + per-tile offset.
			// ex_accumulate matches canonical: false only when
			// (noBias && k0==0) — the only case the accumulator must be
			// overwritten on first compute rather than accumulated into.
			Value dAddr;
			if (k0 != 0 || noBias) {
				dAddr = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(0));
			} else {
				size_t biasRow = repeatingBias ? 0 : i0 * tileI * dim;
				size_t offset = (biasRow * strideD + j0 * tileJ * dim) * sizeofD;
				Value offConst = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(offset));
				dAddr = rewriter.create<arith::AddIOp>(loc, i64Type, D, offConst);
			}
			const bool exAccumulate = !(noBias && k0 == 0);

			// C address: NULL on intermediate K iters (output stays in
			// accumulator), else C + per-tile offset to drain on k0==K0-1.
			Value cAddr;
			if (k0 == K0 - 1) {
				size_t offset =
					(i0 * tileI * dim * strideC + j0 * tileJ * dim) * sizeofC;
				Value offConst = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(offset));
				cAddr = rewriter.create<arith::AddIOp>(loc, i64Type, C, offConst);
			} else {
				cAddr = rewriter.create<arith::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(0));
			}

			// Spad-id alternation per canonical gemmini.h:789-792.
			int aSpadId = 0, bSpadId = 0;
			if (aReuse) aSpadId = ((i0 + k0) == 0) ? 1 : 2;
			if (bReuse) bSpadId = ((j0 + k0) == 0) ? 1 : 2;

			// Emit the 6 LOOP_WS RoCC instructions for this tile.
			uint64_t boundsRs1 =
				(uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
			uint64_t boundsRs2 =
				(uint64_t)Kk << 32 | (uint64_t)Jj << 16 | (uint64_t)Ii;

			// 2026-05-26 LOOP_WS RoCC emission trace.
			// Enabled when env var MERLIN_LOOPWS_TRACE=1 at compile time.
			// Dumps every per-tile (i0,j0,k0) parameter so we can diff
			// against canonical chipyard's `gemmini_loop_ws` output.
			if (const char *e = std::getenv("MERLIN_LOOPWS_TRACE");
				e && e[0] != '0') {
				llvm::errs() << "[loopws] matmul " << dimI << "x" << dimJ
					<< "x" << dimK << " (i0=" << i0 << " j0=" << j0
					<< " k0=" << k0 << ")"
					<< " I=" << Ii << " J=" << Jj << " K=" << Kk
					<< " padI=" << padI << " padJ=" << padJ
					<< " padK=" << padK
					<< " aSpadId=" << aSpadId << " bSpadId=" << bSpadId
					<< " exAcc=" << exAccumulate
					<< " noBias=" << noBias
					<< " aTr=" << aTranspose << " bTr=" << bTranspose
					<< " strideA=" << strideA << " strideB=" << strideB
					<< " strideC=" << strideC << " strideD=" << strideD
					<< " sizeOfElemT=" << sizeOfElemT
					<< " sizeofC=" << sizeofC << " sizeofD=" << sizeofD
					<< " dStrideBytes=" << dStrideBytes
					<< " boundsRs1=0x" << llvm::format_hex_no_prefix(boundsRs1, 16)
					<< " boundsRs2=0x" << llvm::format_hex_no_prefix(boundsRs2, 16)
					<< "\n";
			}
			Value boundsRs1V = rewriter.create<arith::ConstantOp>(
				loc, i64Type, rewriter.getI64IntegerAttr(boundsRs1));
			Value boundsRs2V = rewriter.create<arith::ConstantOp>(
				loc, i64Type, rewriter.getI64IntegerAttr(boundsRs2));
			rewriter.create<LoopWsConfigBoundsIntrOp>(loc, boundsRs1V, boundsRs2V);

			rewriter.create<LoopWsConfigAddrsABIntrOp>(loc, aAddr, bAddr);
			rewriter.create<LoopWsConfigAddrsDCIntrOp>(loc, dAddr, cAddr);

			rewriter.create<LoopWsConfigStridesABIntrOp>(
				loc, aStrideElemsV, bStrideElemsV);
			rewriter.create<LoopWsConfigStridesDCIntrOp>(
				loc, dStrideElemsV, cStrideElemsV);

			const int isResadd = 0;
			uint64_t rs1 = (uint64_t)aSpadId << 18 | (uint64_t)bSpadId << 16 |
				(uint64_t)(act & 0xFF) << 8 | ((lowD ? 1u : 0u) << 2) |
				((fullC ? 1u : 0u) << 1) | (exAccumulate ? 1u : 0u);
			uint64_t rs2 = (uint64_t)isResadd << 2 |
				((bTranspose ? 1u : 0u) << 1) | (aTranspose ? 1u : 0u);
			Value loopWsRs1 = rewriter.create<arith::ConstantOp>(
				loc, i64Type, rewriter.getI64IntegerAttr(rs1));
			Value loopWsRs2 = rewriter.create<arith::ConstantOp>(
				loc, i64Type, rewriter.getI64IntegerAttr(rs2));
			rewriter.create<LoopWsIntrOp>(loc, loopWsRs1, loopWsRs2);
		}
		}
		}
	}

	// Spad-source LOOP_WS — port of gemmini.h's gemmini_loop_ws_spad macro
	// (used by matmul_ws_mx_generic.c on mxGemmini). Emits an explicit
	// pre-MVIN sweep for A and B into scratchpad, then a 3-instruction
	// LOOP_WS preconfig that drives only compute+store inside the loop
	// FSM. This is the path mxGemmini's hardware actually supports for
	// `narrow_type=true` matmuls — the DRAM-source variant
	// (tiledMatmulOuterLoopWs) is not wired into LoopMatmul.scala's load
	// FSM request bundles for MX format.
	void tiledMatmulOuterLoopWsSpad(size_t dimI, size_t dimJ, size_t dimK,
		Value &A, Value &B, Value &D, Value &C, size_t strideA, size_t strideB,
		size_t strideD, size_t strideC, scale_t aScaleFactor,
		scale_t bScaleFactor, scale_acc_t dScaleFactor, int act,
		acc_scale_t scale, bool repeatingBias, bool aTranspose, bool bTranspose,
		bool fullC, bool lowD, TileMatMulOp &tileMatMulOp,
		ConversionPatternRewriter &rewriter) const {
		const size_t I = (dimI + dim - 1) / dim;
		const size_t J = (dimJ + dim - 1) / dim;
		const size_t K = (dimK + dim - 1) / dim;
		const size_t padI = I * dim - dimI;
		const size_t padJ = J * dim - dimJ;
		const size_t padK = K * dim - dimK;

		bool noBias = false;
		if (auto dMemRefType = llvm::dyn_cast<MemRefType>(
				tileMatMulOp.getDArray().getType())) {
			for (int64_t d : dMemRefType.getShape()) {
				if (d == 0) {
					noBias = true;
					break;
				}
			}
		}

		const size_t sizeofD = lowD ? sizeOfElemT : sizeOfAccT;
		(void)sizeofD;
		(void)fullC;
		Location loc = tileMatMulOp.getLoc();
		IntegerType i64Type = rewriter.getI64Type();
		// 2026-05-08: ACC_SCALE_IDENTITY in our headers is 1.0f, but
		// gemmini-mx's gemmini_params.h redefines it to 0 for the MX
		// path (compare `software/libgemmini/gemmini_params.h:1.0`
		// vs `software/gemmini-rocc-tests/include/gemmini_params.h:0`).
		// matmul_tiled_fp8.c (verified working on RadianceGemminiOnlyConfig
		// in 19s) emits CONFIG_EX rs1 = 0x1c004 with the high 32 bits =
		// 0x00000000 — i.e., `acc_scale = 0`. Setting acc_scale to 1.0f
		// (= 0x3F800000) instead is what's been hanging us, because
		// mxGemmini interprets non-zero acc_scale as a real per-block
		// scale mask and stalls waiting for matching SF data.
		llvm::APFloat accScaleIdentityMx(0.0f);

		// Spad layout (matches matmul_tiled_fp8.c:60-65, the working FP8
		// reference). C goes to a *scratchpad* address (not the
		// accumulator) — mxGemmini's BF16 output is written into spad
		// memory and read back from the SoC-mapped spad window at
		// 0x40000000 + spad_addr*2. The runner reads the BF16 result
		// directly without an explicit MVOUT.
		//   a_base = 0
		//   b_base_end = BANK_NUM * bankRows  (top of the spad)
		//   c_spad_addr = 128                 (matmul_tiled_fp8.c:71)
		const uint32_t aSpAddrStart = 0;
		const uint32_t bSpAddrEnd = BANK_NUM * bankRows;
		const uint32_t cSpadAddr = 128;

		// 1) CONFIG_EX — weight-stationary; mxFormat fields packed by
		//    GemminiConfigExLowering. acc_scale forced to 0 for MX
		//    (see comment above). 2026-05-25 also propagate
		//    aTranspose/bTranspose for parity with the per-tile path.
		rewriter.create<ConfigExOp>(loc, /*dataflow=*/WEIGHT_STATIONARY,
			/*sysAct=*/act & 3, /*sysShift=*/0, accScaleIdentityMx,
			/*cStride=*/1, /*aStride=*/1,
			/*aTranspose=*/aTranspose, /*bTranspose=*/bTranspose);

		// 2) CONFIG_ST — output stride. Stride in bytes between output
		// rows in DRAM. matmul_tiled_fp8.c uses bf16-packed-as-uint64_t
		// (4 bf16 per word), so stride = OUT_COLS * sizeof(uint64_t) =
		// (M/4) * 8 = M*2 = 32 for our M=16 case. We emit
		// strideC * 2 (= bf16 size) here regardless of the IREE-side
		// output element type because mxGemmini's WS-MX datapath only
		// writes bf16 into spad.
		Value strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideC * 2));
		// 2026-05-08: gemmini-mx's MVIN_SCALE_IDENTITY is 0 (not 1.0).
		// matmul_tiled_fp8.c calls gemmini_config_st(stride) which
		// expands to CONFIG_ST with rs2[63:32]=0 (acc_scale=0). Force
		// scale=0 here for the MX path. Same applies to CONFIG_LD scale.
		llvm::APFloat mvinScaleIdentityMx(0.0f);
		rewriter.create<ConfigStOp>(
			loc, strideValue, /*act=*/0, mvinScaleIdentityMx);

		// 3) CONFIG_LD A + pre-MVIN every A tile to a_base + (i*K + k)*dim.
		// MX path: force single-block MVIN (one tile per MVIN call) to
		// match the matmul_tiled_fp8.c reference layout. Multi-block
		// MVIN places adjacent K-tiles in interleaved spad rows, which
		// LOOP_WS_CONFIG_SPAD_AB doesn't expect; the rectangular K-tile
		// case (16x16x32 with I=J=1, K=2) produces non-uniform output
		// otherwise. The square 32x32x32 case happens to still work
		// because all tiles are identical, but the layout is wrong.
		const int aBlocks = 1;
		const int bBlocks = 1;

		// Opt-in IREE-binding trace. Pair with the runtime's
		// MERLIN_DISPATCH_DEBUG-built loader to read these stores back at
		// MERLIN_DEBUG_BINDING_TRACE_ADDR. Off in production.
		if (dispatchDebug) {
			auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
			auto emitTrace = [&](uint64_t dramAddr, Value v) {
				Value addrConst = rewriter.create<LLVM::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(dramAddr));
				Value ptr = rewriter.create<LLVM::IntToPtrOp>(
					loc, ptrTy, addrConst);
				rewriter.create<LLVM::StoreOp>(loc, v, ptr,
					/*alignment=*/0, /*isVolatile=*/true);
			};
			Value sentinel = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
				rewriter.getI64IntegerAttr(
					(int64_t)MERLIN_DEBUG_BINDING_TRACE_SENTINEL));
			emitTrace(MERLIN_DEBUG_BINDING_TRACE_ADDR + 0x00ULL, sentinel);
			emitTrace(MERLIN_DEBUG_BINDING_TRACE_ADDR + 0x08ULL, A);
			emitTrace(MERLIN_DEBUG_BINDING_TRACE_ADDR + 0x10ULL, B);
			emitTrace(MERLIN_DEBUG_BINDING_TRACE_ADDR + 0x18ULL, C);
		}

		strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideA * sizeOfElemT));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, mvinScaleIdentityMx, false, 0);
		for (size_t i0 = 0; i0 < I; i0++) {
			for (size_t k0 = 0; k0 < K; k0 += aBlocks) {
				const size_t offset = (i0 * strideA + k0) * dim * sizeOfElemT;
				const uint32_t aSpAddr = aSpAddrStart + (i0 * K + k0) * dim;
				const size_t blocks = k0 + aBlocks <= K ? aBlocks : K - k0;
				const size_t cols =
					blocks * dim - (k0 + blocks >= K ? padK : 0);
				const size_t rows = dim - (i0 == I - 1 ? padI : 0);
				gemminiMvinOffset(
					A, offset, aSpAddr, cols, rows, addrLen, rewriter);
			}
		}

		// 4) CONFIG_LD B + pre-MVIN every B tile to bSpAddrEnd from the top
		//    down (b_addr_start = bSpAddrEnd - K*J*dim per
		//    LoopMatmul.scala:388). Note: matmul_tiled_fp8.c omits a B
		//    CONFIG_LD because A and B happen to share the same stride
		//    (MATMUL_M*sizeof(elem_t)) — we still emit it for safety.
		strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(strideB * sizeOfElemT));
		rewriter.create<ConfigLdOp>(
			loc, strideValue, mvinScaleIdentityMx, false, 1);
		const uint32_t bSpAddrStart = bSpAddrEnd - K * J * dim;
		for (size_t k0 = 0; k0 < K; k0++) {
			for (size_t j0 = 0; j0 < J; j0 += bBlocks) {
				const size_t offset = (k0 * strideB + j0) * dim * sizeOfElemT;
				const uint32_t bSpAddr = bSpAddrStart + (k0 * J + j0) * dim;
				const size_t blocks = j0 + bBlocks <= J ? bBlocks : J - j0;
				const size_t cols =
					blocks * dim - (j0 + blocks >= J ? padJ : 0);
				const size_t rows = dim - (k0 == K - 1 ? padK : 0);
				gemminiMvinOffset(
					B, offset, bSpAddr, cols, rows, addrLen, rewriter);
			}
		}

		// 5) LOOP_WS_CONFIG_BOUNDS — same packing as DRAM-source variant.
		uint64_t boundsRs1 =
			(uint64_t)padK << 32 | (uint64_t)padJ << 16 | (uint64_t)padI;
		uint64_t boundsRs2 =
			(uint64_t)K << 32 | (uint64_t)J << 16 | (uint64_t)I;
		Value boundsRs1Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(boundsRs1));
		Value boundsRs2Value = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(boundsRs2));
		rewriter.create<LoopWsConfigBoundsIntrOp>(
			loc, boundsRs1Value, boundsRs2Value);

		// 6) LOOP_WS_CONFIG_SPAD_AB — A_spad_addr / B_spad_end_addr.
		Value aSpAddrValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(aSpAddrStart));
		Value bSpEndValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(bSpAddrEnd));
		rewriter.create<LoopWsConfigSpadABIntrOp>(
			loc, aSpAddrValue, bSpEndValue);

		// 7) LOOP_WS — packed control words per gemmini_loop_ws_spad.
		// 2026-05-08: matmul_tiled_fp8.c calls this with full_C=false,
		// low_D=false, ex_accumulate=false, NO_ACTIVATION, no transpose,
		// skips=0x38 (ldA|ldB|ldD), spad_only=1, C in spad. Override the
		// caller-supplied fullC/lowD/exAccumulate for the MX path because
		// mxGemmini's compute writes BF16 into spad, not i32 into the
		// accumulator.
		// rs1: (a_spad_id<<18) | (b_spad_id<<16) | (act<<8) |
		//      (low_D<<2) | (full_C<<1) | ex_accumulate
		// rs2: (C_spad_addr<<32) | 0x200 (spad_only) |
		//      (skip_mask=0x38) |
		//      (is_resadd<<2) | (B_transpose<<1) | A_transpose
		const int aSpadId = 0;
		const int bSpadId = 0;
		const int isResadd = 0;
		(void)noBias;
		(void)fullC;
		(void)lowD;
		const bool mxFullC = false;
		const bool mxLowD = false;
		const bool mxExAccumulate = false;
		const int mxAct = 0;            // NO_ACTIVATION
		const uint64_t skipMask = 0x38ULL;          // skip ldA, ldB, ldD
		const uint64_t spadOnly = 0x200ULL;         // bit 9
		uint64_t rs1 = (uint64_t)aSpadId << 18 | (uint64_t)bSpadId << 16 |
			(uint64_t)(mxAct & 0xFF) << 8 | ((mxLowD ? 1u : 0u) << 2) |
			((mxFullC ? 1u : 0u) << 1) | (mxExAccumulate ? 1u : 0u);
		uint64_t rs2 = ((uint64_t)cSpadAddr << 32) | spadOnly | skipMask |
			((uint64_t)isResadd << 2) |
			((bTranspose ? 1u : 0u) << 1) | (aTranspose ? 1u : 0u);
		Value loopWsRs1 = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(rs1));
		Value loopWsRs2 = rewriter.create<arith::ConstantOp>(
			loc, i64Type, rewriter.getI64IntegerAttr(rs2));
		rewriter.create<LoopWsIntrOp>(loc, loopWsRs1, loopWsRs2);

		// 8) Pre-FLUSH + per-cell bf16-read + bf16→i32 integer-only widen
		// + i32 store, iterating directly over SMEM. mxGemmini's WS-MX
		// LOOP_WS with spad_only=1 writes bf16 results into the
		// memory-mapped scratchpad window at SMEM_BASE (0x40000000) +
		// cSpadAddr*16 bytes/row. After the FLUSH busy-wait we read
		// each bf16 from SMEM, decode its integer value via shifts
		// (no FP — bare-metal dispatch ELF may have mstatus.FS=0), and
		// write to bvC. Iterates high-index → low so wider i32 stores
		// don't clobber SMEM (which is the source, not bvC).
		// Decode formula for normal bf16 (sign, exp ∈ [127, 158], mant):
		//   m = mant | 0x80                  ; implicit leading 1
		//   shift = 134 - exp                ; 7-bit mantissa + bias
		//   if shift >= 0: v = m >> shift
		//   else:           v = m << -shift
		//   if exp == 0:    v = 0            ; zero/denormal
		//   if sign:        v = -v
		// Selects (not branches) so no block-split inside the pattern.
		{
			Value flushZero = rewriter.create<arith::ConstantOp>(loc,
				i64Type, rewriter.getI64IntegerAttr(0));
			rewriter.create<FlushIntrOp>(loc, flushZero, flushZero);

			auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
			auto i16Ty = rewriter.getIntegerType(16);
			auto i32Ty = rewriter.getI32Type();

			const uint64_t smemBaseAddr =
				0x40000000ULL + (uint64_t)cSpadAddr * 16ULL;
			const size_t numCells = dimI * dimJ;

			Value cZeroI32 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(0));
			Value c0x7F = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(0x7F));
			Value c0x80 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(0x80));
			Value c0xFF = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(0xFF));
			Value c7 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(7));
			Value c15 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(15));
			Value c134 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
				rewriter.getI32IntegerAttr(134));

			// Unroll: emit straight-line code per cell. For 16x16=256 cells
			// this is ~13 LLVM ops × 256 = ~3300 ops. Bounded and avoids
			// block splits inside the conversion pattern. Iterate high→low
			// so wide i32 stores don't overwrite bf16 source bytes (the
			// source lives in SMEM here, separate from bvC, so this is
			// safety paranoia not correctness).
			for (int64_t cell = (int64_t)numCells - 1; cell >= 0; --cell) {
				Value srcConst = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
					rewriter.getI64IntegerAttr(
						(int64_t)(smemBaseAddr + (uint64_t)cell * 2ULL)));
				Value srcPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, srcConst);
				Value bf16Val = rewriter.create<LLVM::LoadOp>(loc, i16Ty, srcPtr,
					/*alignment=*/0, /*isVolatile=*/true);
				Value ext = rewriter.create<LLVM::ZExtOp>(loc, i32Ty, bf16Val);

				// sign = ext >> 15
				Value sign = rewriter.create<LLVM::LShrOp>(loc, ext, c15);
				// exp = (ext >> 7) & 0xFF
				Value expShift = rewriter.create<LLVM::LShrOp>(loc, ext, c7);
				Value exp = rewriter.create<LLVM::AndOp>(loc, expShift, c0xFF);
				// mant = (ext & 0x7F) | 0x80
				Value mantOnly = rewriter.create<LLVM::AndOp>(loc, ext, c0x7F);
				Value m = rewriter.create<LLVM::OrOp>(loc, mantOnly, c0x80);
				// shift = 134 - exp ; right-shift count
				Value rshift = rewriter.create<LLVM::SubOp>(loc, c134, exp);
				// lshift = exp - 134 ; left-shift count
				Value lshift = rewriter.create<LLVM::SubOp>(loc, exp, c134);
				// rRes = m >> rshift  (saturated to 0 for huge rshift OK on RV64I)
				Value rRes = rewriter.create<LLVM::LShrOp>(loc, m, rshift);
				// lRes = m << lshift
				Value lRes = rewriter.create<LLVM::ShlOp>(loc, m, lshift);
				// shiftIsPos = exp > 134
				Value shiftIsPos = rewriter.create<LLVM::ICmpOp>(loc,
					LLVM::ICmpPredicate::ugt, exp, c134);
				Value v0 = rewriter.create<LLVM::SelectOp>(loc, shiftIsPos, lRes, rRes);
				// expIsZero = exp == 0
				Value expIsZero = rewriter.create<LLVM::ICmpOp>(loc,
					LLVM::ICmpPredicate::eq, exp, cZeroI32);
				Value v1 = rewriter.create<LLVM::SelectOp>(loc, expIsZero, cZeroI32, v0);
				// neg = -v1 ; final = sign ? neg : v1
				Value zero = cZeroI32;
				Value neg = rewriter.create<LLVM::SubOp>(loc, zero, v1);
				Value signIsOne = rewriter.create<LLVM::ICmpOp>(loc,
					LLVM::ICmpPredicate::ne, sign, cZeroI32);
				Value finalVal = rewriter.create<LLVM::SelectOp>(loc, signIsOne, neg, v1);

				// Store to bvC + cell*4
				Value offI32 = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
					rewriter.getI64IntegerAttr((int64_t)cell * 4));
				Value dstAddr = rewriter.create<LLVM::AddOp>(loc, C, offI32);
				Value dstPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, dstAddr);
				rewriter.create<LLVM::StoreOp>(loc, finalVal, dstPtr,
					/*alignment=*/0, /*isVolatile=*/true);
			}
		}
		(void)padJ;
		(void)padI;
	}

	size_t tiledMatmulTotalSpadRows(size_t I, size_t J, size_t K) const {
		return (I * K + K * J) * dim;
	}

	size_t tiledMatmulTotalAccRows(size_t I, size_t J) const {
		return (I * J) * dim;
	}

  public:
	using ConvertOpToLLVMPattern<TileMatMulOp>::ConvertOpToLLVMPattern;
	explicit GemminiTileMatMulLowering(LLVMTypeConverter &typeConverter,
		int64_t dim, int64_t addrLen, int64_t accRows, int64_t bankRows,
		size_t sizeOfElemT, size_t sizeOfAccT, bool clampSingleBlockMvin = false,
		bool useLoopWs = false, int64_t mxFormat = -1,
		bool dispatchDebug = false)
		: ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen),
		  accRows(accRows), bankRows(bankRows), sizeOfElemT(sizeOfElemT),
		  sizeOfAccT(sizeOfAccT),
		  clampSingleBlockMvin(clampSingleBlockMvin),
		  useLoopWs(useLoopWs), mxFormat(mxFormat),
		  dispatchDebug(dispatchDebug) {}
	LogicalResult matchAndRewrite(TileMatMulOp tileMatMulOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		size_t dbPartitionRows = ((BANK_NUM * bankRows / 2) / 2);
		size_t dbMatsInPartition = (dbPartitionRows / dim);
		size_t dbMatsInAcc((accRows / 2) / dim);
		size_t dbMaxTileIJ((size_t)sqrt(dbMatsInAcc));
		size_t dbMaxTileK(dbMatsInPartition / dbMaxTileIJ);

		Value aArray = tileMatMulOp.getAArray();
		Value bArray = tileMatMulOp.getBArray();
		Value cArray = tileMatMulOp.getCArray();
		Value dArray = tileMatMulOp.getDArray();
		MemRefType aArrayType = dyn_cast<MemRefType>(aArray.getType());
		MemRefType bArrayType = dyn_cast<MemRefType>(bArray.getType());
		MemRefType cArrayType = dyn_cast<MemRefType>(cArray.getType());
		MemRefType dArrayType = dyn_cast<MemRefType>(dArray.getType());
		StridedLayoutAttr aArrayLayout =
			dyn_cast<StridedLayoutAttr>(aArrayType.getLayout());
		StridedLayoutAttr bArrayLayout =
			dyn_cast<StridedLayoutAttr>(bArrayType.getLayout());
		StridedLayoutAttr cArrayLayout =
			dyn_cast<StridedLayoutAttr>(cArrayType.getLayout());
		SmallVector<Type> resultType = {rewriter.getIndexType()};
		TypeRange typeRange(resultType);
		Location loc = tileMatMulOp.getLoc();
		IntegerType i64Type = rewriter.getI64Type();
		// FIX (2026-05-19) — gemmini-subspan-offset-dropped: when the memref
		// operand comes from `hal.interface.binding.subspan` with a non-zero
		// `byte_offset`, IREE's LLVMCPU lowering parks the byte_offset in the
		// memref descriptor's offset slot via `fromStaticShape` and stores the
		// raw binding base pointer (no offset) in `aligned_ptr`. Since the
		// Gemmini path reads `aligned_ptr` via ExtractAlignedPointerAsIndex,
		// it must re-apply the byte_offset by walking the def-use chain back
		// to the subspan op. Without this, `mvin` reads bytes from rodata + 0
		// instead of rodata + byte_offset for any `ReadOnly` (non-`Indirect`)
		// binding — silently producing wrong matmul output for every
		// non-first weight tensor (dronet's steer head, etc.).
		auto applySubspanOffset = [&](Value extracted, Value origMemref) {
			auto subspanOff = walkBackToSubspanByteOffset(origMemref);
			if (!subspanOff) return extracted;
			// byte_offset is an index-typed SSA value (bytes). The existing
			// static-layout addition below uses element-multiples, so we add
			// the dynamic byte offset directly in bytes here.
			return static_cast<Value>(
				rewriter.create<arith::AddIOp>(loc, extracted, *subspanOff));
		};
		Value aArrayExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, typeRange, aArray);
		aArrayExtractOp = applySubspanOffset(aArrayExtractOp, aArray);
		if (aArrayLayout) {
			Value offset = rewriter.create<arith::ConstantIndexOp>(
				loc, aArrayLayout.getOffset() * sizeOfElemT);
			aArrayExtractOp =
				rewriter.create<arith::AddIOp>(loc, aArrayExtractOp, offset);
		}
		Value aArrayindexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, aArrayExtractOp);
		Value bArrayExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, typeRange, bArray);
		bArrayExtractOp = applySubspanOffset(bArrayExtractOp, bArray);
		if (bArrayLayout) {
			Value offset = rewriter.create<arith::ConstantIndexOp>(
				loc, bArrayLayout.getOffset() * sizeOfElemT);
			bArrayExtractOp =
				rewriter.create<arith::AddIOp>(loc, bArrayExtractOp, offset);
		}
		Value bArrayindexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, bArrayExtractOp);
		Value cArrayExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, typeRange, cArray);
		cArrayExtractOp = applySubspanOffset(cArrayExtractOp, cArray);
		if (cArrayLayout) {
			Value offset = rewriter.create<arith::ConstantIndexOp>(
				loc, cArrayLayout.getOffset() * sizeOfElemT);
			cArrayExtractOp =
				rewriter.create<arith::AddIOp>(loc, cArrayExtractOp, offset);
		}
		Value cArrayindexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, cArrayExtractOp);
		Value dArrayExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, typeRange, dArray);
		// D is typically a stack alloca (emptyD), not a binding. The walkback
		// will gracefully return nullopt and applySubspanOffset is a no-op.
		dArrayExtractOp = applySubspanOffset(dArrayExtractOp, dArray);
		Value dArrayindexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, dArrayExtractOp);
		// 2026-05-21 Bug A fix attempt: explicitly mask bit 63 of every mvin/mvout
		// rs1 derivative. The chipyard Gemmini RTL's DMA address register does NOT
		// mask bit 63 (LoadController.scala:102 + DMA.scala:251 — paddrBits=56 so
		// bits 56-63 are undefined when set). Setting bit 63 in rs1 causes DMA to
		// silently misread for offset!=0 cases. The toggle is added by upstream
		// MemRefToLLVM/ArithToLLVM (PtrToIntOp + signed IndexCastOp) — we mask it
		// here, at the point all addresses converge to i64, so subsequent ops
		// preserve our cleared bit-63.
		{
			Value mask = rewriter.create<arith::ConstantOp>(
				loc, i64Type, rewriter.getI64IntegerAttr(0x7FFFFFFFFFFFFFFFLL));
			aArrayindexCastOp = rewriter.create<arith::AndIOp>(loc, aArrayindexCastOp, mask);
			bArrayindexCastOp = rewriter.create<arith::AndIOp>(loc, bArrayindexCastOp, mask);
			cArrayindexCastOp = rewriter.create<arith::AndIOp>(loc, cArrayindexCastOp, mask);
			dArrayindexCastOp = rewriter.create<arith::AndIOp>(loc, dArrayindexCastOp, mask);
		}
		// Opt-in Bug A matmul-operand trace. Emits volatile stores of the
		// computed MLIR-level i64 A/B/C/D addresses (post-applySubspanOffset,
		// pre-bit-63-toggle — the eventual mvin rs1 will have bit 63 set on
		// top of these) to MERLIN_DEBUG_MATMUL_TRACE_ADDR. The runtime's
		// [mtrace] probe reads them back. Off in production.
		if (dispatchDebug) {
			auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
			auto emitMatmulTrace = [&](uint64_t dramAddr, Value v) {
				Value addrConst = rewriter.create<LLVM::ConstantOp>(
					loc, i64Type, rewriter.getI64IntegerAttr(dramAddr));
				Value ptr = rewriter.create<LLVM::IntToPtrOp>(
					loc, ptrTy, addrConst);
				rewriter.create<LLVM::StoreOp>(loc, v, ptr,
					/*alignment=*/0, /*isVolatile=*/true);
			};
			Value sentinel = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
				rewriter.getI64IntegerAttr(
					(int64_t)MERLIN_DEBUG_MATMUL_TRACE_SENTINEL));
			emitMatmulTrace(MERLIN_DEBUG_MATMUL_TRACE_ADDR + 0x00ULL, sentinel);
			emitMatmulTrace(MERLIN_DEBUG_MATMUL_TRACE_ADDR + 0x08ULL,
				aArrayindexCastOp);
			emitMatmulTrace(MERLIN_DEBUG_MATMUL_TRACE_ADDR + 0x10ULL,
				bArrayindexCastOp);
			emitMatmulTrace(MERLIN_DEBUG_MATMUL_TRACE_ADDR + 0x18ULL,
				cArrayindexCastOp);
			emitMatmulTrace(MERLIN_DEBUG_MATMUL_TRACE_ADDR + 0x20ULL,
				dArrayindexCastOp);
		}
		llvm::ArrayRef<int64_t> aArrayShape = aArrayType.getShape();
		llvm::ArrayRef<int64_t> bArrayShape = bArrayType.getShape();
		llvm::ArrayRef<int64_t> cArrayShape = cArrayType.getShape();
		llvm::ArrayRef<int64_t> dArrayShape = dArrayType.getShape();
		size_t dimI = aArrayShape[0];
		size_t dimK = aArrayShape[1];
		size_t dimJ = bArrayShape[1];
		size_t strideA = aArrayShape[1];
		size_t strideB = bArrayShape[1];
		size_t strideC = cArrayShape[1];
		size_t strideD = dArrayShape[1];
		scale_t aScaleFactor = tileMatMulOp.getAScaleFactor().convertToFloat();
		scale_t bScaleFactor = tileMatMulOp.getBScaleFactor().convertToFloat();
		scale_acc_t dScaleFactor =
			tileMatMulOp.getDScaleFactor().convertToFloat();
		int act = tileMatMulOp.getAct();
		acc_scale_t scale = tileMatMulOp.getAccScale().convertToFloat();
		acc_scale_t bertScale = tileMatMulOp.getBertScale().convertToFloat();
		bool repeatingBias = tileMatMulOp.getRepeatingBias();
		bool aTranspose = tileMatMulOp.getATranspose();
		bool bTranspose = tileMatMulOp.getBTranspose();
		bool fullC = tileMatMulOp.getFullC();
		bool lowD = tileMatMulOp.getLowD();
		uint8_t weightA = tileMatMulOp.getWeightA();
		size_t dimIPaded = (dimI / dim + (dimI % dim != 0)) * dim;
		size_t dimJPaded = (dimJ / dim + (dimJ % dim != 0)) * dim;
		size_t dimKPaded = (dimK / dim + (dimK % dim != 0)) * dim;
		size_t maxSpadRows = BANK_NUM * bankRows / 2;
		size_t maxAccRows = accRows / 2;
		size_t tileI, tileJ, tileK;
		if (act == LAYERNORM || act == SOFTMAX) {
			tileI = 1;
			tileJ = dimJPaded / dim;
			tileK = 1;
		} else {
			tileI =
				dimIPaded / dim < dbMaxTileIJ ? dimIPaded / dim : dbMaxTileIJ;
			tileJ =
				dimJPaded / dim < dbMaxTileIJ ? dimJPaded / dim : dbMaxTileIJ;
			tileK = dimKPaded / dim < dbMaxTileK ? dimKPaded / dim : dbMaxTileK;
		}
		// 2026-05-25 Phase 3a — shape-aware tile growth.
		// Original ordering (J → I → K) caps tall-skinny shapes early because
		// the first dim grown consumes SPAD/acc budget. For dronet's
		// matmul_3136x32x27 we want I to grow first (I_max=196 vs J_max=2),
		// not J. For 16x128x576 we want K (K_max=36 vs J_max=8).
		// Strategy: each iteration, compute how many DIM units each axis
		// still has room to grow toward its dimension's max, pick the axis
		// with the largest remaining count and try to grow it. If the
		// constraint fails for the largest, fall through to the next.
		// Only one axis grows per iteration (vs the original's up-to-three)
		// so SPAD budget is allocated to the largest-need dim first.
		const size_t maxI = dimIPaded / dim;
		const size_t maxJ = dimJPaded / dim;
		const size_t maxK = dimKPaded / dim;
		auto canGrowI = [&]() {
			return tileI < maxI &&
				tiledMatmulTotalSpadRows(tileI + 1, tileJ, tileK) <= maxSpadRows &&
				tiledMatmulTotalAccRows(tileI + 1, tileJ) <= maxAccRows;
		};
		auto canGrowJ = [&]() {
			return tileJ < maxJ &&
				tiledMatmulTotalSpadRows(tileI, tileJ + 1, tileK) <= maxSpadRows &&
				tiledMatmulTotalAccRows(tileI, tileJ + 1) <= maxAccRows;
		};
		auto canGrowK = [&]() {
			return tileK < maxK &&
				tiledMatmulTotalSpadRows(tileI, tileJ, tileK + 1) <= maxSpadRows;
		};
		while (true) {
			// Remaining DIM-units that each axis could still grow toward its max.
			const size_t remI = maxI > tileI ? maxI - tileI : 0;
			const size_t remJ = maxJ > tileJ ? maxJ - tileJ : 0;
			const size_t remK = maxK > tileK ? maxK - tileK : 0;
			if (remI == 0 && remJ == 0 && remK == 0) break;
			// Pick the axis with the largest remaining count and try it
			// first. Ties broken by the static priority I > J > K (matches the
			// canonical gemmini-rocc-tests preference: amortize MVIN-B across
			// the outer-M loop).
			bool grew = false;
			std::array<std::pair<size_t, int>, 3> order = {{
				{remI, 0}, {remJ, 1}, {remK, 2}
			}};
			std::sort(order.begin(), order.end(),
				[](const std::pair<size_t, int>& a,
					const std::pair<size_t, int>& b) {
					return a.first > b.first;
				});
			for (auto& p : order) {
				if (p.first == 0) continue;
				if (p.second == 0 && canGrowI()) {
					tileI++; grew = true; break;
				}
				if (p.second == 1 && canGrowJ()) {
					tileJ++; grew = true; break;
				}
				if (p.second == 2 && canGrowK()) {
					tileK++; grew = true; break;
				}
			}
			if (!grew) break;
		}
		// 2026-05-24: tileI=1 hardcode REMOVED for experimentation.
		// Original comment (kept for context): "spTiledMatmulOs at tileI>1
		// hangs on FireSimGemminiAndOPUShuttleConfig for any matmul with
		// M > dim". Since May 16 we've landed: subspan-offset fix
		// (project_gemmini_subspan_offset_RESOLVED), multi-N-tile MVIN-D
		// OOB fix (project_gemmini_multi_n_tile_d_oob_fix), Zephyr
		// worker-stack bump to 4 MiB. Worth retrying tileI>1 to amortize
		// MVIN-B across the outer-M loop (canonical gemmini-rocc-tests
		// pattern). Falls back to tileI=1 if this re-introduces a hang.
		// tileI = 1;  // disabled

		// Shape-keyed tile override (opt-in, off by default). Reads
		// MERLIN_GEMMINI_TILE_OVERRIDE at compile time so tile triples
		// can be swept without rebuilding iree-compile.
		// Format: "M,N,K:I,J,K;M,N,K:I,J,K;..."
		// Bounds-checked against SPAD/acc budget; unmatched shapes use
		// the auto-grow result.
		if (const char *ovr = std::getenv("MERLIN_GEMMINI_TILE_OVERRIDE")) {
			std::string spec(ovr);
			size_t pos = 0;
			while (pos < spec.size()) {
				size_t entryEnd = spec.find(';', pos);
				if (entryEnd == std::string::npos) entryEnd = spec.size();
				std::string entry = spec.substr(pos, entryEnd - pos);
				pos = entryEnd + 1;
				size_t colon = entry.find(':');
				if (colon == std::string::npos) continue;
				size_t shM, shN, shK, ovI, ovJ, ovK;
				if (sscanf(entry.c_str(), "%zu,%zu,%zu:%zu,%zu,%zu",
						&shM, &shN, &shK, &ovI, &ovJ, &ovK) == 6) {
					// Match raw matmul dims (pre-padding). Use Kpadded
					// for K matching since IR sees padded.
					if (dimI == shM && dimJ == shN &&
						(dimK == shK || dimKPaded == shK)) {
						const size_t ovItry = std::min(ovI, maxI);
						const size_t ovJtry = std::min(ovJ, maxJ);
						const size_t ovKtry = std::min(ovK, maxK);
						if (tiledMatmulTotalSpadRows(
								ovItry, ovJtry, ovKtry) <= maxSpadRows &&
							tiledMatmulTotalAccRows(ovItry, ovJtry) <=
								maxAccRows) {
							tileI = ovItry;
							tileJ = ovJtry;
							tileK = ovKtry;
						}
					}
				}
			}
		}
		int dataflow = tileMatMulOp.getDataflow();

		// 2026-05-26 LOOP_WS gate (Opt#A — drop K-alignment requirement).
		// Pad propagation in tiledMatmulOuterLoopWs (paddingI/J/K computed
		// from dim*Padded - dim* at lines ~1271, then threaded into the
		// per-iteration CONFIG_BOUNDS rs1 field at lines ~1372-1374) is
		// already correct. libgemmini's loop_ws() (gemmini.cc:696-823)
		// applies `rows = DIM - (k == K-1 ? pad_K : 0)` on every MVIN-A/B
		// and COMPUTE — unaligned K is a first-class case. The canonical
		// chipyard tiled_matmul_ws_dronet_3136x32x27 test (K=27, padK=5)
		// runs in 28,800 cycles on the same FireSim Shuttle bitstream, so
		// the RTL path is healthy. The earlier "aligned-only" gate was a
		// conservative protection against the byte-vs-element stride bug
		// (fixed in v9); with Opt#0's i8-MVOUT fusion landed, LOOP_WS's
		// smaller MVOUT bandwidth wins on K-unaligned shapes that the
		// per-tile OS path used to handle (yolov8n's matmul_25600x16x27 at
		// 167M cycles was the headliner). fitsHalfSpad remains the only
		// structural gate (it tracks libgemmini's internal double-buffer
		// budget).
		// Opt#A2 retry — per-tile spad gate. tiledMatmulOuterLoopWs already
		// tiles I/J/K, so each LOOP_WS only consumes (tileI*tileK +
		// tileK*tileJ)*dim spad rows; old full-matrix gate forced
		// large-M shapes onto the slow OS path even when the inner
		// LOOP_WS budget fits fine.
		const size_t spadHalf = (BANK_NUM * bankRows) / 2;
		const size_t loopWsSpadRowsPerTile =
			(tileI * tileK + tileK * tileJ) * (size_t)dim;
		const bool fitsHalfSpad = loopWsSpadRowsPerTile <= spadHalf;
		// Opt#H probe (2026-05-27): TESTED, REVERTED. Gating small-K (≤48)
		// matmuls onto OS path moved yolov8n by ≤0.5% across all dispatches
		// (matmul_6400x32x32 LOOP_WS 114,595,966 vs OS 114,495,811 cycles —
		// 0.09% delta). The 8–17 cycles/MAC inefficiency for small-K is
		// NOT due to LOOP_WS vs OS path choice — it's intrinsic to the
		// large-M / small-K Gemmini matmul codegen (both paths). Real fix
		// requires per-LOOP_WS cycle profiling (MVIN/COMPUTE/MVOUT
		// breakdown) to identify whether DRAM bandwidth, command-queue
		// serialization, or SPAD-bank conflicts dominate.
		const bool useLoopWsActual = useLoopWs && fitsHalfSpad;

		if (useLoopWsActual) {
			// Phase 8: emit a single LOOP_WS sequence for the whole matmul
			// (~12 commands total: 5 configs + 6 LOOP_WS_* + flush) instead
			// of the per-tile MVIN/PRELOAD/COMPUTE/MVOUT expansion (~56
			// commands). Required to clear the GemminiTile.scala:446
			// backpressure assertion under MMIO command issue.
			//
			// 2026-05-17: prior session attempted to "fix dronet" by
			// switching this branch to force `dataflow=WS` and route
			// through the per-tile path — that broke mlp_wide
			// matmul_16x16x32 on FireSim (Zephyr misaligned-load fault
			// after [dc] o=2). Root cause of dronet hang turned out to
			// be a Zephyr 96 KiB stack overflow, NOT a matmul-path
			// issue. The bulk-LOOP_WS path here is correct and is what
			// mlp_wide × Gemmini × FireSim was validated against on
			// 2026-05-14 (hash 0xbb3076f6865e2266).
			//
			// Path B (2026-05-08): when MX format is enabled, route to
			// the spad-source LOOP_WS variant that pre-MVINs A/B and
			// uses LOOP_WS_CONFIG_SPAD_AB (funct=24) — that's the path
			// matmul_ws_mx_generic.c uses on mxGemmini and the only one
			// that propagates narrow_type correctly through the load FSM.
			if (mxFormat >= 0) {
				tiledMatmulOuterLoopWsSpad(dimI, dimJ, dimK, aArrayindexCastOp,
					bArrayindexCastOp, dArrayindexCastOp, cArrayindexCastOp,
					strideA, strideB, strideD, strideC, aScaleFactor,
					bScaleFactor, dScaleFactor, act, scale, repeatingBias,
					aTranspose, bTranspose, fullC, lowD, tileMatMulOp,
					rewriter);
			} else {
				tiledMatmulOuterLoopWs(dimI, dimJ, dimK, aArrayindexCastOp,
					bArrayindexCastOp, dArrayindexCastOp, cArrayindexCastOp,
					strideA, strideB, strideD, strideC, aScaleFactor,
					bScaleFactor, dScaleFactor,
					tileI, tileJ, tileK,
					act, scale, repeatingBias,
					aTranspose, bTranspose, fullC, lowD, tileMatMulOp,
					rewriter);
			}

			IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
			Value flushValue = rewriter.create<arith::ConstantOp>(
				loc, rewriter.getI64Type(), flushAttr);
			rewriter.replaceOpWithNewOp<FlushIntrOp>(
				tileMatMulOp, flushValue, flushValue);
			insertFence(loc, rewriter);
			return success();
		}

		tiledMatmulOuter(dimI, dimJ, dimK, aArrayindexCastOp, bArrayindexCastOp,
			dArrayindexCastOp, cArrayindexCastOp, strideA, strideB, strideD,
			strideC, aScaleFactor, bScaleFactor, dScaleFactor, tileI, tileJ,
			tileK, act, scale, bertScale, repeatingBias, aTranspose, bTranspose,
			fullC, lowD, weightA, dataflow, tileMatMulOp, rewriter);

		insertFence(loc, rewriter);

		return success();
	};

  private:
	int64_t dim;
	int64_t addrLen;
	int64_t accRows;
	int64_t bankRows;
	size_t sizeOfElemT;
	size_t sizeOfAccT;
	// When true, force a single-block MVIN per j-tile (cols == dim) instead
	// of the default `blocks * dim` packed-row MVIN. The default lowering
	// emits `blocks = min(j, MAX_BYTES/(dim*1)) = 4` for j=4 / dim=16,
	// producing MVINs with cols=64. The mxGemmini-MMIO hardware
	// (RadianceGemminiOnlyConfig with MxFloat-typed datapath) only allocates
	// a 6-bit `num_cols` field in the MvinRs2 bundle (see
	// `chipyard/sims/vcs/.../LoadController.sv:220` — the generated SV
	// reads `cols = rs2[37:32]` regardless of what `mvin_cols_bits` Chisel
	// computes for that config). cols=64 truncates to 0, the load
	// controller's `bytes_to_read = cols * rows * elem_bits / 8` becomes 0,
	// and `LoadController.scala:191`'s "A single mvin instruction must
	// load more than 0 bytes" assertion fires deep into the dispatch.
	// Forcing blocks=1 makes every MVIN's cols stay at `dim` (=16, fits
	// in 6 bits) and matches the hand-coded reference kernel
	// `gemmini-rocc-tests/bareMetalC/matmul_ws_mx_generic.c:162` which
	// also uses `gemmini_extended_mvin(..., DIM, DIM)` per tile.
	bool clampSingleBlockMvin;

	// Phase 8: when true, replace the per-tile MVIN/PRELOAD/COMPUTE/MVOUT
	// expansion with a single LOOP_WS sequence. The hardware loops over
	// the tiles internally, dramatically reducing the host-side command
	// count (from ~56 to ~11 per 16x64x64 matmul) so the dispatch ELF
	// doesn't overrun gemmini's command queue under MMIO issue.
	// Reference kernel that emits the same sequence: `gemmini.h:390-398`
	// (tiled_matmul_loop_ws) and `gemmini-rocc-tests/.../tiled_matmul_ws.c`.
	bool useLoopWs;
	// MX format selector: -1 (Disabled), 0 (FP4), 1 (FP6_0), 2 (FP8_0),
	// 3 (FP6_1), 4 (FP8_1). When != -1 and useLoopWs is true, the
	// spad-source LOOP_WS path is used (gemmini_loop_ws_spad with
	// pre-MVINs, funct=24 LOOP_WS_CONFIG_SPAD_AB, skip mask 0x38, and
	// rs2[9]=spad_only=1) — that matches what the upstream
	// matmul_ws_mx_generic.c gold reference does on mxGemmini. The
	// vanilla DRAM-source LOOP_WS path (funct=10/11/12/13) is not used
	// for MX matmuls because it isn't wired for narrow_type=true in
	// LoopMatmul.scala (the load FSM request bundles don't carry
	// narrow_type), causing the loop_matmul_unroller_busy signal to
	// hang forever after issue.
	int64_t mxFormat;

	// Opt-in: when true, emit volatile stores of binding pointers + matmul
	// operand pointers (A/B/C/D) to fixed DRAM trace regions defined in
	// merlin_debug_addresses.h. Pairs with the runtime's
	// MERLIN_DISPATCH_DEBUG build. Default false; off in production.
	bool dispatchDebug;
};

class GemminiTileConvLowering : public ConvertOpToLLVMPattern<TileConvOp> {

	void gemminiLoopConvWs(int batchSize, int inDim, int inChannels,
		int outChannels, int outDim, int poolOutDim, int stride, int padding,
		int kernelDim, int kernelDilation, int poolSize, int poolStride,
		int poolPadding, int batches, int porows, int pocols, int pochs,
		int krows, int kcols, int kchs, int lpad, int rpad, int upad, int dpad,
		int plpad, int prpad, int pupad, int pdpad, int orows, int ocols,
		Value &weights, Value &output, Value &bias, Value &input, bool noBias,
		bool noPool, bool downsample, bool writ180, bool inputDilated, int act,
		bool transOutput1203, bool transWeight1203, bool transWeight0132,
		bool transInput3120, int maxPixelsPerRow, int inStride,
		int weightStride, int outStride, bool dw, TileConvOp &tileConvOp,
		ConversionPatternRewriter &rewriter) const {
		Location loc = tileConvOp.getLoc();
		// loopConvWsConfig1
		uint64_t rs1 = (uint64_t)outChannels << 48 |
			(uint64_t)inChannels << 32 | (uint64_t)inDim << 16 |
			(uint64_t)batchSize;
		uint64_t rs2 = (uint64_t)padding << 56 | (uint64_t)stride << 48 |
			(uint64_t)outDim << 32 | (uint64_t)poolOutDim << 16 |
			(uint64_t)outDim;
		TypedAttr rs1Attr = rewriter.getI64IntegerAttr(rs1);
		TypedAttr rs2Attr = rewriter.getI64IntegerAttr(rs2);
		Value rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
		Value rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
		rewriter.create<LoopConvWsConfig1IntrOp>(loc, rs1Value, rs2Value);
		// loopConvWsConfig2
		rs1 = (uint64_t)kernelDim << 48 | (uint64_t)poolOutDim << 32 |
			(uint64_t)poolSize << 16 | (uint64_t)poolStride << 8 |
			(uint64_t)poolPadding;
		rs2 = (uint64_t)batches << 48 | (uint64_t)porows << 32 |
			(uint64_t)pocols << 16 | (uint64_t)pochs;
		rs1Attr = rewriter.getI64IntegerAttr(rs1);
		rs2Attr = rewriter.getI64IntegerAttr(rs2);
		rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
		rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
		rewriter.create<LoopConvWsConfig2IntrOp>(loc, rs1Value, rs2Value);
		// loopConvWsConfig3
		rs1 = (uint64_t)krows << 48 | (uint64_t)kcols << 32 |
			(uint64_t)kchs << 16 | (uint64_t)lpad;
		rs2 = (uint64_t)rpad << 48 | (uint64_t)upad << 32 |
			(uint64_t)dpad << 24 | (uint64_t)plpad << 16 | (uint64_t)inDim;
		rs1Attr = rewriter.getI64IntegerAttr(rs1);
		rs2Attr = rewriter.getI64IntegerAttr(rs2);
		rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
		rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
		rewriter.create<LoopConvWsConfig3IntrOp>(loc, rs1Value, rs2Value);
		// loopConvWsconfig4
		rs1 = (uint64_t)orows << 48 | (uint64_t)prpad << 32 |
			(uint64_t)pupad << 21 | (uint64_t)pdpad << 10 |
			(uint64_t)kernelDilation;
		rs2 = (uint64_t)inStride << 48 | (uint64_t)weightStride << 32 |
			(uint64_t)outStride << 16 | (uint64_t)ocols;
		rs1Attr = rewriter.getI64IntegerAttr(rs1);
		rs2Attr = rewriter.getI64IntegerAttr(rs2);
		rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
		rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
		rewriter.create<LoopConvWsConfig4IntrOp>(loc, rs1Value, rs2Value);
		// loopConvWsconfig5
		rewriter.create<LoopConvWsConfig5IntrOp>(loc, weights, output);
		// loopConvWsconfig6
		rewriter.create<LoopConvWsConfig6IntrOp>(loc, bias, input);
		// loopConvWs
		const int aSpadId = 0;
		const int bSpadId = 0;
		rs1 = (uint64_t)aSpadId << 18 | (uint64_t)bSpadId << 16 |
			(uint64_t)maxPixelsPerRow << 8 | dw << 6 | transInput3120 << 5 |
			transWeight0132 << 4 | transWeight1203 << 3 | transOutput1203 << 2 |
			writ180 << 1 | noBias;
		rs2 = act << 3 | inputDilated << 2 | downsample << 1 | noPool;
		rs1Attr = rewriter.getI64IntegerAttr(rs1);
		rs2Attr = rewriter.getI64IntegerAttr(rs2);
		rs1Value = rewriter.create<arith::ConstantOp>(loc, rs1Attr);
		rs2Value = rewriter.create<arith::ConstantOp>(loc, rs2Attr);
		rewriter.create<LoopConvWsIntrOp>(loc, rs1Value, rs2Value);
	}

	void spTiledConv(int batchSize, int inRowDim, int inColDim, int inChannels,
		int outChannels, int outRowDim, int outColDim, int poolOutRowDim,
		int poolOutColDim, int stride, int padding, int kernelDim,
		int kernelDilation, int inStride, int weightStride, int outStride,
		int poolSize, int poolStride, int poolPadding, int batches, int porows,
		int pocols, int pochs, int krows, int kcols, int kchs, int lpad,
		int rpad, int upad, int dpad, int plpad, int prpad, int pupad,
		int pdpad, Value &input, Value &weights, Value &output, Value &bias,
		int act, acc_scale_t scale, bool wrot180, bool transOutput1203,
		bool transInput3120, bool transWeight1203, bool transWeight0132,
		bool noBias, bool noPool, bool downsample, bool inputDilated, bool dw,
		TileConvOp &tileConvOp, ConversionPatternRewriter &rewriter) const {

		Location loc = tileConvOp.getLoc();
		if (dw) {
			kchs = 1;
			pochs = 1;
		}

		const int orows = porows * poolStride + poolSize - 1 - pupad - pdpad;
		const int ocols = pocols * poolStride + poolSize - 1 - plpad - prpad;
		const int ochs = pochs;

		// Calculate image dimensions
		// Note: "irows" and "icols" includes padding
		const int dilatedKrows = krows + (kernelDilation - 1) * (krows - 1);
		const int dilatedKcols = kcols + (kernelDilation - 1) * (kcols - 1);
		int irows = orows * stride + dilatedKrows - 1;
		int icols = ocols * stride + dilatedKcols - 1;
		int irowsUnpadded = irows - upad - dpad;
		int icolsUnpadded = icols - lpad - rpad;

		const int ichs = kchs;

#define UNDILATED(x) ((inputDilated) ? (((x) + 1) / 2) : (x))

		if (inputDilated) {
			irowsUnpadded = (irowsUnpadded + 1) / 2;
			icolsUnpadded = (icolsUnpadded + 1) / 2;

			irows = irowsUnpadded + UNDILATED(upad) + UNDILATED(dpad);
			icols = icolsUnpadded + UNDILATED(lpad) + UNDILATED(rpad);
		}

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
		const bool transposed = transOutput1203 || transInput3120 ||
			transWeight1203 || transWeight0132;
		int maxPixelsPerRow = transposed || wrot180 || downsample ||
				inputDilated || kernelDilation > 1 || ichs > dim
			? 1
			: dim / ichs;
		if (maxPixelsPerRow > kcols)
			maxPixelsPerRow = kcols;
#else
		const int maxPixelsPerRow = 1;
#endif
		// Calculate spad address offsets
		const int outChannelsPerBank = ochs / dim + (ochs % dim != 0);
		const int inChannelsPerBank = kchs / dim + (kchs % dim != 0);
		const int bRows = transWeight0132
			? inChannelsPerBank * kcols * krows * ochs
			: outChannelsPerBank * kcols * krows * kchs;

		static uint32_t dSpAddrRow = 0;
		static uint32_t cSpAddrRow = 0;

		const uint32_t aSpAddrStart = 0;
		const uint32_t bSpAddrStart = BANK_NUM * bankRows - bRows;
		const uint32_t dSpAddrStart = (1 << (addrLen - 1)) + dSpAddrRow;
		const uint32_t cSpAddrStart = (3 << (addrLen - 2)) + cSpAddrRow;

		if (bias != 0) {
			dSpAddrRow = (dSpAddrRow + accRows / 2) % accRows;
		}

		if (output != 0) {
			cSpAddrRow = (cSpAddrRow + accRows / 2) % accRows;
		}
		if (inRowDim == inColDim && outRowDim == outColDim &&
			poolOutRowDim == poolOutColDim) {
			gemminiLoopConvWs(batchSize, inRowDim, inChannels, outChannels,
				outRowDim, poolOutRowDim, stride, padding, kernelDim,
				kernelDilation, poolSize, poolStride, poolPadding, batches,
				porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad,
				dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output,
				bias, input, noBias, noPool, downsample, wrot180, inputDilated,
				act, transOutput1203, transWeight1203, transWeight0132,
				transInput3120, maxPixelsPerRow, inStride, weightStride,
				outStride, dw, tileConvOp, rewriter);
			return;
		}
		if (!noPool) {
			llvm::outs()
				<< "Pooling with rectangular convolutions is currently not "
				   "supported.\n";
			return;
		}
		// Only rectangular convolutions will use the following C code
		// mvin bias
		const size_t maxBlockLen = MAX_BYTES / (dim * 1);
		const size_t maxBlockLenAcc = MAX_BYTES / (dim * 4);
		if (bias != NULL) {
			// TODO(2026-05-27, Opt#E dormant): the bias MVIN loop here mirrors
			// libgemmini's structure but may have redundant inner iterations
			// for the common single-channel-block case. Audit if native
			// tile_conv is revived (task #90).
			const int maxOchsPerMvin = ochs < (int)(maxBlockLenAcc * dim)
				? ochs
				: maxBlockLenAcc * dim;
			Value zeroValue = rewriter.create<arith::ConstantOp>(
				loc, rewriter.getI64IntegerAttr(0));
			rewriter.create<ConfigLdOp>(loc, zeroValue,
				llvm::APFloat((float)MVIN_SCALE_IDENTITY), false, 2,
				batches * orows * ocols);
			for (int b = 0; b < batches; b++)
				for (int orow = 0; orow < orows; orow++)
					for (int ocol = 0; ocol < ocols; ocol += dim) {
						const int I = ocols - ocol > dim ? dim : ocols - ocol;
						for (int och = 0; och < ochs; och += maxOchsPerMvin) {
							const int J = ochs - och > maxOchsPerMvin
								? maxOchsPerMvin
								: ochs - och;
							const uint32_t dSpAddr = dSpAddrStart +
								(och / dim) * batches * orows * ocols +
								b * orows * ocols + orow * ocols + ocol;
							if (noBias) {
								gemminiMvinOffset<Mvin3IntrOp>(zeroValue,
									0 * sizeOfAccT, dSpAddr, J, I, addrLen,
									rewriter);
							} else {
								gemminiMvinOffset<Mvin3IntrOp>(bias,
									och * sizeOfAccT, dSpAddr, J, I, addrLen,
									rewriter);
							}
						}
					}
		}
		// mvin input
		if (input != NULL) {
			int maxChsPerMvin =
				ichs < (int)(maxBlockLen * dim) ? ichs : maxBlockLen * dim;
			if (transInput3120) {
				maxChsPerMvin = batches < (int)(maxBlockLen * dim)
					? batches
					: maxBlockLen * dim;
			}
			const int dramStride = transInput3120 ? batchSize * sizeOfElemT
												  : inChannels * sizeOfElemT;
			const int spadStride = transInput3120
				? ichs * (irows >> downsample) * (icols >> downsample)
				: batches * (irows >> downsample) * (icols >> downsample);
			Value strideValue = rewriter.create<arith::ConstantOp>(
				loc, rewriter.getI64IntegerAttr(dramStride << downsample));
			rewriter.create<ConfigLdOp>(loc, strideValue,
				llvm::APFloat((float)MVIN_SCALE_IDENTITY), false, 0, spadStride,
				maxPixelsPerRow);
			const int b_it = transInput3120 ? maxChsPerMvin : 1;
			const int ich_it = transInput3120 ? 1 : maxChsPerMvin;
			for (int b = 0; b < batches; b += b_it)
				for (int irow = -UNDILATED(upad);
					 irow < irowsUnpadded + UNDILATED(dpad);
					 irow += 1 + downsample) {
					const int irowPadded = irow + UNDILATED(upad);
					for (int icol = -UNDILATED(lpad);
						 icol < icolsUnpadded + UNDILATED(rpad);) {
						// TODO(2026-05-27, Opt#E dormant): edge-of-image MVINs
						// may overlap when (upad + lpad + dilation) push the
						// input column outside the unpadded range. Profile in
						// the tile_conv revival pass (task #90).
						int I = icolsUnpadded - icol > (dim << downsample)
							? (dim << downsample)
							: icolsUnpadded - icol;
						if (icol < 0) {
							I = -icol > dim ? dim : -icol;
						} else if (icol >= icolsUnpadded) {
							I = icolsUnpadded + UNDILATED(rpad) - icol > dim
								? dim
								: icolsUnpadded + UNDILATED(rpad) - icol;
						}
						const int icolPadded = icol + UNDILATED(lpad);
						for (int ich = 0; ich < ichs; ich += ich_it) {
							int K = ichs - ich > maxChsPerMvin ? maxChsPerMvin
															   : ichs - ich;
							if (transInput3120) {
								K = batches - b > maxChsPerMvin ? maxChsPerMvin
																: batches - b;
							}
#define DS(x) ((x) >> (downsample))
							uint32_t aSpAddr = aSpAddrStart +
								(ich / dim) * batches * DS(irows) * DS(icols) +
								b * DS(irows) * DS(icols) +
								DS(irowPadded) * DS(icols) + DS(icolPadded);
							if (transInput3120) {
								aSpAddr = aSpAddrStart +
									(b / dim) * ichs * DS(irows) * DS(icols) +
									ich * DS(irows) * DS(icols) +
									DS(irowPadded) * DS(icols) + DS(icolPadded);
							}
							const bool is_zeros = irow < 0 ||
								irow >= irowsUnpadded || icol < 0 ||
								icol >= icolsUnpadded;
							size_t offset = (b * inRowDim * inColDim +
												irow * inColDim + icol) *
									inStride +
								ich;
							Value memAddr = input;
							if (is_zeros) {
								memAddr = rewriter.create<arith::ConstantOp>(
									loc, rewriter.getI64IntegerAttr(0));
								offset = 0;
							} else if (transInput3120) {
								offset = (ich * inRowDim * inColDim +
											 irow * inColDim + icol) *
										batchSize +
									b;
							}
							gemminiMvinOffset(memAddr, offset * sizeOfElemT,
								aSpAddr, K, I >> downsample, addrLen, rewriter);
						}
						icol += I;
					}
				}
		}
		// mvin weights
		if (weights != NULL) {
			int max_chs_per_mvin =
				ochs < (int)(maxBlockLen * dim) ? ochs : maxBlockLen * dim;
			if (transWeight0132) {
				max_chs_per_mvin =
					kchs < (int)(maxBlockLen * dim) ? kchs : maxBlockLen * dim;
			}
			size_t dramStride = weightStride * sizeOfElemT;
			if (dw) {
				dramStride = sizeOfElemT;
			} else if (transWeight1203) {
				dramStride = kernelDim * kernelDim * outChannels * sizeOfElemT;
			} else if (transWeight0132) {
				dramStride = inChannels * sizeOfElemT;
			}
			const size_t spadBlockStride =
				transWeight0132 ? krows * kcols * ochs : krows * kcols * kchs;
			Value dramStrideValue = rewriter.create<arith::ConstantOp>(
				loc, rewriter.getI64IntegerAttr(dramStride));
			rewriter.create<ConfigLdOp>(loc, dramStrideValue,
				llvm::APFloat((float)MVIN_SCALE_IDENTITY), false, 1,
				spadBlockStride);

			const size_t och_it = transWeight0132 ? dim : max_chs_per_mvin;
			const size_t kch_it = transWeight0132 ? max_chs_per_mvin : dim;
			for (int och = 0; och < ochs; och += och_it) {
				for (int krow = 0; krow < krows; krow++)
					for (int kcol = 0; kcol < kcols; kcol++)
						for (int kch = 0; kch < kchs; kch += kch_it) {
							int K = kchs - kch > dim ? dim : kchs - kch;
							int J = ochs - och > max_chs_per_mvin
								? max_chs_per_mvin
								: ochs - och;
							if (transWeight0132) {
								K = ochs - och > dim ? dim : ochs - och;
								J = kchs - kch > max_chs_per_mvin
									? max_chs_per_mvin
									: kchs - kch;
							}
							uint32_t bSpAddr = bSpAddrStart +
								(och / dim) * krows * kcols * kchs +
								krow * kcols * kchs + kcol * kchs + kch;
							if (transWeight0132) {
								bSpAddr = bSpAddrStart +
									(kch / dim) * krows * kcols * ochs +
									krow * kcols * ochs + kcol * ochs + och;
							}
							size_t offset = (krow * kernelDim * inChannels +
												kcol * inChannels + kch) *
									weightStride +
								och;
							if (dw) {
								offset = krow * kernelDim + kcol;
							} else if (transWeight1203) {
								offset = (kch * kernelDim * kernelDim +
											 krow * kernelDim + kcol) *
										outChannels +
									och;
							} else if (transWeight0132) {
								offset = (krow * kernelDim * outChannels +
											 kcol * outChannels + och) *
										inChannels +
									kch;
							}
							gemminiMvinOffset<Mvin2IntrOp>(weights,
								offset * sizeOfElemT, bSpAddr, J, K, addrLen,
								rewriter);
						}
			}
		}
		// Compute
		{
			const int b_it = transInput3120 ? dim : 1;
			const int ocol_it = transInput3120 ? 1 : (dim << inputDilated);
			if (transInput3120) {
				rewriter.create<ConfigExOp>(loc,
					/*dataflow = */ OUTPUT_STATIONARY,
					/*act = */ 0, /*shift = */ 0,
					/*scale = */ llvm::APFloat((float)0),
					/*cStride = */ orows * ocols,
					/*aStride = */ irows * icols,
					/*aTranspose = */ 0, /*bTranspose*/ 0,
					/*setOnlyStrides = */ true);
			}
			for (int och = 0; och < ochs; och += dim) {
				for (int krow = 0; krow < krows; krow++) {
					for (int kcol = 0; kcol < kcols; kcol += maxPixelsPerRow) {
						for (int kch = 0; kch < kchs; kch += dim) {
							bool newWeights = true;
							for (int b = 0; b < batches; b += b_it) {
								for (int orow = 0; orow < orows; orow++) {
									// Skip some kernel rows due to
									// input-dilation
									if (inputDilated &&
										((krow * kernelDilation +
											 orow * stride - upad) %
												2 !=
											0)) {
										continue;
									}
									for (int ocol = 0; ocol < ocols;) {
										// Skip some cols dimensions due to
										// input-dilation
										if (inputDilated &&
											((kcol + ocol * stride - lpad) %
													2 !=
												0)) {
											ocol++;
											continue;
										}
										int irow = orow * stride +
											krow * kernelDilation;
										int icol = ocol * stride +
											kcol * kernelDilation;
										if (inputDilated) {
											irow = (irow + 1) / 2;
											icol = (icol + 1) / 2;
										}
										const int pixels =
											kcols - kcol > maxPixelsPerRow
											? maxPixelsPerRow
											: kcols - kcol;
										const uint32_t cSpAddr = cSpAddrStart +
											(och / dim) * batches * orows *
												ocols +
											b * orows * ocols + orow * ocols +
											ocol;
										// Over here, construct a new matrix
										//
										// Let us assume that we only ever
										// operate on one pixel in one row.
										// Thus, krows == kcols == 1
										//
										// Then, for every set of I, J, and K
										// values
										//     - I = ocols
										//     - J = ochs
										//     - K = kchs
										int I = UNDILATED(
											ocols - ocol > (dim << inputDilated)
												? (dim << inputDilated)
												: ocols - ocol);
										const int J =
											ochs - och > dim ? dim : ochs - och;
										const int K = pixels *
											(kchs - kch > dim ? dim
															  : kchs - kch);
										if (transInput3120) {
											I = batches - b > dim ? dim
																  : batches - b;
										}
										uint32_t aSpAddr = aSpAddrStart +
											(kch / dim) * batches * DS(irows) *
												DS(icols) +
											b * DS(irows) * DS(icols) +
											DS(irow) * DS(icols) + DS(icol);
										if (transInput3120) {
											aSpAddr = aSpAddrStart +
												(b / dim) * kchs * DS(irows) *
													DS(icols) +
												kch * DS(irows) * DS(icols) +
												DS(irow) * DS(icols) + DS(icol);
										}
										const int krow_ =
											wrot180 ? krows - krow - 1 : krow;
										const int kcol_ =
											wrot180 ? kcols - kcol - 1 : kcol;
										uint32_t bSpAddr = bSpAddrStart +
											(och / dim) * krows * kcols * kchs +
											krow_ * kcols * kchs +
											kcol_ * kchs + kch;
										if (transWeight0132) {
											bSpAddr = bSpAddrStart +
												(kch / dim) * krows * kcols *
													ochs +
												krow_ * kcols * ochs +
												kcol_ * ochs + och;
										}
										const uint32_t perSpAddr =
											newWeights ? bSpAddr : GARBAGE_ADDR;

										Value garbageAddrOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(
													GARBAGE_ADDR));
										Value iOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(I));
										Value jOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(J));
										Value kOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(K));
										Value perSpAddrOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(
													perSpAddr));
										Value aSpAddrOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(
													aSpAddr));
										Value cSpAddrOp =
											rewriter.create<arith::ConstantOp>(
												loc,
												rewriter.getI64IntegerAttr(
													cSpAddr));

										rewriter.create<PreloadOp>(loc,
											perSpAddrOp, cSpAddrOp, kOp, jOp,
											iOp, jOp);
										if (newWeights) {
											rewriter.create<ComputePreloadedOp>(
												loc, aSpAddrOp, garbageAddrOp,
												iOp, kOp, iOp, jOp);
										} else {
											rewriter
												.create<ComputeAccumulatedOp>(
													loc, aSpAddrOp,
													garbageAddrOp, iOp, kOp,
													iOp, jOp);
										}
										ocol += ocol_it;
										newWeights = false;
									}
								}
							}
						}
					}
				}
			}
		}
#undef DS
#undef UNDILATED
		// mvout output
		if (output != NULL) {
			if (noPool) {
				for (int b = 0; b < batches; b++)
					for (int orow = 0; orow < orows; orow++)
						for (int ocol = 0; ocol < ocols; ocol += dim) {
							const int I =
								ocols - ocol > dim ? dim : ocols - ocol;
							for (int och = 0; och < ochs; och += dim) {
								const int J =
									ochs - och > dim ? dim : ochs - och;
								const uint32_t cSpAddr = cSpAddrStart +
									(och / dim) * batches * orows * ocols +
									b * orows * ocols + orow * ocols + ocol;
								size_t outOffset =
									(b * outRowDim * outColDim +
										orow * outColDim + ocol) *
										outStride +
									och;
								if (transOutput1203) {
									outOffset = (orow * outColDim * batchSize +
													ocol * batchSize + b) *
											outChannels +
										och;
								}
								gemminiMvoutOffset(output,
									outOffset * sizeOfElemT, cSpAddr, J, I,
									addrLen, rewriter);
							}
						}
			} else {
				printf("Pooling with rectangular convolutions is currently not "
					   "supported.\n");
				exit(1);
			}
		}
	}

	void tiledConv(int batchSize, int inRowDim, int inColDim, int inChannels,
		int outChannels, int outRowDim, int outColDim, int stride,
		int inputDilation, int kernelDilation, int padding, int kernelDim,
		int inStride, int weightStride, int outStride, bool wrot180,
		bool transOutput1203, bool transInput3120, bool transWeight1203,
		bool transWeight0132, int batches, int porows, int pocols, int pochs,
		int krows, int kcols, int kchs, const Value &input,
		const Value &weights, const Value &bias, Value &output, int act,
		acc_scale_t scale, int poolSize, int poolStride, int poolPadding,
		TileConvOp &tileConvOp, ConversionPatternRewriter &rewriter) const {
		bool noBias = false;
		bool noPool = poolStride == 0;
		if (noPool) {
			poolSize = 1;
			poolStride = 1;
			poolPadding = 0;
		}
		const bool downsample = stride == 2 && kernelDim == 1 &&
			inRowDim % 2 == 0 && inColDim % 2 == 0 && padding == 0 && noPool &&
			inputDilation == 1 && !transInput3120;
		const int inputDilated = inputDilation == 2;
		int64_t stDramStride = transOutput1203
			? batchSize * outChannels * sizeOfElemT
			: outChannels * sizeOfElemT;
		Location loc = tileConvOp.getLoc();
		Value strideValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64IntegerAttr(stDramStride));
		rewriter.create<ConfigStOp>(
			loc, strideValue, act, llvm::APFloat(scale));
		rewriter.create<ConfigExOp>(loc, /*dataflow = */ WEIGHT_STATIONARY,
			/*act = */ 0, /*shift = */ 0,
			/*scale = */ llvm::APFloat((float)0), /*cStride = */ inputDilation,
			/*aStride = */ stride >> downsample,
			/*aTranspose = */ transInput3120, /*bTranspose*/ transWeight0132,
			/*setOnlyStrides = */ false);
		const int poolOutRowDim =
			(outRowDim + 2 * poolPadding - poolSize) / poolStride + 1;
		const int poolOutColDim =
			(outColDim + 2 * poolPadding - poolSize) / poolStride + 1;
		const int dilatedInRowDim =
			inRowDim + (inputDilation - 1) * (inRowDim - 1);
		const int dilatedInColDim =
			inColDim + (inputDilation - 1) * (inColDim - 1);

		int porowEnd = poolOutRowDim;

		for (int b = 0; b < batchSize; b += batches) {
			for (int porow = 0; porow < porowEnd; porow += porows) {
				const int orow = porow * poolStride - poolPadding;
				for (int pocol = 0; pocol < poolOutColDim; pocol += pocols) {
					const int ocol = pocol * poolStride - poolPadding;
					for (int poch = 0; poch < outChannels; poch += pochs) {
						for (int krow = 0; krow < kernelDim; krow += krows) {
							const int orow_floored = orow < 0 ? 0 : orow;

							int irow = orow_floored * stride +
								krow * kernelDilation - padding;
							for (int kcol = 0; kcol < kernelDim;
								 kcol += kcols) {
								const int ocol_floored = ocol < 0 ? 0 : ocol;
								int icol = ocol_floored * stride +
									kcol * kernelDilation - padding;

								for (int kch = 0; kch < inChannels;
									 kch += kchs) {
									TypedAttr offsetAttr =
										rewriter.getI64IntegerAttr(
											((b * poolOutRowDim *
													 poolOutColDim +
												 porow * poolOutColDim +
												 pocol) *
													outChannels +
												poch) *
											sizeOfElemT);
									Value offsetValue =
										rewriter.create<arith::ConstantOp>(
											loc, offsetAttr);
									Value out = rewriter.create<arith::AddIOp>(
										tileConvOp.getLoc(),
										rewriter.getI64Type(), output,
										offsetValue);
									if (transOutput1203) {
										offsetAttr = rewriter.getI64IntegerAttr(
											((porow * poolOutColDim *
													 batchSize +
												 pocol * batchSize + b) *
													outChannels +
												poch) *
											sizeOfElemT);
										offsetValue =
											rewriter.create<arith::ConstantOp>(
												loc, offsetAttr);
										out = rewriter.create<arith::AddIOp>(
											tileConvOp.getLoc(),
											rewriter.getI64Type(), output,
											offsetValue);
									}

									if (krow + krows < kernelDim ||
										kcol + kcols < kernelDim ||
										kch + kchs < inChannels) {
										out =
											rewriter.create<arith::ConstantOp>(
												tileConvOp.getLoc(),
												rewriter.getI64IntegerAttr(0));
									}
									Value pochValue =
										rewriter.create<arith::ConstantOp>(
											tileConvOp.getLoc(),
											rewriter.getI64IntegerAttr(
												poch * sizeOfAccT));
									Value bias_ =
										rewriter.create<arith::AddIOp>(
											tileConvOp.getLoc(),
											rewriter.getI64Type(), bias,
											pochValue);
									if (krow > 0 || kcol > 0 || kch > 0) {
										bias_ =
											rewriter.create<arith::ConstantOp>(
												tileConvOp.getLoc(),
												rewriter.getI64IntegerAttr(0));
									}

									const int batches_ = batchSize - b > batches
										? batches
										: batchSize - b;
									const int porows_ =
										poolOutRowDim - porow > porows
										? porows
										: poolOutRowDim - porow;
									const int pocols_ =
										poolOutColDim - pocol > pocols
										? pocols
										: poolOutColDim - pocol;
									const int pochs_ =
										outChannels - poch > pochs
										? pochs
										: outChannels - poch;
									const int krows_ = kernelDim - krow > krows
										? krows
										: kernelDim - krow;
									const int kcols_ = kernelDim - kcol > kcols
										? kcols
										: kernelDim - kcol;
									const int kchs_ = inChannels - kch > kchs
										? kchs
										: inChannels - kch;

									const int ocols_ =
										pocols_ * poolStride + poolSize - 1;
									const int orows_ =
										porows_ * poolStride + poolSize - 1;

									const int plpad = ocol < 0 ? -ocol : 0;
									const int prpad = ocol + ocols_ > outColDim
										? ocol + ocols_ - outColDim
										: 0;
									const int pupad = orow < 0 ? -orow : 0;
									const int pdpad = orow + orows_ > outRowDim
										? orow + orows_ - outRowDim
										: 0;

									const int dilatedKrows_ = krows_ +
										(kernelDilation - 1) * (krows_ - 1);
									const int dilatedKcols_ = kcols_ +
										(kernelDilation - 1) * (kcols_ - 1);

									const int icols_ =
										(ocols_ - plpad - prpad) * stride +
										dilatedKcols_ - 1;
									const int irows_ =
										(orows_ - pupad - pdpad) * stride +
										dilatedKrows_ - 1;

									int lpad = icol < 0 ? -icol : 0;
									int rpad = icol + icols_ > dilatedInColDim
										? icol + icols_ - dilatedInColDim
										: 0;
									int upad = irow < 0 ? -irow : 0;
									int dpad = irow + irows_ > dilatedInRowDim
										? irow + irows_ - dilatedInRowDim
										: 0;

									if (inputDilated) {
										lpad += lpad == 0 && icol % 2 != 0;
										rpad += rpad == 0 &&
											(icol + icols_) % 2 != 1;
										upad += upad == 0 && irow % 2 != 0;
										dpad += dpad == 0 &&
											(irow + irows_) % 2 != 1;
									}

									int krow_ = krow;
									int kcol_ = kcol;
									if (wrot180) {
										krow_ = kernelDim - krow - krows_;
										kcol_ = kernelDim - kcol - kcols_;
									}
									offsetAttr = rewriter.getI64IntegerAttr(
										((krow_ * kernelDim * inChannels +
											 kcol_ * inChannels + kch) *
												outChannels +
											poch) *
										sizeOfElemT);
									offsetValue =
										rewriter.create<arith::ConstantOp>(
											tileConvOp.getLoc(), offsetAttr);
									Value weightsSlice =
										rewriter.create<arith::AddIOp>(
											tileConvOp.getLoc(),
											rewriter.getI64Type(), weights,
											offsetValue);
									if (transWeight1203) {
										offsetAttr = rewriter.getI64IntegerAttr(
											((kch * kernelDim * kernelDim +
												 krow_ * kernelDim + kcol_) *
													outChannels +
												poch) *
											sizeOfElemT);
										offsetValue =
											rewriter.create<arith::ConstantOp>(
												tileConvOp.getLoc(),
												offsetAttr);
										weightsSlice =
											rewriter.create<arith::AddIOp>(
												tileConvOp.getLoc(),
												rewriter.getI64Type(), weights,
												offsetValue);
									} else if (transWeight0132) {
										offsetAttr = rewriter.getI64IntegerAttr(
											((krow_ * kernelDim * outChannels +
												 kcol_ * outChannels + poch) *
													inChannels +
												kch) *
											sizeOfElemT);
										offsetValue =
											rewriter.create<arith::ConstantOp>(
												tileConvOp.getLoc(),
												offsetAttr);
										weightsSlice =
											rewriter.create<arith::AddIOp>(
												tileConvOp.getLoc(),
												rewriter.getI64Type(), weights,
												offsetValue);
									}
									offsetAttr = rewriter.getI64IntegerAttr(
										((b * inRowDim * inColDim +
											 ((irow + upad) >> inputDilated) *
												 inColDim +
											 ((icol + lpad) >> inputDilated)) *
												inChannels +
											kch) *
										sizeOfElemT);
									offsetValue =
										rewriter.create<arith::ConstantOp>(
											tileConvOp.getLoc(), offsetAttr);
									Value in = rewriter.create<arith::AddIOp>(
										tileConvOp.getLoc(),
										rewriter.getI64Type(), input,
										offsetValue);
									if (transInput3120) {
										offsetAttr = rewriter.getI64IntegerAttr(
											((kch * inRowDim * inColDim +
												 ((irow + upad) >>
													 inputDilated) *
													 inColDim +
												 ((icol + lpad) >>
													 inputDilated)) *
													batchSize +
												b) *
											sizeOfElemT);
										in = rewriter.create<arith::AddIOp>(
											tileConvOp.getLoc(),
											rewriter.getI64Type(), input,
											offsetValue);
									}

									spTiledConv(batchSize, inRowDim, inColDim,
										inChannels, outChannels, outRowDim,
										outColDim, poolOutRowDim, poolOutColDim,
										stride, padding, kernelDim,
										kernelDilation, inStride, weightStride,
										outStride, poolSize, poolStride,
										poolPadding, batches_, porows_, pocols_,
										pochs_, krows_, kcols_, kchs_, lpad,
										rpad, upad, dpad, plpad, prpad, pupad,
										pdpad, in, weightsSlice, out, bias_,
										act, scale, wrot180, transOutput1203,
										transInput3120, transWeight1203,
										transWeight0132, noBias, noPool,
										downsample, inputDilated, false,
										tileConvOp, rewriter);
								}
							}
						}
					}
				}
			}
		}
		IntegerAttr flushAttr = rewriter.getI64IntegerAttr(0);
		Value flushValue = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI64Type(), flushAttr);
		rewriter.replaceOpWithNewOp<FlushIntrOp>(
			tileConvOp, flushValue, flushValue);
	}

	int tiledConvTotalSpadRows(bool acc, int stride, int inputDilation,
		int kernelDilation, bool downsample, bool transWeight0132,
		bool transInput3120, int batches, int porows, int pocols, int ochs,
		int krows, int kcols, int kchs, int poolSize, int poolStride) const {

		const int orows = porows * poolStride + poolSize - 1;
		const int ocols = pocols * poolStride + poolSize - 1;

		const int krowsDilated = krows + (kernelDilation - 1) * (krows - 1);
		const int kcolsDilated = kcols + (kernelDilation - 1) * (kcols - 1);

		int irows = orows * stride + krowsDilated - 1;
		int icols = ocols * stride + kcolsDilated - 1;
		const int ichs = kchs;

		irows = irows / inputDilation + (irows % inputDilation != 0);
		icols = icols / inputDilation + (icols % inputDilation != 0);

		const int inChannelsPerBank = ichs / dim + (ichs % dim != 0);
		const int outChannelsPerBank = ochs / dim + (ochs % dim != 0);
		const int batchesPerBank = batches / dim + (batches % dim != 0);

		const int aRows = transInput3120
			? (batchesPerBank * ichs * (irows >> downsample) *
				  (icols >> downsample))
			: (inChannelsPerBank * batches * (irows >> downsample) *
				  (icols >> downsample));

		const int bRows = transWeight0132
			? inChannelsPerBank * kcols * krows * ochs
			: outChannelsPerBank * kcols * krows * kchs;

		const int cRows = outChannelsPerBank * batches * orows * ocols;

		return acc ? cRows : aRows + bRows;
	}

  public:
	using ConvertOpToLLVMPattern<TileConvOp>::ConvertOpToLLVMPattern;
	explicit GemminiTileConvLowering(LLVMTypeConverter &typeConverter,
		int64_t dim, int64_t addrLen, int64_t accRows, int64_t bankRows,
		size_t sizeOfElemT, size_t sizeOfAccT)
		: ConvertOpToLLVMPattern(typeConverter), dim(dim), addrLen(addrLen),
		  accRows(accRows), bankRows(bankRows), sizeOfElemT(sizeOfElemT),
		  sizeOfAccT(sizeOfAccT) {}
	LogicalResult matchAndRewrite(TileConvOp tileConvOp, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value input = tileConvOp.getInput();
		Value output = tileConvOp.getOutput();
		Value weights = tileConvOp.getWeights();
		Value bias = tileConvOp.getBias();
		MemRefType inputType = dyn_cast<MemRefType>(input.getType());
		MemRefType biasType = dyn_cast<MemRefType>(bias.getType());
		ArrayRef<int64_t> inputShape = inputType.getShape();
		ArrayRef<int64_t> biasShape = biasType.getShape();

		Value outRowDimValue = tileConvOp.getOutRowDim();
		int outRowDim = getNumberFromValue(outRowDimValue);
		Value outColDimValue = tileConvOp.getOutColDim();
		int outColDim = getNumberFromValue(outColDimValue);
		Value kernelDimValue = tileConvOp.getKernelDim();
		int kernelDim = getNumberFromValue(kernelDimValue);
		int batchSize = inputShape[0];
		int inRowDim = inputShape[1];
		int inColDim = inputShape[2];
		int inChannels = inputShape[3];
		int outChannels = biasShape[0];
		int stride = tileConvOp.getStride();
		int inputDilation = tileConvOp.getInputDilation();
		int kernelDilation = tileConvOp.getKernelDilation();
		int padding = tileConvOp.getPadding();
		int act = tileConvOp.getAct();
		float scale = tileConvOp.getScale().convertToFloat();
		int poolSize = tileConvOp.getPoolSize();
		int poolStride = tileConvOp.getPoolStride();
		int poolPadding = tileConvOp.getPoolPadding();
		bool wrot180 = tileConvOp.getWrot180();
		bool transOutput1203 = tileConvOp.getTransOutput1203();
		bool transInput3120 = tileConvOp.getTransInput3120();
		bool transWeight1203 = tileConvOp.getTransWeight1203();
		bool transWeight0132 = tileConvOp.getTransWeight0132();
		Location loc = tileConvOp.getLoc();
		IntegerType i64Type = rewriter.getI64Type();
		Value inputExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, input);
		Value inputIndexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, inputExtractOp);
		Value outputExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, output);
		Value outputIndexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, outputExtractOp);
		Value biasExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, bias);
		Value biasIndexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, biasExtractOp);
		Value weightsExtractOp =
			rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
				loc, weights);
		Value weightsIndexCastOp =
			rewriter.create<arith::IndexCastOp>(loc, i64Type, weightsExtractOp);
		const bool noPool = poolSize == 0;
		if (noPool) {
			poolSize = 1;
			poolStride = 1;
			poolPadding = 0;
		}
		const int poolOutRowDim =
			(outRowDim + 2 * poolPadding - poolSize) / poolStride + 1;
		const int poolOutColDim =
			(outColDim + 2 * poolPadding - poolSize) / poolStride + 1;
		const bool downsample = stride == 2 && kernelDim == 1 && padding == 0 &&
			noPool && inRowDim % 2 == 0 && inColDim % 2 == 0;
		int args[] = {batchSize, poolOutRowDim, poolOutColDim, outChannels,
			kernelDim, kernelDim, inChannels};
		const int maxArgs[] = {batchSize, poolOutRowDim, poolOutColDim,
			outChannels, kernelDim, kernelDim, inChannels};
		const int orowsIdx = 1;
		const int ocolsIdx = 2;
		const int outChannelsIdx = 3;
		const int inChannelsIdx = 6;
		const int maxSpadRows = (BANK_NUM * bankRows / 2);
		const int maxAccRows = (accRows / 2);
		int spadRows = tiledConvTotalSpadRows(false, stride, inputDilation,
			kernelDilation, downsample, transWeight0132, transInput3120,
			args[0], args[1], args[2], args[3], args[4], args[5], args[6],
			poolSize, poolStride);
		int accRows = tiledConvTotalSpadRows(true, stride, inputDilation,
			kernelDilation, downsample, transWeight0132, transInput3120,
			args[0], args[1], args[2], args[3], args[4], args[5], args[6],
			poolSize, poolStride);
		while (spadRows > maxSpadRows || accRows > maxAccRows) {
			int maxVal = -1;
			int maxIdx = -1;
			for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
				if (!(i == ocolsIdx && args[i] <= dim && args[orowsIdx] > 1) &&
					args[i] > maxVal) {
					maxVal = args[i];
					maxIdx = i;
				}
			}

			if (maxIdx == outChannelsIdx || maxIdx == inChannelsIdx) {
				if (args[maxIdx] % dim != 0) {
					args[maxIdx] = (args[maxIdx] / dim) * dim;
				} else {
					args[maxIdx] -= dim;
				}
				args[maxIdx] = args[maxIdx] == 0 ? 1 : args[maxIdx];
			} else {
				args[maxIdx]--;
			}
			spadRows = tiledConvTotalSpadRows(false, stride, inputDilation,
				kernelDilation, downsample, transWeight0132, transInput3120,
				args[0], args[1], args[2], args[3], args[4], args[5], args[6],
				poolSize, poolStride);
			accRows = tiledConvTotalSpadRows(true, stride, inputDilation,
				kernelDilation, downsample, transWeight0132, transInput3120,
				args[0], args[1], args[2], args[3], args[4], args[5], args[6],
				poolSize, poolStride);
		}
		bool notIncreased = false;
		while (!notIncreased) {
			notIncreased = true;

			int argsCandidate[] = {
				args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
			argsCandidate[ocolsIdx]++;

			if (argsCandidate[ocolsIdx] > maxArgs[ocolsIdx])
				continue;

			spadRows = tiledConvTotalSpadRows(false, stride, inputDilation,
				kernelDilation, downsample, transWeight0132, transInput3120,
				argsCandidate[0], argsCandidate[1], argsCandidate[2],
				argsCandidate[3], argsCandidate[4], argsCandidate[5],
				argsCandidate[6], poolSize, poolStride);
			accRows = tiledConvTotalSpadRows(true, stride, inputDilation,
				kernelDilation, downsample, transWeight0132, transInput3120,
				argsCandidate[0], argsCandidate[1], argsCandidate[2],
				argsCandidate[3], argsCandidate[4], argsCandidate[5],
				argsCandidate[6], poolSize, poolStride);

			if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
				args[ocolsIdx] = argsCandidate[ocolsIdx];
				notIncreased = false;
			}
		}

		bool nothingIncreased = false;
		while (!nothingIncreased) {
			nothingIncreased = true;
			for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++) {
				int argsCandidate[] = {args[0], args[1], args[2], args[3],
					args[4], args[5], args[6]};
				argsCandidate[i]++;

				if (argsCandidate[i] > maxArgs[i])
					continue;
				spadRows = tiledConvTotalSpadRows(false, stride, inputDilation,
					kernelDilation, downsample, transWeight0132, transInput3120,
					argsCandidate[0], argsCandidate[1], argsCandidate[2],
					argsCandidate[3], argsCandidate[4], argsCandidate[5],
					argsCandidate[6], poolSize, poolStride);
				accRows = tiledConvTotalSpadRows(true, stride, inputDilation,
					kernelDilation, downsample, transWeight0132, transInput3120,
					argsCandidate[0], argsCandidate[1], argsCandidate[2],
					argsCandidate[3], argsCandidate[4], argsCandidate[5],
					argsCandidate[6], poolSize, poolStride);

				if (spadRows <= maxSpadRows && accRows <= maxAccRows) {
					args[i] = argsCandidate[i];
					nothingIncreased = false;
				}
			}
		}
		const int batches = args[0];
		const int orows = args[1];
		const int ocols = args[2];
		const int ochs = args[3];
		const int krows = args[4];
		const int kcols = args[5];
		const int kchs = args[6];

		const int inStride = inChannels;
		const int outStride = outChannels;
		const int weightStride = outChannels;
		tiledConv(batchSize, inRowDim, inColDim, inChannels, outChannels,
			outRowDim, outColDim, stride, inputDilation, kernelDilation,
			padding, kernelDim, inStride, weightStride, outStride, wrot180,
			transOutput1203, transInput3120, transWeight1203, transWeight0132,
			batches, orows, ocols, ochs, krows, kcols, kchs, inputIndexCastOp,
			weightsIndexCastOp, biasIndexCastOp, outputIndexCastOp, act, scale,
			poolSize, noPool ? 0 : poolStride, poolPadding, tileConvOp,
			rewriter);

		insertFence(loc, rewriter);

		return success();
	}

  private:
	int64_t dim;
	int64_t addrLen;
	int64_t accRows;
	int64_t bankRows;
	size_t sizeOfElemT;
	size_t sizeOfAccT;
};

void mlir::populateGemminiLegalizeForLLVMExportPatterns(
	LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
	int64_t addrLen, int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
	size_t sizeOfAccT, int64_t mxFormat, bool clampSingleBlockMvin,
	bool useLoopWs, bool dispatchDebug) {
	patterns.add<ForwardOperands<func::CallOp>,
		ForwardOperands<func::CallIndirectOp>, ForwardOperands<func::ReturnOp>>(
		converter, &converter.getContext());
	patterns.add<GemminiFlushLowering>(converter);
	patterns.add<GemminiConfigStLowering>(converter);
	patterns.add<GemminiConfigLdLowering>(converter, dim);
	patterns.add<GemminiMvinLowering>(converter, addrLen);
	patterns.add<GemminiMvin2Lowering>(converter, addrLen);
	patterns.add<GemminiMvin3Lowering>(converter, addrLen);
	patterns.add<GemminiMvoutLowering>(converter, addrLen);
	patterns.add<GemminiConfigExLowering>(converter, mxFormat);
	patterns.add<GemminiConfigNormLowering>(converter);
	patterns.add<GemminiPreloadZerosLowering>(converter, dim, addrLen);
	patterns.add<GemminiPreloadLowering>(converter, addrLen);
	patterns.add<GemminiComputePreloadedLowering>(converter, addrLen);
	patterns.add<GemminiComputeAccumulatedLowering>(converter, addrLen);
	patterns.add<GemminiTileMatMulLowering>(converter, dim, addrLen, accRows,
		bankRows, sizeOfElemT, sizeOfAccT, clampSingleBlockMvin, useLoopWs,
		mxFormat, dispatchDebug);
	patterns.add<GemminiTileConvLowering>(
		converter, dim, addrLen, accRows, bankRows, sizeOfElemT, sizeOfAccT);
}

void mlir::configureGemminiLegalizeForExportTarget(
	LLVMConversionTarget &target) {
	target.addLegalOp<FlushIntrOp, ConfigIntrOp, MvinIntrOp, Mvin2IntrOp,
		Mvin3IntrOp, MvoutIntrOp, PreloadIntrOp, ComputePreloadedIntrOp,
		ComputeAccumulatedIntrOp, LoopWsConfigBoundsIntrOp,
		LoopWsConfigAddrsABIntrOp, LoopWsConfigAddrsDCIntrOp,
		LoopWsConfigStridesABIntrOp, LoopWsConfigStridesDCIntrOp,
		LoopWsIntrOp, LoopConvWsConfig1IntrOp, LoopConvWsConfig2IntrOp,
		LoopConvWsConfig3IntrOp, LoopConvWsConfig4IntrOp,
		LoopConvWsConfig5IntrOp, LoopConvWsConfig6IntrOp,
		LoopConvWsIntrOp>();
	target.addIllegalOp<FlushOp, ConfigStOp, ConfigLdOp, ConfigExOp, MvinOp,
		Mvin2Op, Mvin3Op, MvoutOp, PrintOp, PreloadZerosOp, PreloadOp,
		ComputePreloadedOp, ComputeAccumulatedOp, TileMatMulOp, TileConvOp,
		ConfigNormOp>();
}

//===----------------------------------------------------------------------===//
// Pass wrapper exposing the populate/configure helpers above as a plain
// FunctionOpInterface pass so it can be plugged into the Gemmini compiler
// pipeline alongside the recovery passes.
//===----------------------------------------------------------------------===//

#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINILEGALIZEFORLLVMEXPORTPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

struct GemminiLegalizeForLLVMExportPass final
	: public impl::GemminiLegalizeForLLVMExportPassBase<
		  GemminiLegalizeForLLVMExportPass> {
	using Base = impl::GemminiLegalizeForLLVMExportPassBase<
		GemminiLegalizeForLLVMExportPass>;
	using Base::Base;

	explicit GemminiLegalizeForLLVMExportPass(
		const GemminiTransformOptions &transformOpts) {
		// Initialize the tablegen-generated pass options from the
		// in-memory descriptor. This is the path used by the host-scope
		// plugin pipeline; the textual-pipeline path goes through the
		// tablegen-parsed options directly.
		this->dim = transformOpts.target.dim;
		this->addrLen = transformOpts.target.addrLen;
		this->accRows = transformOpts.target.accRows;
		this->bankRows = transformOpts.target.bankRows;
		this->elemBits = transformOpts.target.elemBits;
		this->accBits = transformOpts.target.accBits;
		this->mxFormat = static_cast<int64_t>(transformOpts.target.mxFormat);
		this->commandIssue =
			transformOpts.target.commandIssue == CommandIssue::MMIO ? "mmio"
																	: "rocc";
		this->loopWs = transformOpts.target.useLoopWs;
		this->dispatchDebug = transformOpts.target.dispatchDebug;
	}

	void runOnOperation() override {
		auto func = getOperation();
		MLIRContext *ctx = &getContext();

		LowerToLLVMOptions llvmOptions(
			ctx, DataLayout::closest(func.getOperation()));
		LLVMTypeConverter converter(ctx, llvmOptions);

		RewritePatternSet patterns(ctx);
		// Hardware descriptor: read from the tablegen pass options. Defaults
		// (set in Passes.td) match the Spike libgemmini.so build (DIM=16,
		// ADDR_LEN=32, ACC_ROWS=1024, BANK_ROWS=4096, elem_t=int8_t,
		// acc_t=int32_t). Overridden via plugin flags
		// (--iree-gemmini-{dim,addr-len,acc-rows,bank-rows,elem-bits,acc-bits})
		// or via the textual pipeline syntax
		// `merlin-gemmini-legalize-for-llvm-export{dim=16 addr-len=32 ...}`
		// for the inside-dispatch codegen path.
		// IMPORTANT: addrLen MUST match libgemmini's runtime addr_len — see
		// dev-blog 14.13 for the symptom of a mismatch (silent SPAD-slot
		// corruption, matmul produces all zeros).
		const size_t sizeOfElemT =
			static_cast<size_t>(this->elemBits.getValue() / 8);
		const size_t sizeOfAccT =
			static_cast<size_t>(this->accBits.getValue() / 8);
		const bool clampSingleBlockMvin =
			this->commandIssue.getValue() == "mmio";
		const bool useLoopWs = this->loopWs.getValue();
		const bool dispatchDebug = this->dispatchDebug.getValue();
		populateGemminiLegalizeForLLVMExportPatterns(converter, patterns,
			this->dim.getValue(), this->addrLen.getValue(),
			this->accRows.getValue(), this->bankRows.getValue(), sizeOfElemT,
			sizeOfAccT, this->mxFormat.getValue(), clampSingleBlockMvin,
			useLoopWs, dispatchDebug);

		LLVMConversionTarget target(*ctx);
		configureGemminiLegalizeForExportTarget(target);
		// Anything not in the Gemmini illegal list stays legal.
		target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

		if (failed(applyPartialConversion(
				func.getOperation(), target, std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

// createGemminiLegalizeForLLVMExportPass() is auto-generated as a friend of
// GemminiLegalizeForLLVMExportPassBase by GEN_PASS_DEF_*; do not redeclare.

std::unique_ptr<Pass> createGemminiLegalizeForLLVMExportPassWithOptions(
	const GemminiTransformOptions &options) {
	return std::make_unique<GemminiLegalizeForLLVMExportPass>(options);
}

} // namespace mlir::iree_compiler::Gemmini
