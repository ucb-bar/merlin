//===- LowerIntrToMmio.cpp - Lower gemmini.intr.* → MMIO stores ----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//
//
// For systems where Gemmini sits as a cluster-side MMIO peripheral instead
// of a RoCC-attached accelerator (e.g. chipyard `RadianceGemminiOnlyConfig`,
// in which the Rocket "small core" has no RoCC), the
// `gemmini-lower-intr-to-mmio` pass replaces every `gemmini.intr.<op>(rs1,
// rs2)` produced by `merlin-gemmini-legalize-for-llvm-export` with a
// three-store sequence to the gemmini control window:
//
//     llvm.store volatile %rs1,       (mmioBase + 0x10)
//     llvm.store volatile %rs2,       (mmioBase + 0x18)
//     llvm.store volatile %instWord,  (mmioBase + 0x00)
//
// where `instWord` is a 32-bit packed RoCC instruction word
//   0x7B | (3 << 12) | (1 << 15) | (2 << 20) | (funct << 25)
// matching the encoding in
// `chipyard/.../gemmini-rocc-tests/bareMetalC/matmul_ws_mx_generic.c:84`.
//
// Each gemmini.intr.* op has a known funct (per
// `chipyard/.../gemmini-rocc-tests/include/gemmini.h:31-67`):
//   CONFIG = 0, MVIN2 = 1, MVIN = 2, MVOUT = 3, COMPUTE_PRELOADED = 4,
//   COMPUTE_ACCUMULATE = 5, PRELOAD = 6, FLUSH = 7,
//   LOOP_WS = 8, LOOP_WS_CONFIG_BOUNDS = 9, ..., LOOP_CONV_WS = 15,
//   LOOP_CONV_WS_CONFIG_{1..6} = 16..21, MVIN3 = 14,
//   CONFIG_SCALE_MEM = 26.
//
// After this pass runs there are no `gemmini.intr.*` ops left, so the
// `GemminiToLLVMIRTranslation` interface is a no-op and the standard
// `convert-to-llvm` finishes lowering everything else.
//
//===----------------------------------------------------------------------===//

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINILOWERINTRTOMMIOPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

// custom-3 base opcode + fixed rd/funct3/rs1/rs2 fields per the kernel
// reference; only `funct` (bits [31:25]) varies per op.
//   bits [6:0]   = 0x7B  (custom-3)
//   bits [11:7]  = 0     (rd)
//   bits [14:12] = 3     (funct3)
//   bits [19:15] = 1     (rs1)
//   bits [24:20] = 2     (rs2)
//   bits [31:25] = funct
static constexpr uint32_t kCustom3Base =
	0x7Bu | (0u << 7) | (3u << 12) | (1u << 15) | (2u << 20);

static uint32_t encodeInstWord(uint32_t funct) {
	return kCustom3Base | (funct << 25);
}

// Map each gemmini.intr.* op to its funct code. Returns -1 if the op is
// not a gemmini intrinsic.
static int functForIntr(Operation *op) {
	StringRef name = op->getName().getStringRef();
	if (name == "gemmini.intr.config")
		return 0;
	if (name == "gemmini.intr.mvin2")
		return 1;
	if (name == "gemmini.intr.mvin")
		return 2;
	if (name == "gemmini.intr.mvout")
		return 3;
	if (name == "gemmini.intr.compute.preloaded")
		return 4;
	if (name == "gemmini.intr.compute.accumulated")
		return 5;
	if (name == "gemmini.intr.preload")
		return 6;
	if (name == "gemmini.intr.flush")
		return 7;
	if (name == "gemmini.intr.loop_ws")
		return 8;
	if (name == "gemmini.intr.loop_ws.config_bounds")
		return 9;
	if (name == "gemmini.intr.loop_ws.config_addrs_ab")
		return 10;
	if (name == "gemmini.intr.loop_ws.config_addrs_dc")
		return 11;
	if (name == "gemmini.intr.loop_ws.config_strides_ab")
		return 12;
	if (name == "gemmini.intr.loop_ws.config_strides_dc")
		return 13;
	if (name == "gemmini.intr.mvin3")
		return 14;
	if (name == "gemmini.intr.loop_conv_ws")
		return 15;
	if (name == "gemmini.intr.loop_conv_ws.config1")
		return 16;
	if (name == "gemmini.intr.loop_conv_ws.config2")
		return 17;
	if (name == "gemmini.intr.loop_conv_ws.config3")
		return 18;
	if (name == "gemmini.intr.loop_conv_ws.config4")
		return 19;
	if (name == "gemmini.intr.loop_conv_ws.config5")
		return 20;
	if (name == "gemmini.intr.loop_conv_ws.config6")
		return 21;
	// mxGemmini extensions.
	if (name == "gemmini.intr.mvout_spad")
		return 23;
	if (name == "gemmini.intr.loop_ws.config_spad_ab")
		return 24;
	if (name == "gemmini.intr.mxquant_config")
		return 26;
	return -1;
}

// Build a volatile LLVM store of `value` to `address` (an i64 absolute
// address). Used by the rewrite below for the rs1/rs2/inst stores.
static void emitVolatileStore(
	OpBuilder &b, Location loc, Value address, Value value) {
	auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
	auto ptrCast = b.create<LLVM::IntToPtrOp>(loc, ptrTy, address);
	b.create<LLVM::StoreOp>(loc, value, ptrCast,
		/*alignment=*/0,
		/*isVolatile=*/true);
}

// Phase 8 MMIO synchronization: emit a busy-wait loop polling
// `*(uint32_t volatile *)(mmioBase + 0x20)` until it reads zero. The
// reference kernel's `gemmini_fence()` (matmul_ws_mx_generic.c:55-56)
// uses this exact pattern after `gemmini_flush(0)` so the host doesn't
// race ahead of the LOOP_WS hardware loop's MVOUTs into DRAM. Without
// this, the dispatch returns to IREE while gemmini is still writing C,
// and `iree_hal_device_transfer_d2h` reads stale memory.
//
// The emitted IR is:
//   <previous-block-tail>
//     llvm.br ^poll
//   ^poll:
//     %busy = llvm.load volatile (mmioBase + 0x20) : i32
//     %z = llvm.icmp eq %busy, 0
//     llvm.cond_br %z, ^cont, ^poll
//   ^cont:
//     <continue>
static void emitBusyWaitPoll(OpBuilder &b, Location loc, int64_t mmioBase) {
	auto i32Ty = b.getI32Type();
	auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());

	Value busyAddr = b.create<LLVM::ConstantOp>(
		loc, b.getI64Type(), b.getI64IntegerAttr(mmioBase + 0x20));
	Value busyPtr = b.create<LLVM::IntToPtrOp>(loc, ptrTy, busyAddr);

	Block *currentBlock = b.getInsertionBlock();
	Block *continuation = currentBlock->splitBlock(b.getInsertionPoint());

	Region *region = currentBlock->getParent();
	Block *pollBlock = b.createBlock(region, region->end());
	pollBlock->moveBefore(continuation);

	b.setInsertionPointToEnd(currentBlock);
	b.create<LLVM::BrOp>(loc, ValueRange{}, pollBlock);

	b.setInsertionPointToStart(pollBlock);
	Value busy = b.create<LLVM::LoadOp>(loc, i32Ty, busyPtr,
		/*alignment=*/0,
		/*isVolatile=*/true);
	Value zero = b.create<LLVM::ConstantOp>(loc, i32Ty, b.getI32IntegerAttr(0));
	Value isZero =
		b.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, busy, zero);
	b.create<LLVM::CondBrOp>(loc, isZero,
		/*trueDest=*/continuation, /*trueOperands=*/ValueRange{},
		/*falseDest=*/pollBlock, /*falseOperands=*/ValueRange{});

	b.setInsertionPointToStart(continuation);
}

// Lower a single gemmini.intr.* op into the rs1/rs2/inst MMIO triple-store.
// When `op` is a FLUSH (funct=7), append a busy-wait poll on
// `mmioBase + 0x20` so the dispatch doesn't return to IREE while gemmini is
// still writing C to DRAM under LOOP_WS. The poll mirrors the upstream
// reference kernel's `gemmini_fence()` macro.
static void rewriteIntrToMmio(Operation *op, int funct, int64_t mmioBase) {
	OpBuilder b(op);
	Location loc = op->getLoc();
	auto i64Ty = b.getI64Type();
	auto i32Ty = b.getI32Type();

	Value rs1 = op->getOperand(0);
	Value rs2 = op->getOperand(1);

	Value rs1Addr = b.create<LLVM::ConstantOp>(
		loc, i64Ty, b.getI64IntegerAttr(mmioBase + 0x10));
	Value rs2Addr = b.create<LLVM::ConstantOp>(
		loc, i64Ty, b.getI64IntegerAttr(mmioBase + 0x18));
	Value instAddr = b.create<LLVM::ConstantOp>(
		loc, i64Ty, b.getI64IntegerAttr(mmioBase + 0x00));

	uint32_t instWord = encodeInstWord(static_cast<uint32_t>(funct));
	Value instConst = b.create<LLVM::ConstantOp>(
		loc, i32Ty, b.getI32IntegerAttr(static_cast<int32_t>(instWord)));

	// Fence-free triple-store. Order matters: rs1, rs2 must be committed
	// before the instruction-word write triggers Gemmini. The volatile
	// attribute keeps the compiler from reordering them.
	emitVolatileStore(b, loc, rs1Addr, rs1);
	emitVolatileStore(b, loc, rs2Addr, rs2);
	emitVolatileStore(b, loc, instAddr, instConst);

	// Phase 8: after the FLUSH command is queued, busy-wait on
	// GEMMINI_BUSY_ADDR until the LOOP_WS hardware loop drains. Without
	// this, the dispatch returns to IREE before gemmini's internal MVOUTs
	// have completed and the host reads stale C from DRAM.
	if (funct == 7) {
		emitBusyWaitPoll(b, loc, mmioBase);
	}

	op->erase();
}

struct GemminiLowerIntrToMmioPass final
	: public impl::GemminiLowerIntrToMmioPassBase<GemminiLowerIntrToMmioPass> {
	using Base =
		impl::GemminiLowerIntrToMmioPassBase<GemminiLowerIntrToMmioPass>;
	using Base::Base;

	void runOnOperation() override {
		auto func = getOperation();
		const int64_t mmioBase = this->mmioBase.getValue();

		// Manual two-pass walk: first collect all gemmini.intr.* ops, then
		// rewrite them in order. The rewrite may split blocks (for the
		// FLUSH busy-wait poll), so we must collect first to avoid walking
		// a structure being mutated.
		llvm::SmallVector<std::pair<Operation *, int>> work;
		func->walk([&](Operation *op) {
			int funct = functForIntr(op);
			if (funct >= 0 && op->getNumOperands() == 2) {
				work.emplace_back(op, funct);
			}
		});
		for (auto [op, funct] : work) {
			rewriteIntrToMmio(op, funct, mmioBase);
		}
	}
};

} // namespace

// createGemminiLowerIntrToMmioPass() is auto-generated as a friend of
// GemminiLowerIntrToMmioPassBase by GEN_PASS_DEF_*; do not redeclare.

} // namespace mlir::iree_compiler::Gemmini
