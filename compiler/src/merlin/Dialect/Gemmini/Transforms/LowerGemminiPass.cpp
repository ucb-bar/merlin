//====- LowerGemminiPass.cpp - Gemmini Dialect Lowering Pass  -------------===//
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
//===----------------------------------------------------------------------===//
//
// This file defines Gemmini dialect lowering pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "merlin/Dialect/Gemmini/IR/GemminiDialect.h"
#include "merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "merlin/Dialect/Gemmini/Transforms/Transforms.h"

using namespace mlir;
using namespace merlin;

namespace {

template <typename OpTy>
void lowerGemminiIntrinsicOpToLLVMCall(ModuleOp module, StringRef intrinsic) {
  SmallVector<OpTy> ops;
  module.walk([&](OpTy op) { ops.push_back(op); });
  for (OpTy op : ops) {
  OpBuilder builder(op);
    auto call = builder.create<LLVM::CallIntrinsicOp>(
      op.getLoc(), builder.getStringAttr(intrinsic), op->getOperands());
  if (op->getNumResults() == 0) {
      op.erase();
  } else {
      op->replaceAllUsesWith(call->getResults());
      op.erase();
  }
  }
}

void lowerGemminiIntrinsicsToLLVMCalls(ModuleOp module) {
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::Flush_IntrOp>(
    module, "llvm.riscv.flush");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::ConfigSt_IntrOp>(
    module, "llvm.riscv.config.st");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::ConifgLd_IntrOp>(
    module, "llvm.riscv.config.ld");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::ConfigEX_IntrOp>(
    module, "llvm.riscv.config.ex");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::ConfigNorm_IntrOp>(
    module, "llvm.riscv.config.norm");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::Mvin_IntrOp>(module,
                              "llvm.riscv.mvin");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::Mvin2_IntrOp>(
    module, "llvm.riscv.mvin2");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::Mvin3_IntrOp>(
    module, "llvm.riscv.mvin3");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::Mvout_IntrOp>(
    module, "llvm.riscv.mvout");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::Preload_IntrOp>(
    module, "llvm.riscv.preload");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::ComputePreloaded_IntrOp>(
    module, "llvm.riscv.compute.preloaded");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::ComputeAccumulated_IntrOp>(
    module, "llvm.riscv.compute.accumulated");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopWsConfigBounds_IntrOp>(
    module, "llvm.riscv.loop.ws.config.bounds");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopWsConfigAddrsAB_IntrOp>(
    module, "llvm.riscv.loop.ws.config.addrs.ab");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopWsConfigAddrsDC_IntrOp>(
    module, "llvm.riscv.loop.ws.config.addrs.dc");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopWsConfigStridesAB_IntrOp>(
    module, "llvm.riscv.loop.ws.config.strides.ab");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopWsConfigStridesDC_IntrOp>(
    module, "llvm.riscv.loop.ws.config.strides.dc");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopWs_IntrOp>(module,
                               "llvm.riscv.loop.ws");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWsConfig1_IntrOp>(
    module, "llvm.riscv.loop.conv.ws.config1");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWsConfig2_IntrOp>(
    module, "llvm.riscv.loop.conv.ws.config2");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWsConfig3_IntrOp>(
    module, "llvm.riscv.loop.conv.ws.config3");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWsConfig4_IntrOp>(
    module, "llvm.riscv.loop.conv.ws.config4");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWsConfig5_IntrOp>(
    module, "llvm.riscv.loop.conv.ws.config5");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWsConfig6_IntrOp>(
    module, "llvm.riscv.loop.conv.ws.config6");
  lowerGemminiIntrinsicOpToLLVMCall<gemmini::LoopConvWs_IntrOp>(
    module, "llvm.riscv.loop.conv.ws");
}

LogicalResult runStubLowering(ModuleOp module) {
  auto ensureDeclaration = [&](StringRef calleeName,
                 ArrayRef<Type> inputTypes) {
  if (module.lookupSymbol<func::FuncOp>(calleeName)) {
    return;
  }
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(inputTypes, TypeRange{});
  auto callee =
    builder.create<func::FuncOp>(module.getLoc(), calleeName, funcType);
  callee.setPrivate();
  };

  SmallVector<gemmini::TileMatMulOp> matmuls;
  module.walk([&](gemmini::TileMatMulOp op) { matmuls.push_back(op); });
  for (gemmini::TileMatMulOp op : matmuls) {
  SmallVector<Type> inputTypes;
  inputTypes.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    inputTypes.push_back(operand.getType());
  }
  constexpr StringLiteral callee = "__iree_gemmini_tile_matmul";
  ensureDeclaration(callee, inputTypes);
  OpBuilder builder(op);
  builder.create<func::CallOp>(op.getLoc(), callee, TypeRange{},
                 op->getOperands());
  op.erase();
  }

  SmallVector<gemmini::TileConvOp> convs;
  module.walk([&](gemmini::TileConvOp op) { convs.push_back(op); });
  for (gemmini::TileConvOp op : convs) {
  SmallVector<Type> inputTypes;
  inputTypes.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    inputTypes.push_back(operand.getType());
  }
  constexpr StringLiteral callee = "__iree_gemmini_tile_conv";
  ensureDeclaration(callee, inputTypes);
  OpBuilder builder(op);
  builder.create<func::CallOp>(op.getLoc(), callee, TypeRange{},
                 op->getOperands());
  op.erase();
  }
  return success();
}

} // namespace

// PrintOpLowering refers to the toy.print op.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(gemmini::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = rewriter.getContext();
    auto memRefType = llvm::cast<MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    Type memElementType = memRefType.getElementType();
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst;
    if (memElementType == rewriter.getF32Type() ||
        memElementType == rewriter.getF64Type()) {
      formatSpecifierCst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    } else if (memElementType == rewriter.getI8Type() ||
               memElementType == rewriter.getI32Type()) {
      formatSpecifierCst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%d \0", 4), parentModule);
    }
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody()) {
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(loop.getInductionVar());

      rewriter.setInsertionPointToEnd(loop.getBody());

      if (i != e - 1) {
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                      newLineCst);
      }
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto printOp = cast<gemmini::PrintOp>(op);
    Value elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    if (elementLoad.getType() == rewriter.getF32Type()) {
      elementLoad = rewriter.create<mlir::LLVM::FPExtOp>(
          loc, rewriter.getF64Type(), elementLoad);
    } else if (elementLoad.getType() == rewriter.getI8Type()) {
      elementLoad = rewriter.create<mlir::LLVM::SExtOp>(
          loc, rewriter.getI32Type(), elementLoad);
    }
    rewriter.create<LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtr = LLVM::LLVMPointerType::get(context);
    return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtr, true);
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
      return SymbolRefAttr::get(context, "printf");
    }

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value), 0);
    }

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

namespace {
class LowerGemminiToLLVMPass
    : public PassWrapper<LowerGemminiToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemminiToLLVMPass)
  StringRef getArgument() const final { return "lower-gemmini"; }
  StringRef getDescription() const final {
    return "gemmini dialect lowering pass.";
  }
  LowerGemminiToLLVMPass() = default;
  LowerGemminiToLLVMPass(const LowerGemminiToLLVMPass &) {}

  Option<int64_t> dim{*this, "dim", llvm::cl::desc("Size of systolic array."),
                      llvm::cl::init(16)};
  Option<int64_t> addrLen{*this, "addr_len",
                          llvm::cl::desc("The length of address."),
                          llvm::cl::init(32)};
  Option<int64_t> accRows{*this, "acc_rows", llvm::cl::desc("The row of acc."),
                          llvm::cl::init(1024)};
  Option<int64_t> bankRows{*this, "bank_rows",
                           llvm::cl::desc("The row of the bank."),
                           llvm::cl::init(4096)};
  Option<std::string> elemType{*this, "elem_t",
                               llvm::cl::desc("The type of elem_t."),
                               llvm::cl::init("i8")};
  Option<std::string> accType{*this, "acc_t",
                              llvm::cl::desc("The type of acc_t."),
                              llvm::cl::init("i32")};

  Option<bool> useCStubLowering{
      *this, "use-c-stub-lowering",
      llvm::cl::desc("Use legacy C-stub call lowering instead of Gemmini "
                     "intrinsic lowering."),
      llvm::cl::init(false)};

  // Override explicitly to allow conditional dialect dependence.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<gemmini::GemminiDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerGemminiToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (useCStubLowering) {
    if (failed(runStubLowering(module))) {
      signalPassFailure();
    }
    return;
  }

  auto parseTypeSize = [&](StringRef type, StringRef optionName)
      -> FailureOr<size_t> {
    if (type == "i8")
      return static_cast<size_t>(1);
    if (type == "i16")
      return static_cast<size_t>(2);
    if (type == "i32" || type == "f32")
      return static_cast<size_t>(4);
    if (type == "i64" || type == "f64")
      return static_cast<size_t>(8);
    module.emitError() << "unsupported " << optionName << " value: " << type;
    return failure();
  };

  FailureOr<size_t> elemSize = parseTypeSize(elemType, "elem_t");
  FailureOr<size_t> accSize = parseTypeSize(accType, "acc_t");
  if (failed(elemSize) || failed(accSize)) {
    signalPassFailure();
    return;
  }

  LowerToLLVMOptions options(&getContext());
  LLVMTypeConverter converter(&getContext(), options);
  RewritePatternSet patterns(&getContext());
  populateGemminiLegalizeForLLVMExportPatterns(
      converter, patterns, dim, addrLen, accRows, bankRows, *elemSize,
      *accSize);

  LLVMConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  configureGemminiLegalizeForExportTarget(target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    module.emitError() << "failed to legalize Gemmini ops for intrinsic "
                          "lowering";
    signalPassFailure();
    return;
  }

  lowerGemminiIntrinsicsToLLVMCalls(module);
}

namespace mlir {
namespace merlin {
std::unique_ptr<Pass> createLowerGemminiPass() {
  return std::make_unique<LowerGemminiToLLVMPass>();
}

void registerLowerGemminiPass() { PassRegistration<LowerGemminiToLLVMPass>(); }
} // namespace merlin
} // namespace mlir
