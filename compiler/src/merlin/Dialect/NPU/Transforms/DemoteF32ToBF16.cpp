/// DemoteF32ToBF16 — convert all f32 types to bf16 throughout the module.
///
/// This is the bf16 equivalent of IREE's --iree-input-demote-f32-to-f16.
/// Walks every operation and replaces f32 element types with bf16.
/// Integer types and fp8 types are left unchanged.
///
/// This ensures zero f32 in the output MLIR, matching the NPU hardware
/// where the VPU operates exclusively in bf16.

#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::NPU {
namespace {

/// Replace f32 with bf16 in a type.
static Type demoteType(Type type) {
  if (type.isF32())
    return BFloat16Type::get(type.getContext());
  if (auto shaped = dyn_cast<RankedTensorType>(type)) {
    if (shaped.getElementType().isF32()) {
      return RankedTensorType::get(shaped.getShape(),
                                   BFloat16Type::get(type.getContext()),
                                   shaped.getEncoding());
    }
  }
  if (auto shaped = dyn_cast<UnrankedTensorType>(type)) {
    if (shaped.getElementType().isF32()) {
      return UnrankedTensorType::get(BFloat16Type::get(type.getContext()));
    }
  }
  return type;
}

/// Check if a type contains f32.
static bool hasF32(Type type) {
  if (type.isF32())
    return true;
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType().isF32();
  return false;
}

struct DemoteF32ToBF16Pass
    : public PassWrapper<DemoteF32ToBF16Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DemoteF32ToBF16Pass)

  StringRef getArgument() const override { return "demote-f32-to-bf16"; }

  StringRef getDescription() const override {
    return "Convert all f32 types to bf16 throughout the module. "
           "Matches NPU hardware where VPU operates in bf16 only.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Type bf16Type = BFloat16Type::get(ctx);

    module.walk([&](Operation *op) {
      bool modified = false;

      // Demote result types
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        Type oldType = op->getResult(i).getType();
        Type newType = demoteType(oldType);
        if (oldType != newType) {
          op->getResult(i).setType(newType);
          modified = true;
        }
      }

      // Demote block argument types in regions
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument &arg : block.getArguments()) {
            Type oldType = arg.getType();
            Type newType = demoteType(oldType);
            if (oldType != newType) {
              arg.setType(newType);
              modified = true;
            }
          }
        }
      }

      // Update f32 constant attributes
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
          if (floatAttr.getType().isF32()) {
            double val = floatAttr.getValueAsDouble();
            auto newAttr = FloatAttr::get(bf16Type, val);
            constOp.setValueAttr(newAttr);
          }
        }
      }

    });

    // Second pass: fix extf/truncf that became identity after demoting
    SmallVector<Operation *> toErase;
    module.walk([&](Operation *op) {
      if (isa<arith::ExtFOp, arith::TruncFOp>(op)) {
        if (op->getNumOperands() == 1 && op->getNumResults() == 1) {
          Type inType = op->getOperand(0).getType();
          Type outType = op->getResult(0).getType();
          if (inType == outType) {
            op->getResult(0).replaceAllUsesWith(op->getOperand(0));
            toErase.push_back(op);
          }
        }
      }
    });
    for (auto *op : llvm::reverse(toErase))
      op->erase();

    // Demote function signature types (works with any function-like op)
    module.walk([&](FunctionOpInterface func) {
      auto funcType = func.getFunctionType();
      if (auto ft = dyn_cast<FunctionType>(funcType)) {
        bool changed = false;
        SmallVector<Type> newInputs;
        for (Type t : ft.getInputs()) {
          Type newT = demoteType(t);
          if (newT != t) changed = true;
          newInputs.push_back(newT);
        }
        SmallVector<Type> newResults;
        for (Type t : ft.getResults()) {
          Type newT = demoteType(t);
          if (newT != t) changed = true;
          newResults.push_back(newT);
        }
        if (changed)
          func.setType(FunctionType::get(ctx, newInputs, newResults));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createDemoteF32ToBF16Pass() {
  return std::make_unique<DemoteF32ToBF16Pass>();
}

void registerDemoteF32ToBF16Pass() {
  PassRegistration<DemoteF32ToBF16Pass>();
}

} // namespace mlir::iree_compiler::NPU
