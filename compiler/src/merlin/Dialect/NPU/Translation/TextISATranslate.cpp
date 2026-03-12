#include "compiler/src/merlin/Dialect/NPU/Translation/TranslateToTextISA.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace mlir::iree_compiler::NPU {
namespace {

LogicalResult translateToTextISA(Operation *op, llvm::raw_ostream &os) {
	op->walk([&](Operation *nested) {
		if (auto load = llvm::dyn_cast<NPUISA::DmaLoadOp>(nested)) {
			os << "dma.load"
			   << " rd=" << load.getRd() << ", base=" << load.getBase()
			   << ", size=" << load.getSize() << ", flag=" << load.getFlag()
			   << "\n";
			return;
		}
		if (auto load = llvm::dyn_cast<NPUISA::DmaLoadMxu0Op>(nested)) {
			os << "dma.load.mxu0"
			   << " rd=" << load.getRd() << ", base=" << load.getBase()
			   << ", size=" << load.getSize() << ", flag=" << load.getFlag()
			   << "\n";
			return;
		}
		if (auto load = llvm::dyn_cast<NPUISA::DmaLoadMxu1Op>(nested)) {
			os << "dma.load.mxu1"
			   << " rd=" << load.getRd() << ", base=" << load.getBase()
			   << ", size=" << load.getSize() << ", flag=" << load.getFlag()
			   << "\n";
			return;
		}
		if (auto store = llvm::dyn_cast<NPUISA::DmaStoreOp>(nested)) {
			os << "dma.store"
			   << " rs1=" << store.getRs1() << ", base=" << store.getBase()
			   << ", size=" << store.getSize() << ", flag=" << store.getFlag()
			   << "\n";
			return;
		}
		if (auto wait = llvm::dyn_cast<NPUISA::DmaWaitOp>(nested)) {
			os << "dma.wait"
			   << " flag=" << wait.getFlag() << "\n";
			return;
		}
		if (auto matmul = llvm::dyn_cast<NPUISA::MatmulMxu0Op>(nested)) {
			os << "matmul.mxu0"
			   << " rd=" << matmul.getRd() << ", rs1=" << matmul.getRs1()
			   << ", rs2=" << matmul.getRs2() << "\n";
			return;
		}
		if (auto vmul = llvm::dyn_cast<NPUISA::VMulOp>(nested)) {
			os << "vmul"
			   << " vrd=" << vmul.getVrd() << ", vs1=" << vmul.getVs1()
			   << ", vs2=" << vmul.getVs2() << "\n";
			return;
		}
		if (auto vexp = llvm::dyn_cast<NPUISA::VExpOp>(nested)) {
			os << "vexp"
			   << " vrd=" << vexp.getVrd() << ", vs1=" << vexp.getVs1() << "\n";
			return;
		}
		if (auto vrsum = llvm::dyn_cast<NPUISA::VReduceSumOp>(nested)) {
			os << "vreduce.sum"
			   << " vrd=" << vrsum.getVrd() << ", vs1=" << vrsum.getVs1()
			   << "\n";
			return;
		}
		if (auto vrcp = llvm::dyn_cast<NPUISA::VRcpOp>(nested)) {
			os << "vrcp"
			   << " vrd=" << vrcp.getVrd() << ", vs1=" << vrcp.getVs1() << "\n";
			return;
		}
	});

	return success();
}

} // namespace

void registerToTextISATranslation() {
	TranslateFromMLIRRegistration reg("mlir-to-npu-text-isa",
		"Translate npu_isa ops to simulator-compatible textual ISA",
		translateToTextISA, [](DialectRegistry &registry) {
			registry.insert<arith::ArithDialect>();
			registry.insert<func::FuncDialect>();
			registry.insert<linalg::LinalgDialect>();
			registry.insert<memref::MemRefDialect>();
			registry.insert<scf::SCFDialect>();
			registry.insert<tensor::TensorDialect>();
			registry.insert<NPUISA::NPUISADialect>();
		});
}

} // namespace mlir::iree_compiler::NPU
