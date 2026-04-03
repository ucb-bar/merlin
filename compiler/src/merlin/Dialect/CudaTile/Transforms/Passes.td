#ifndef MERLIN_COMPILER_DIALECT_CUDATILE_TRANSFORMS_PASSES_TD_
#define MERLIN_COMPILER_DIALECT_CUDATILE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def CudaTileConvertDataMovementPass :
    InterfacePass<"cuda-tile-convert-data-movement",
                  "mlir::FunctionOpInterface"> {
  let summary = "Convert data movement ops to cuda_tile text IR";
  let description = [{
    Converts tensor.extract_slice, tensor.insert_slice, linalg.copy,
    linalg.transpose, tensor.collapse_shape, tensor.expand_shape, and
    linalg.broadcast to cuda_tile make_tensor_view + load_view_tko +
    store_view_tko patterns.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CudaTileConvertElementwisePass :
    InterfacePass<"cuda-tile-convert-elementwise",
                  "mlir::FunctionOpInterface"> {
  let summary = "Convert elementwise ops to cuda_tile text IR";
  let description = [{
    Converts arith.addf/mulf/subf/divf, math.exp/tanh/sqrt, and type
    conversions to corresponding cuda_tile tile operations. Detects
    and fuses chains of elementwise ops between load/store.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect"
  ];
}

def CudaTileConvertReductionsPass :
    InterfacePass<"cuda-tile-convert-reductions",
                  "mlir::FunctionOpInterface"> {
  let summary = "Convert reduction ops to cuda_tile text IR";
  let description = [{
    Converts linalg.reduce with add/max/min/mul combiners to
    cuda_tile.reduce with appropriate identity and dimension.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def CudaTileConvertContractionsPass :
    InterfacePass<"cuda-tile-convert-contractions",
                  "mlir::FunctionOpInterface"> {
  let summary = "Convert contraction ops to cuda_tile text IR";
  let description = [{
    Converts linalg.matmul, linalg.batch_matmul, and linalg.conv_2d_*
    to cuda_tile.mmaf/mmai with K-loop tiling.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def CudaTileConvertSCFPass :
    InterfacePass<"cuda-tile-convert-scf",
                  "mlir::FunctionOpInterface"> {
  let summary = "Convert standalone SCF ops to cuda_tile text IR";
  let description = [{
    Converts scf.for/if/while to cuda_tile.for/if/loop+break.
    Most SCF is already handled by the contraction pass (K-loop);
    this pass handles remaining standalone SCF.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect"
  ];
}

#endif // MERLIN_COMPILER_DIALECT_CUDATILE_TRANSFORMS_PASSES_TD_
