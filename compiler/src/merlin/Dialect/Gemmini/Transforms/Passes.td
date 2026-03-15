#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_TD_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def GemminiConvertToGemminiPass :
    InterfacePass<"gemmini-convert-to-gemmini", "mlir::FunctionOpInterface"> {
  let summary = "Recover Gemmini ops from normalized linalg.generic IR";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def GemminiCanonicalizePass :
    InterfacePass<"gemmini-canonicalize", "mlir::FunctionOpInterface"> {
  let summary = "Canonicalize Gemmini ops";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect"
  ];
}

def GemminiLowerToISAPass :
    InterfacePass<"gemmini-lower-to-isa", "mlir::FunctionOpInterface"> {
  let summary = "Lower high-level Gemmini ops to staged ISA-like forms";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect"
  ];
}

def GemminiLowerToIREEPass :
    InterfacePass<"gemmini-lower-gemmini-to-iree", "mlir::FunctionOpInterface"> {
  let summary = "Lower Gemmini ops back to ordinary IREE/MLIR IR";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_TD_
