add_mlir_library(MerlinDialectGemminiTransforms
  LegalizeForLLVMExport.cpp
  
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
    MerlinDialectGemmini
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
)