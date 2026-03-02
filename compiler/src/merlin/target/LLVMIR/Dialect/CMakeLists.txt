add_mlir_library(MerlinTargetLLVMIRDialectGemmini
  GemminiToLLVMIRTranslation.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
    MerlinDialectGemmini
    MLIRIR
    MLIRLLVMDialect
    MLIRTargetLLVMIR
)