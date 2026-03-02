# Define the dialect library
add_mlir_dialect(MerlinDialectGemmini
  TD_FILE Gemmini.td
  SOURCES
    GemminiDialect.cpp
  DEPENDS
    MLIRFuncDialect
    MLIRLLVMDialect
)

# Link dependencies
target_link_libraries(MerlinDialectGemmini
  PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRSideEffectInterfaces
)