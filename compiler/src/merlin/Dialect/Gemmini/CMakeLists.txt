add_subdirectory(IR)
add_subdirectory(Transforms)

# Create a library specifically for registration
add_mlir_library(MerlinDialectGemminiRegistration
  RegisterGemmini.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
    MerlinDialectGemmini              # The Dialect IR
    MerlinTargetLLVMIRDialectGemmini  # The Translation logic
    MLIRIR
)