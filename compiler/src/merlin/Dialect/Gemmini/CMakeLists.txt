add_subdirectory(IR)
add_subdirectory(Transforms)

iree_cc_library(
  NAME
    GemminiRegistration
  HDRS
    "RegisterGemmini.h"
  SRCS
    "RegisterGemmini.cpp"
  DEPS
    merlin::Dialect::Gemmini::IR::GemminiDialect
    merlin::Dialect::Gemmini::Transforms::GemminiTransforms
    MLIRIR
    MLIRPass
  PUBLIC
)