// Synthetic fixture.

include "mlir/Pass/PassBase.td"

def CudaTileLowering : Pass<"cuda-tile-lowering"> {
  let summary = "synthetic pass";
}
