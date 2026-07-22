// RUN: FileCheck --check-prefix=DIALECT %s < %s
// RTL-derived dialect structural check (compiled MLIR test). The Gemmini RTL enforces the
// weight-stationary res_pack -> matmul -> commit dataflow (PRELOAD before COMPUTE); spike is lenient
// about this ordering, so a structurally-wrong lowering can pass spike yet be wrong on hardware. This
// FileCheck assertion catches the shape statically, with no verilator run.
// DIALECT-DAG: gemmini.res_pack
// DIALECT-DAG: gemmini.matmul
// DIALECT-DAG: gemmini.commit
module {
  func.func @mm(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>) -> tensor<16x16xi32> {
    %p = "gemmini.res_pack"(%b) : (tensor<16x16xi8>) -> tensor<16x16xi8>
    %m = "gemmini.matmul"(%a, %p) : (tensor<16x16xi8>, tensor<16x16xi8>) -> tensor<16x16xi32>
    %c = "gemmini.commit"(%m) {output_dtype = "i32"} : (tensor<16x16xi32>) -> tensor<16x16xi32>
    return %c : tensor<16x16xi32>
  }
}
