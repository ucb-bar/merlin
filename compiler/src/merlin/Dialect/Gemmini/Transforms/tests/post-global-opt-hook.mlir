// Drives the gemmini plugin's post-global-opt hook in the native-codegen
// path (`--iree-gemmini-lower-back-to-iree=false`, the default shipped by
// `models/gemmini_spike.yaml`). In this path the plugin does NOT lower
// linalg.matmul into gemmini.matmul_tile at host scope; instead it attaches
// `iree_codegen.compilation_info` on each linalg.matmul, pinning a textual
// `iree_codegen.pass_pipeline` that runs the gemmini recovery + ISA-tier
// lowering inside the dispatch executable (via PipelineAttrInterface).
//
// The host-recovery + lower-back-to-IREE path still exists (the
// `--iree-gemmini-lower-back-to-iree=true` flag); see `convert-to-gemmini.mlir`
// and `matmul-lower-to-isa.mlir` for direct lit coverage of the recovery
// patterns at the dialect level.
// RUN: iree-compile %s --iree-input-type=none --iree-hal-target-backends=llvm-cpu --compile-to=global-optimization --iree-plugin=gemmini --iree-gemmini-enable --iree-gemmini-lower-back-to-iree=false | FileCheck %s --check-prefix=CHECK-I8
// RUN: iree-compile %s --iree-input-type=none --iree-hal-target-backends=llvm-cpu --compile-to=global-optimization --iree-plugin=gemmini --iree-gemmini-enable --iree-gemmini-enable-fp8-matmul --iree-gemmini-lower-back-to-iree=false | FileCheck %s --check-prefix=CHECK-FP8

#lhs_map = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs_map = affine_map<(d0, d1, d2) -> (d1, d2)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @main(%lhs: tensor<16x1024xi8>, %rhs: tensor<128x1024xi8>) -> tensor<16x128xi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x128xi32>
  %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<16x128xi32>) -> tensor<16x128xi32>
  %0 = linalg.generic
      {indexing_maps = [
         affine_map<(d0, d1, d2) -> (d0, d2)>,
         affine_map<(d0, d1, d2) -> (d1, d2)>,
         affine_map<(d0, d1, d2) -> (d0, d1)>
       ],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x1024xi8>, tensor<128x1024xi8>)
      outs(%init : tensor<16x128xi32>) {
    ^bb0(%in0: i8, %in1: i8, %acc: i32):
      %0 = arith.extsi %in0 : i8 to i32
      %1 = arith.extsi %in1 : i8 to i32
      %2 = arith.muli %0, %1 : i32
      %3 = arith.addi %acc, %2 : i32
      linalg.yield %3 : i32
  } -> tensor<16x128xi32>
  return %0 : tensor<16x128xi32>
}

func.func @main_named(%lhs: tensor<16x32xi8>, %rhs: tensor<64x32xi8>) -> tensor<16x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x64xi32>
  %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<16x64xi32>) -> tensor<16x64xi32>
  %0 = linalg.matmul indexing_maps = [#lhs_map, #rhs_map, #out_map]
      ins(%lhs, %rhs : tensor<16x32xi8>, tensor<64x32xi8>)
      outs(%init : tensor<16x64xi32>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>
}

func.func @main_fp8(%lhs: tensor<16x32xf8E4M3FN>, %rhs: tensor<64x32xf8E4M3FN>) -> tensor<16x64xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<16x64xbf16>
  %init = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<16x64xbf16>) -> tensor<16x64xbf16>
  %0 = linalg.matmul indexing_maps = [#lhs_map, #rhs_map, #out_map]
      ins(%lhs, %rhs : tensor<16x32xf8E4M3FN>, tensor<64x32xf8E4M3FN>)
      outs(%init : tensor<16x64xbf16>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// In the native-codegen path the gemmini matmul ops never appear at host
// scope; instead the linalg ops carry an `iree_codegen.compilation_info`
// attribute that pins our textual `iree_codegen.pass_pipeline` for use
// inside the dispatch executable. The textual pipeline body is opaque to
// FileCheck so we just verify it is present and references the gemmini
// passes.

// CHECK-I8-LABEL: util.func public @main
// CHECK-I8: compilation_info = #iree_codegen.compilation_info
// CHECK-I8-SAME: merlin-gemmini-legalize-for-llvm-export
// CHECK-I8-LABEL: util.func public @main_named
// CHECK-I8: compilation_info = #iree_codegen.compilation_info
// CHECK-I8-SAME: merlin-gemmini-legalize-for-llvm-export

// CHECK-FP8-LABEL: util.func public @main
// CHECK-FP8: compilation_info = #iree_codegen.compilation_info
// CHECK-FP8-SAME: merlin-gemmini-legalize-for-llvm-export
// CHECK-FP8-LABEL: util.func public @main_named
// CHECK-FP8: compilation_info = #iree_codegen.compilation_info
// CHECK-FP8-SAME: merlin-gemmini-legalize-for-llvm-export
// CHECK-FP8-LABEL: util.func public @main_fp8
// CHECK-FP8: compilation_info = #iree_codegen.compilation_info
// CHECK-FP8-SAME: merlin-gemmini-legalize-for-llvm-export
