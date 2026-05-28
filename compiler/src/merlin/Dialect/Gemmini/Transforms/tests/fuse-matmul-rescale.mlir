// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-lower-tile-to-isa))' | FileCheck %s

// Quantized matmul + bias-add + rescale (i32→i8) fold into a single
// gemmini.tile_matmul with fullC=false + accScale=in/out. Mirrors the
// post-bufferization shape that dronet's conv→matmul→rescale dispatches
// emit. See LowerTileToISA.cpp::tryMatchI32ToI8Rescale.

#desc = #hal.descriptor_type<storage_buffer>

// CHECK-LABEL: func.func @fuse_matmul_biasadd_rescale
func.func @fuse_matmul_biasadd_rescale(
    %A: memref<16x32xi8, strided<[32, 1], offset: ?>, #desc>,
    %B: memref<32x16xi8, #desc>,
    %bias: memref<16x16xi32, #desc>,
    %out: memref<16x16xi8, #desc>) {
  %c0_i32 = arith.constant 0 : i32
  %cst_in = arith.constant 6.25015527E-5 : f32
  %cst_out = arith.constant 2.5029665E-2 : f32
  %cst_zp = arith.constant 0.0 : f32
  %cst_lo = arith.constant -1.280000e+02 : f32
  %cst_hi = arith.constant 1.270000e+02 : f32
  %acc = memref.alloca() : memref<16x16xi32>
  %biased = memref.alloca() : memref<16x16xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%acc : memref<16x16xi32>)
  linalg.matmul ins(%A, %B : memref<16x32xi8, strided<[32, 1], offset: ?>, #desc>, memref<32x16xi8, #desc>) outs(%acc : memref<16x16xi32>)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%acc, %bias : memref<16x16xi32>, memref<16x16xi32, #desc>)
    outs(%biased : memref<16x16xi32>) {
  ^bb0(%in: i32, %in_b: i32, %o: i32):
    %s = arith.addi %in, %in_b : i32
    linalg.yield %s : i32
  }
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%biased : memref<16x16xi32>)
    outs(%out : memref<16x16xi8, #desc>) {
  ^bb0(%in: i32, %o: i8):
    %a = arith.sitofp %in : i32 to f32
    %b = arith.mulf %a, %cst_in : f32
    %c = arith.divf %b, %cst_out : f32
    %d = math.roundeven %c : f32
    %e = arith.addf %d, %cst_zp : f32
    %f = arith.maximumf %e, %cst_lo : f32
    %g = arith.minimumf %f, %cst_hi : f32
    %h = arith.fptosi %g : f32 to i8
    linalg.yield %h : i8
  }
  return
}

// CHECK: gemmini.tile_matmul
// CHECK-SAME: accScale = 0.00249709
// CHECK-SAME: act = 0 : i64
// fullC=false is the default (TableGen DefaultValuedAttr) and elided.
// CHECK-NOT: fullC = true
// The bias buffer is forwarded as the tile_matmul's dArray operand.
// CHECK-SAME: memref<16x16xi8
// CHECK-SAME: memref<16x16xi32

// -----

// Same shape but the rescale body has the canonical ReLU tail:
// fptosi → cmpi sgt 0 → select → sitofp → roundeven → addf 0 → max(-128) →
// min(127) → fptosi. Expect act=1 (RELU) on the fused tile_matmul.

#desc2 = #hal.descriptor_type<storage_buffer>

// CHECK-LABEL: func.func @fuse_matmul_biasadd_rescale_relu
func.func @fuse_matmul_biasadd_rescale_relu(
    %A: memref<16x32xi8, strided<[32, 1], offset: ?>, #desc2>,
    %B: memref<32x16xi8, #desc2>,
    %bias: memref<16x16xi32, #desc2>,
    %out: memref<16x16xi8, #desc2>) {
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8
  %cst_in = arith.constant 6.25015527E-5 : f32
  %cst_out = arith.constant 3.6599629E-2 : f32
  %cst_zp = arith.constant 0.0 : f32
  %cst_lo = arith.constant -1.280000e+02 : f32
  %cst_hi = arith.constant 1.270000e+02 : f32
  %acc = memref.alloca() : memref<16x16xi32>
  %biased = memref.alloca() : memref<16x16xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%acc : memref<16x16xi32>)
  linalg.matmul ins(%A, %B : memref<16x32xi8, strided<[32, 1], offset: ?>, #desc2>, memref<32x16xi8, #desc2>) outs(%acc : memref<16x16xi32>)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%acc, %bias : memref<16x16xi32>, memref<16x16xi32, #desc2>)
    outs(%biased : memref<16x16xi32>) {
  ^bb0(%in: i32, %in_b: i32, %o: i32):
    %s = arith.addi %in, %in_b : i32
    linalg.yield %s : i32
  }
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%biased : memref<16x16xi32>)
    outs(%out : memref<16x16xi8, #desc2>) {
  ^bb0(%in: i32, %o: i8):
    %a = arith.sitofp %in : i32 to f32
    %b = arith.mulf %a, %cst_in : f32
    %c = arith.divf %b, %cst_out : f32
    %d = math.roundeven %c : f32
    %e = arith.addf %d, %cst_zp : f32
    %f = arith.maximumf %e, %cst_lo : f32
    %g = arith.minimumf %f, %cst_hi : f32
    %h = arith.fptosi %g : f32 to i8
    %cmp = arith.cmpi sgt, %h, %c0_i8 : i8
    %sel = arith.select %cmp, %h, %c0_i8 : i8
    %h2f = arith.sitofp %sel : i8 to f32
    %r2 = math.roundeven %h2f : f32
    %a2 = arith.addf %r2, %cst_zp : f32
    %x2 = arith.maximumf %a2, %cst_lo : f32
    %y2 = arith.minimumf %x2, %cst_hi : f32
    %z2 = arith.fptosi %y2 : f32 to i8
    linalg.yield %z2 : i8
  }
  return
}

// CHECK: gemmini.tile_matmul
// CHECK-SAME: act = 1 : i64
// CHECK-NOT: fullC = true
