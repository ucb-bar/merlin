// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(merlin-gemmini-legalize-for-llvm-export{loop-ws=true}))' | FileCheck %s

// Phase 8: gemmini.tile_matmul should lower to a single LOOP_WS sequence
// when the pass option `loop-ws=true` is set. The sequence is:
//   gemmini.intr.config           (ConfigEx — WS dataflow)
//   gemmini.intr.config           (ConfigSt — output stride)
//   gemmini.intr.config x3        (ConfigLd — A / B / D strides+scales)
//   gemmini.intr.loop_ws.config_bounds
//   gemmini.intr.loop_ws.config_addrs_ab
//   gemmini.intr.loop_ws.config_addrs_dc
//   gemmini.intr.loop_ws.config_strides_ab
//   gemmini.intr.loop_ws.config_strides_dc
//   gemmini.intr.loop_ws
//   gemmini.intr.flush
// Total: 11 commands per matmul + final flush vs ~56 for the per-tile
// MVIN/PRELOAD/COMPUTE/MVOUT expansion. Reference kernel:
// chipyard/.../gemmini-rocc-tests/include/gemmini.h:390-398
// (`gemmini_loop_ws` macro) and the `tiled_matmul_loop_ws` macro it
// expands into.
//
// The IntrOps already exist (Phase 1, GemminiIntrinsicOps.td:47-52) and
// each maps to its funct code in LowerIntrToMmio.cpp::functForIntr
// (LOOP_WS=8, LOOP_WS_CONFIG_BOUNDS=9, _ADDRS_AB=10, _ADDRS_DC=11,
// _STRIDES_AB=12, _STRIDES_DC=13).

// CHECK-LABEL: func.func @tile_matmul_lowers_to_loop_ws
func.func @tile_matmul_lowers_to_loop_ws(
    %a: memref<16x16xi8>, %b: memref<16x64xi8>,
    %c: memref<16x64xi32>, %d: memref<0x0xi32>) {
  // Five config commands (ConfigEx + ConfigSt + 3 × ConfigLd) all lower
  // to gemmini.intr.config — total 5 of those.
  // CHECK-COUNT-5: gemmini.intr.config

  // Six LOOP_WS_* commands (one of each).
  // CHECK: gemmini.intr.loop_ws.config_bounds
  // CHECK: gemmini.intr.loop_ws.config_addrs_ab
  // CHECK: gemmini.intr.loop_ws.config_addrs_dc
  // CHECK: gemmini.intr.loop_ws.config_strides_ab
  // CHECK: gemmini.intr.loop_ws.config_strides_dc
  // CHECK: gemmini.intr.loop_ws

  // Single closing FLUSH.
  // CHECK: gemmini.intr.flush

  // No per-tile MVIN/PRELOAD/COMPUTE/MVOUT expansion in LOOP_WS mode.
  // CHECK-NOT: gemmini.intr.mvin
  // CHECK-NOT: gemmini.intr.mvout
  // CHECK-NOT: gemmini.intr.preload
  // CHECK-NOT: gemmini.intr.compute
  gemmini.tile_matmul %a, %b, %c, %d {
    aScaleFactor = 1.0 : f32, bScaleFactor = 1.0 : f32,
    dScaleFactor = 1.0 : f32, act = 0 : i64,
    accScale = 1.0 : f32, bertScale = 0.0 : f32,
    dataflow = 1 : i64
  } : memref<16x16xi8>, memref<16x64xi8>, memref<16x64xi32>, memref<0x0xi32>
  return
}
