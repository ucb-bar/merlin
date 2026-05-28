// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-lower-intr-to-mmio{mmio-base=0x40084000}))' | FileCheck %s

// gemmini-lower-intr-to-mmio replaces every gemmini.intr.<op>(rs1, rs2)
// with three volatile llvm.store ops at mmioBase + 0x10/0x18/0x00.
// This is the path used by chipyard `RadianceGemminiOnlyConfig`, where
// the small Rocket core has no RoCC and Gemmini sits as an MMIO
// peripheral. Reference kernel:
// chipyard/.../gemmini-rocc-tests/bareMetalC/matmul_ws_mx_generic.c:84
// for the encoded instruction word.
//
// Each gemmini.intr.* op has a known funct (gemmini.h:31-67):
//   CONFIG=0, MVIN2=1, MVIN=2, MVOUT=3, COMPUTE_PRELOADED=4,
//   COMPUTE_ACCUMULATE=5, PRELOAD=6, FLUSH=7, ...
// The encoded instruction word is
//   0x7B | (3<<12) | (1<<15) | (2<<20) | (funct<<25)
// For mmioBase = 0x40084000 (default; matches matmul_ws_mx_generic.c:28):
//   rs1 addr  = 0x40084010 = 1074282512
//   rs2 addr  = 0x40084018 = 1074282520
//   inst addr = 0x40084000 = 1074282496
// For MVIN  (funct=2): inst word = 0x0420B07B = 69251195
// For FLUSH (funct=7): inst word = 0x0E20B07B = 236958331

// CHECK-LABEL: func.func @mvin_lowers_to_three_volatile_stores
func.func @mvin_lowers_to_three_volatile_stores(%rs1: i64, %rs2: i64) {
  // CHECK-DAG: llvm.mlir.constant(1074282512 : i64)
  // CHECK-DAG: llvm.mlir.constant(1074282520 : i64)
  // CHECK-DAG: llvm.mlir.constant(1074282496 : i64)
  // CHECK-DAG: llvm.mlir.constant(69251195 : i32)
  // CHECK-COUNT-3: llvm.store volatile
  // CHECK-NOT: gemmini.intr.mvin
  "gemmini.intr.mvin"(%rs1, %rs2) : (i64, i64) -> ()
  return
}

// CHECK-LABEL: func.func @flush_and_config_both_lower
func.func @flush_and_config_both_lower(%rs1: i64, %rs2: i64) {
  // CONFIG → 3 volatile stores. FLUSH → 3 volatile stores + a busy-wait
  // poll loop on (mmioBase + 0x20). After the rewrite no gemmini.intr.*
  // ops remain.
  // CHECK-COUNT-6: llvm.store volatile
  // CHECK: llvm.load volatile
  // CHECK-NOT: gemmini.intr
  "gemmini.intr.config"(%rs1, %rs2) : (i64, i64) -> ()
  "gemmini.intr.flush"(%rs1, %rs2) : (i64, i64) -> ()
  return
}

// Phase 8: every LOOP_WS ISA-tier op must lower to the MMIO triple-store
// just like MVIN/PRELOAD/COMPUTE. If the name dispatch in functForIntr
// misses any of these, the IntrOp survives into the LLVM-IR translation
// interface and becomes a custom-3 RoCC instruction — which the small
// Rocket core in RadianceGemminiOnlyConfig has no port to consume,
// silently dropping the entire LOOP_WS sequence.
// CHECK-LABEL: func.func @loop_ws_family_lowers
func.func @loop_ws_family_lowers(%rs1: i64, %rs2: i64) {
  // 6 LOOP_WS ops → 18 volatile stores.
  // CHECK-COUNT-18: llvm.store volatile
  // CHECK-NOT: gemmini.intr.loop_ws
  "gemmini.intr.loop_ws.config_bounds"(%rs1, %rs2) : (i64, i64) -> ()
  "gemmini.intr.loop_ws.config_addrs_ab"(%rs1, %rs2) : (i64, i64) -> ()
  "gemmini.intr.loop_ws.config_addrs_dc"(%rs1, %rs2) : (i64, i64) -> ()
  "gemmini.intr.loop_ws.config_strides_ab"(%rs1, %rs2) : (i64, i64) -> ()
  "gemmini.intr.loop_ws.config_strides_dc"(%rs1, %rs2) : (i64, i64) -> ()
  "gemmini.intr.loop_ws"(%rs1, %rs2) : (i64, i64) -> ()
  return
}
