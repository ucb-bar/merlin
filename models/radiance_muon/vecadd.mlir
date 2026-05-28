// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 2.6 reference input: vecadd as a Radiance kernel in MLIR.
//
// The function is tagged with the `radiance.kernel` unit attribute and a
// `radiance.num_warps = 4` attribute. Memref operands carry the
// `#radiance.global` address-space attribute matching mu_intrinsics.h's
// `__global` qualifier.
//
// Compile pipeline (when Phase 2.6c lands):
//   merlin compile vecadd.mlir --target radiance_muon
//     ↓ iree-compile --iree-radiance-enable=true --iree-radiance-emit-llvm-ir=true
//   kernel_body.ll  (LLVM IR text, extern-C symbol radiance_vecadd_body)
//     ↓ tools/kernels/precompile.py with manifest entry source_lang=ll
//   radiance_vecadd_body.muon.o
//     ↓ link with kernel_phase2.cpp wrapper + libmuonrt.a + tohost.S
//   kernel.radiance.elf

module {
  func.func @radiance_vecadd_body(
      %A: memref<?xf32, #radiance.addrspace<global>>,
      %B: memref<?xf32, #radiance.addrspace<global>>,
      %C: memref<?xf32, #radiance.addrspace<global>>,
      %n: i32,
      %tid: i32,
      %tpt: i32,
      %tbid: i32)
      attributes {
        "radiance.kernel",
        "radiance.num_warps" = 4 : i32,
        "radiance.entry_symbol" = "radiance_vecadd_body"
      } {
    %tid_idx = arith.index_cast %tid : i32 to index
    %tpt_idx = arith.index_cast %tpt : i32 to index
    %n_idx = arith.index_cast %n : i32 to index
    scf.for %i = %tid_idx to %n_idx step %tpt_idx {
      %a = memref.load %A[%i] : memref<?xf32, #radiance.addrspace<global>>
      %b = memref.load %B[%i] : memref<?xf32, #radiance.addrspace<global>>
      %c = arith.addf %a, %b : f32
      memref.store %c, %C[%i] : memref<?xf32, #radiance.addrspace<global>>
    }
    return
  }
}
