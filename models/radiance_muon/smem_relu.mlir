// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 2.9 demo: kernel that uses __shared scratchpad memory.
//
// Each warp loads a tile of `src` into shared memory, applies relu,
// and writes the tiled result to `dst`. Exercises the
// #radiance.addrspace<shared> path of ConvertRadianceAddrSpaces and
// confirms the lowered LLVM IR routes shared accesses to addrspace(1).

module {
  func.func @radiance_smem_relu_body(
      %src:    memref<?xf32, #radiance.addrspace<global>>,
      %dst:    memref<?xf32, #radiance.addrspace<global>>,
      %smem:   memref<64xf32, #radiance.addrspace<shared>>,
      %n:      i32,
      %tid:    i32,
      %tpt:    i32,
      %tbid:   i32)
      attributes {
        "radiance.kernel",
        "radiance.num_warps" = 4 : i32,
        "radiance.entry_symbol" = "radiance_smem_relu_body"
      } {
    %tid_idx = arith.index_cast %tid : i32 to index
    %tpt_idx = arith.index_cast %tpt : i32 to index
    %n_idx   = arith.index_cast %n   : i32 to index
    %zero_f  = arith.constant 0.0 : f32
    scf.for %i = %tid_idx to %n_idx step %tpt_idx {
      %v = memref.load %src[%i] : memref<?xf32, #radiance.addrspace<global>>
      // Stage in shared memory at slot (tid mod 64)
      %c64 = arith.constant 64 : index
      %slot = arith.remui %tid_idx, %c64 : index
      memref.store %v, %smem[%slot] : memref<64xf32, #radiance.addrspace<shared>>
      %sv = memref.load %smem[%slot] : memref<64xf32, #radiance.addrspace<shared>>
      %ge = arith.cmpf ogt, %sv, %zero_f : f32
      %r = arith.select %ge, %sv, %zero_f : f32
      memref.store %r, %dst[%i] : memref<?xf32, #radiance.addrspace<global>>
    }
    return
  }
}
