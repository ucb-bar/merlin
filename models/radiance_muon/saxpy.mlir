// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 2.6 second-kernel reference: saxpy.
//
// dst[i] = dst[i] + src[i] * factor
//
// Differs from vecadd by introducing a scalar f32 parameter (`factor`)
// in addition to memref globals — exercises arith.mulf and the
// scalar-arg path through the lowering.

module {
  func.func @radiance_saxpy_body(
      %src: memref<?xf32, #radiance.addrspace<global>>,
      %dst: memref<?xf32, #radiance.addrspace<global>>,
      %factor: f32,
      %n: i32,
      %tid: i32,
      %tpt: i32,
      %tbid: i32)
      attributes {
        "radiance.kernel",
        "radiance.num_warps" = 4 : i32,
        "radiance.entry_symbol" = "radiance_saxpy_body"
      } {
    %tid_idx = arith.index_cast %tid : i32 to index
    %tpt_idx = arith.index_cast %tpt : i32 to index
    %n_idx = arith.index_cast %n : i32 to index
    scf.for %i = %tid_idx to %n_idx step %tpt_idx {
      %s = memref.load %src[%i] : memref<?xf32, #radiance.addrspace<global>>
      %d = memref.load %dst[%i] : memref<?xf32, #radiance.addrspace<global>>
      %sf = arith.mulf %s, %factor : f32
      %r = arith.addf %d, %sf : f32
      memref.store %r, %dst[%i] : memref<?xf32, #radiance.addrspace<global>>
    }
    return
  }
}
