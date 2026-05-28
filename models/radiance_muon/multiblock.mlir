// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 2.10 demo: multi-threadblock kernel using %tbid.
//
// Each threadblock owns a contiguous slice of the output:
//   slice_size = n / num_blocks
//   slice_base = tbid * slice_size
//   for i in tid..slice_size step tpt:
//     dst[slice_base + i] = src[slice_base + i] * 2.0
//
// Confirms %tbid is correctly threaded through the LLVM IR and
// usable in computation (i.e. not just dropped). The mu_schedule
// runtime currently launches a single block per warp, but the kernel
// is multi-block-correct: a future scheduler that issues N blocks
// will have each one process its own slice.

module {
  func.func @radiance_multiblock_body(
      %src:    memref<?xf32, #radiance.addrspace<global>>,
      %dst:    memref<?xf32, #radiance.addrspace<global>>,
      %slice:  i32,
      %tid:    i32,
      %tpt:    i32,
      %tbid:   i32)
      attributes {
        "radiance.kernel",
        "radiance.num_warps" = 4 : i32,
        "radiance.entry_symbol" = "radiance_multiblock_body"
      } {
    %tid_idx   = arith.index_cast %tid   : i32 to index
    %tpt_idx   = arith.index_cast %tpt   : i32 to index
    %slice_idx = arith.index_cast %slice : i32 to index
    %tbid_idx  = arith.index_cast %tbid  : i32 to index
    %base      = arith.muli %tbid_idx, %slice_idx : index
    %two_f     = arith.constant 2.0 : f32
    scf.for %i = %tid_idx to %slice_idx step %tpt_idx {
      %off = arith.addi %base, %i : index
      %v   = memref.load %src[%off] : memref<?xf32, #radiance.addrspace<global>>
      %r   = arith.mulf %v, %two_f : f32
      memref.store %r, %dst[%off] : memref<?xf32, #radiance.addrspace<global>>
    }
    return
  }
}
