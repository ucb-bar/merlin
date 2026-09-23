// AF13_matmul_bf16_pt ports npu_model's `smolvla_matmul`: one 32x32 tile contraction C = A @ B. Execution reference: atlas-npu `baremetal/assembly/smolvla_matmul_mxu0.S` (and `_mxu1.S` for the second mesh). VTRPOSE: that assembly transposes B (`VTRPOSE.XLU 5, 4`) before `VMATPUSH.W.MXU0` because the MXU loads weights in transposed orientation; npu_model's own assembly has no VTRPOSE at all, so the program it ships computes A @ B^T on real hardware. This capsule's contract is A @ B. DTYPE -- option (b), the promoted form: npu_model builds both operands with `.to(torch.float8_e4m3fn)` and the MXU multiplies fp8_e4m3 into a bf16 accumulator, while this capsule is bf16 on both operands. A host torch-eager golden can only be authored in a float torch dtype, and the fp8 form of this same contraction is already carried by AT7/AT8 (one per mesh) and AS0_matmul_spec (specir refmodel); what this entry adds is the pytorch-sourced 32x32 shape.
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x32xbf16>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<32x32xbf16>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x32xbf16>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<32x32xbf16>, !merlin_iface.resident) -> !merlin_iface.acc<f32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "f32"} : (!merlin_iface.acc<f32>) -> tensor<32x32xf32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
