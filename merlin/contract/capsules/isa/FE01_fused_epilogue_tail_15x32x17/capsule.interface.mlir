module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x17xi8>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<15x32xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x17xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<15x32xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["acc_scale", "relu"], output_dtype = "i8", acc_scale = 0.0625 : f32} : (!merlin_iface.acc<i32>) -> tensor<15x17xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
