module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<1040x16xf32>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x1040xf32>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<1040x16xf32>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<16x1040xf32>, !merlin_iface.resident) -> !merlin_iface.acc<f32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "f32"} : (!merlin_iface.acc<f32>) -> tensor<16x16xf32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
