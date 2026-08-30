module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<128x32xbf16>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<32x128xbf16>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<128x32xbf16>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<32x128xbf16>, !merlin_iface.resident) -> !merlin_iface.acc<f32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "f32"} : (!merlin_iface.acc<f32>) -> tensor<32x32xf32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
