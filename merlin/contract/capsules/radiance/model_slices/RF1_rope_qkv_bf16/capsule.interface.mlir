module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xbf16>
  %Wqkv = merlin_iface.tensor {name = "Wqkv", role = "weight"} : tensor<16x32xbf16>
  %Wqkv_res = merlin_iface.resident_pack %Wqkv {layout = "packed_rhs"} : (tensor<16x32xbf16>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %Wqkv_res : (tensor<16x16xbf16>, !merlin_iface.resident) -> !merlin_iface.acc<f32>
  %H = merlin_iface.commit %acc0 {name = "H", epilogue = [], output_dtype = "f32"} : (!merlin_iface.acc<f32>) -> tensor<16x32xf32>
  %Y0 = merlin_iface.rope %H {name = "Y0", theta = 10000.0 : f64, output_dtype = "f32"} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  merlin_iface.evict %Wqkv_res : (!merlin_iface.resident) -> ()
}
