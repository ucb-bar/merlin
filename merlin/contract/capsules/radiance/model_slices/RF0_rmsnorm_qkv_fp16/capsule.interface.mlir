module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xf16>
  %G = merlin_iface.tensor {name = "G", role = "weight"} : tensor<1x16xf16>
  %Wqkv = merlin_iface.tensor {name = "Wqkv", role = "weight"} : tensor<16x48xf16>
  %H = merlin_iface.rmsnorm %X, %G {name = "H", eps = 1.000000000e-05 : f64, output_dtype = "f32"} : (tensor<16x16xf16>, tensor<1x16xf16>) -> tensor<16x16xf32>
  %Wqkv_res = merlin_iface.resident_pack %Wqkv {layout = "packed_rhs"} : (tensor<16x48xf16>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %H, %Wqkv_res : (tensor<16x16xf32>, !merlin_iface.resident) -> !merlin_iface.acc<f32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "f32"} : (!merlin_iface.acc<f32>) -> tensor<16x48xf32>
  merlin_iface.evict %Wqkv_res : (!merlin_iface.resident) -> ()
}
