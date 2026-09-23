// C1_mlp_activation_linear2
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W2 = merlin_iface.tensor {name = "W2", role = "weight"} : tensor<64x64xi8>
  %H = merlin_iface.tensor {name = "H", role = "input"} : tensor<16x64xi8>
  %W2_res = merlin_iface.resident_pack %W2 {layout = "packed_rhs"} : (tensor<64x64xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %H, %W2_res : (tensor<16x64xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["relu"], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x64xi32>
  merlin_iface.evict %W2_res : (!merlin_iface.resident) -> ()
}
