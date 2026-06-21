// C0_mlp_linear1
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W1 = merlin_iface.tensor {name = "W1", role = "weight"} : tensor<64x64xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x64xi8>
  %W1_res = merlin_iface.resident_pack %W1 {layout = "packed_rhs"} : (tensor<64x64xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W1_res : (tensor<16x64xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x64xi32>
  merlin_iface.evict %W1_res : (!merlin_iface.resident) -> ()
}
