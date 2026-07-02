// H0 hidden matmul
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wh = merlin_iface.tensor {name = "Wh", role = "weight"} : tensor<16x16xi8>
  %Ah = merlin_iface.tensor {name = "Ah", role = "input"} : tensor<16x16xi8>
  %Wh_res = merlin_iface.resident_pack %Wh {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %Ah, %Wh_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %Wh_res : (!merlin_iface.resident) -> ()
}
