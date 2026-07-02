module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wh0 = merlin_iface.tensor {name = "Wh0", role = "weight"} : tensor<16x16xi8>
  %Ah0 = merlin_iface.tensor {name = "Ah0", role = "input"} : tensor<16x16xi8>
  %W_res = merlin_iface.resident_pack %Wh0 {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %Ah0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Yh0 = merlin_iface.commit %acc0 {name = "Yh0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
