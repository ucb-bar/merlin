module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wh2 = merlin_iface.tensor {name = "Wh2", role = "weight"} : tensor<16x16xi8>
  %Ah2 = merlin_iface.tensor {name = "Ah2", role = "input"} : tensor<16x16xi8>
  %W_res = merlin_iface.resident_pack %Wh2 {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %Ah2, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Yh2 = merlin_iface.commit %acc0 {name = "Yh2", epilogue = ["acc_scale"], output_dtype = "i8", acc_scale = 0.0625 : f32} : (!merlin_iface.acc<i32>) -> tensor<16x16xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
