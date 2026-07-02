// H1 hidden acc_scale i8
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wq = merlin_iface.tensor {name = "Wq", role = "weight"} : tensor<16x16xi8>
  %Aq = merlin_iface.tensor {name = "Aq", role = "input"} : tensor<16x16xi8>
  %Wq_res = merlin_iface.resident_pack %Wq {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %Aq, %Wq_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["acc_scale"], output_dtype = "i8", acc_scale = 0.0625 : f32} : (!merlin_iface.acc<i32>) -> tensor<16x16xi8>
  merlin_iface.evict %Wq_res : (!merlin_iface.resident) -> ()
}
