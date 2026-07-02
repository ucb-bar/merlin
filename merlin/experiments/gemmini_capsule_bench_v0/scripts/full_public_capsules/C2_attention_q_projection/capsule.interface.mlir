// C2_attention_q_projection
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wq = merlin_iface.tensor {name = "Wq", role = "weight"} : tensor<64x16xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x64xi8>
  %Wq_res = merlin_iface.resident_pack %Wq {layout = "packed_rhs"} : (tensor<64x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %Wq_res : (tensor<16x64xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %Wq_res : (!merlin_iface.resident) -> ()
}
