// C5_attention_qk_matmul
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Kt = merlin_iface.tensor {name = "Kt", role = "weight"} : tensor<16x16xi8>
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<16x16xi8>
  %Kt_res = merlin_iface.resident_pack %Kt {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %Q, %Kt_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %Kt_res : (!merlin_iface.resident) -> ()
}
