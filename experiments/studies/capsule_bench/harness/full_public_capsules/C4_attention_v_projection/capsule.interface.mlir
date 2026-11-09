// C4_attention_v_projection
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wv = merlin_iface.tensor {name = "Wv", role = "weight"} : tensor<64x16xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x64xi8>
  %Wv_res = merlin_iface.resident_pack %Wv {layout = "packed_rhs"} : (tensor<64x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %Wv_res : (tensor<16x64xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %Wv_res : (!merlin_iface.resident) -> ()
}
