// K_attn_pv_64x64x64 (PV (seq64, head64))
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %V = merlin_iface.tensor {name = "V", role = "weight"} : tensor<64x64xi8>
  %P = merlin_iface.tensor {name = "P", role = "input"} : tensor<64x64xi8>
  %V_res = merlin_iface.resident_pack %V {layout = "packed_rhs"} : (tensor<64x64xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %P, %V_res : (tensor<64x64xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<64x64xi32>
  merlin_iface.evict %V_res : (!merlin_iface.resident) -> ()
}
