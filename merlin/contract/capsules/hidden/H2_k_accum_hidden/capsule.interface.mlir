// H2 hidden k-accum
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Wk = merlin_iface.tensor {name = "Wk", role = "weight"} : tensor<32x16xi8>
  %Ak = merlin_iface.tensor {name = "Ak", role = "input"} : tensor<16x32xi8>
  %Wk_res = merlin_iface.resident_pack %Wk {layout = "packed_rhs"} : (tensor<32x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %Ak, %Wk_res : (tensor<16x32xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %Wk_res : (!merlin_iface.resident) -> ()
}
