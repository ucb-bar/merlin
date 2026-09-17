// GM1: deep-K int8 contraction sized past the operand store (100.2%) -- the spills regime.
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<8208x16xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x8208xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<8208x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<16x8208xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
