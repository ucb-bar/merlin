// G08_large_sq_128x128x128 (larger square (util headroom))
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<128x128xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<128x128xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<128x128xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<128x128xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<128x128xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
