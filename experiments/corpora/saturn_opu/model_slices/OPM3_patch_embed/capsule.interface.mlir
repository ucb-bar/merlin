module attributes {merlin_iface.version = "0.1", merlin_iface.target = "saturn_opu_mxv256d128", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<768x196xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<256x768xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<768x196xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<256x768xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<256x196xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
