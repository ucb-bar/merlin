module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<31x15xi8>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x31xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<31x15xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<16x31xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["maxpool"], output_dtype = "i8", pool_in_dims = [4, 4], pool_size = [2, 2], pool_stride = [2, 2], pool_padding = [0, 0, 0, 0]} : (!merlin_iface.acc<i32>) -> tensor<4x15xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
