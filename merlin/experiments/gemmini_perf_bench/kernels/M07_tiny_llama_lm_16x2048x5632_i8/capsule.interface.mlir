// M07_tiny_llama_lm_16x2048x5632_i8 (harvested 8x2048x5632)
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<2048x5632xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x2048xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<2048x5632xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<16x2048xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["acc_scale"], output_dtype = "i8", acc_scale = 0.0625 : f32} : (!merlin_iface.acc<i32>) -> tensor<16x5632xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
