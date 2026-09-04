module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x32xf6E3M2FN>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<32x32xf6E3M2FN>
  %A0_scale = merlin_iface.tensor {name = "A0_scale", role = "scale", scale_of = "A0", block = 32 : i64} : tensor<1x32xi8>
  %W_scale = merlin_iface.tensor {name = "W_scale", role = "scale", scale_of = "W", block = 32 : i64} : tensor<1x32xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x32xf6E3M2FN>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<32x32xf6E3M2FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "bf16"} : (!merlin_iface.acc<bf16>) -> tensor<32x32xbf16>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
