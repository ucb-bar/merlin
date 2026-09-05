module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x64xf8E4M3FN>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<64x32xf8E4M3FN>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x64xf8E4M3FN>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<64x32xf8E4M3FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "bf16"} : (!merlin_iface.acc<bf16>) -> tensor<64x64xbf16>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
