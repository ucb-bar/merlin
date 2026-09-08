module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %W0 = merlin_iface.tensor {name = "W0", role = "weight"} : tensor<32x8xf8E4M3FN>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<4x32xf8E4M3FN>
  %W1 = merlin_iface.tensor {name = "W1", role = "weight"} : tensor<8x5xbf16>
  %W0_res = merlin_iface.resident_pack %W0 {layout = "packed_rhs"} : (tensor<32x8xf8E4M3FN>) -> !merlin_iface.resident
  %W1_res = merlin_iface.resident_pack %W1 {layout = "packed_rhs"} : (tensor<8x5xbf16>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W0_res : (tensor<4x32xf8E4M3FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "bf16"} : (!merlin_iface.acc<bf16>) -> tensor<4x8xbf16>
  %acc1 = merlin_iface.matmul %Y0, %W1_res : (tensor<4x8xbf16>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y1 = merlin_iface.commit %acc1 {name = "Y1", epilogue = [], output_dtype = "bf16"} : (!merlin_iface.acc<bf16>) -> tensor<4x5xbf16>
  merlin_iface.evict %W0_res : (!merlin_iface.resident) -> ()
  merlin_iface.evict %W1_res : (!merlin_iface.resident) -> ()
}
