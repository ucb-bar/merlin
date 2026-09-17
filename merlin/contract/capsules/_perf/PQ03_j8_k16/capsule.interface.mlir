module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>
  %A1 = merlin_iface.tensor {name = "A1", role = "input"} : tensor<16x16xi8>
  %A2 = merlin_iface.tensor {name = "A2", role = "input"} : tensor<16x16xi8>
  %A3 = merlin_iface.tensor {name = "A3", role = "input"} : tensor<16x16xi8>
  %A4 = merlin_iface.tensor {name = "A4", role = "input"} : tensor<16x16xi8>
  %A5 = merlin_iface.tensor {name = "A5", role = "input"} : tensor<16x16xi8>
  %A6 = merlin_iface.tensor {name = "A6", role = "input"} : tensor<16x16xi8>
  %A7 = merlin_iface.tensor {name = "A7", role = "input"} : tensor<16x16xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc1 = merlin_iface.matmul %A1, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y1 = merlin_iface.commit %acc1 {name = "Y1", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc2 = merlin_iface.matmul %A2, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y2 = merlin_iface.commit %acc2 {name = "Y2", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc3 = merlin_iface.matmul %A3, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y3 = merlin_iface.commit %acc3 {name = "Y3", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc4 = merlin_iface.matmul %A4, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y4 = merlin_iface.commit %acc4 {name = "Y4", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc5 = merlin_iface.matmul %A5, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y5 = merlin_iface.commit %acc5 {name = "Y5", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc6 = merlin_iface.matmul %A6, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y6 = merlin_iface.commit %acc6 {name = "Y6", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc7 = merlin_iface.matmul %A7, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y7 = merlin_iface.commit %acc7 {name = "Y7", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
