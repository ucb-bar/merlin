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
  %A8 = merlin_iface.tensor {name = "A8", role = "input"} : tensor<16x16xi8>
  %A9 = merlin_iface.tensor {name = "A9", role = "input"} : tensor<16x16xi8>
  %A10 = merlin_iface.tensor {name = "A10", role = "input"} : tensor<16x16xi8>
  %A11 = merlin_iface.tensor {name = "A11", role = "input"} : tensor<16x16xi8>
  %A12 = merlin_iface.tensor {name = "A12", role = "input"} : tensor<16x16xi8>
  %A13 = merlin_iface.tensor {name = "A13", role = "input"} : tensor<16x16xi8>
  %A14 = merlin_iface.tensor {name = "A14", role = "input"} : tensor<16x16xi8>
  %A15 = merlin_iface.tensor {name = "A15", role = "input"} : tensor<16x16xi8>
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
  %acc8 = merlin_iface.matmul %A8, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y8 = merlin_iface.commit %acc8 {name = "Y8", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc9 = merlin_iface.matmul %A9, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y9 = merlin_iface.commit %acc9 {name = "Y9", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc10 = merlin_iface.matmul %A10, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y10 = merlin_iface.commit %acc10 {name = "Y10", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc11 = merlin_iface.matmul %A11, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y11 = merlin_iface.commit %acc11 {name = "Y11", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc12 = merlin_iface.matmul %A12, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y12 = merlin_iface.commit %acc12 {name = "Y12", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc13 = merlin_iface.matmul %A13, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y13 = merlin_iface.commit %acc13 {name = "Y13", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc14 = merlin_iface.matmul %A14, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y14 = merlin_iface.commit %acc14 {name = "Y14", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  %acc15 = merlin_iface.matmul %A15, %W_res : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y15 = merlin_iface.commit %acc15 {name = "Y15", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
