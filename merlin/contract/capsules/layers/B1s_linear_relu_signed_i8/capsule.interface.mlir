// B1s: linear + relu over a SIGNED stimulus, so the activation actually binds.
//
// B1_linear_relu_i8 declares the same fused relu and cannot detect whether it happened: with the
// default 0..3 stimulus every operand is non-negative, so the accumulator is never negative and
// max(0,x) is the identity on every value that program can produce. This capsule declares
// stimulus_range [-3, 3], which makes roughly half the accumulator entries negative -- so a run
// that drops the activation produces a different answer and fails.
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x16xi8>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x32xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x16xi8>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<16x32xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = ["relu"], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
