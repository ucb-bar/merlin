// RUN: not %mlir-opt --irdl-file=%iface-irdl %s 2>&1 | %filecheck %s
//
// NEGATIVE CONTROL for type ARITY.
//
// `!merlin_iface.acc` carries the accumulator's element type; the bare spelling drops it. This test
// exists because the constraint caught a real mistake: the positive test in this directory was
// written with a bare `!merlin_iface.acc` and the IRDL rejected it. That is the layer working — an
// accumulator whose width is unstated is exactly the ambiguity the frozen grammar pins down.

// CHECK: error
// CHECK-SAME: type arguments
module {
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi8>
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  %acc = "merlin_iface.matmul"(%A, %Wr) : (tensor<4x4xi8>, !merlin_iface.resident) -> !merlin_iface.acc
}
