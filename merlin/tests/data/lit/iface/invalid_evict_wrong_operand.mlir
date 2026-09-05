// RUN: not %mlir-opt --irdl-file=%iface-irdl %s 2>&1 | %filecheck %s
//
// NEGATIVE CONTROL for operand TYPES, not just attribute presence.
//
// `merlin_iface.evict` releases a resident handle. Handing it the raw tensor instead means the
// residency lifetime was never established -- the exact confusion behind "packed twice" and
// "evicted while still live" bugs, which otherwise surface as a wrong number many minutes later at
// the RTL tier. The IRDL definition of the frozen grammar catches it here, in milliseconds.

// CHECK: error
// CHECK-SAME: expected base type
// CHECK-SAME: merlin_iface.resident
module {
  %0 = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  "merlin_iface.evict"(%0) : (tensor<4x4xi8>) -> ()
}
