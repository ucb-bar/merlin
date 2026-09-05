
// RUN: not %mlir-opt --irdl-file=%iface-irdl %s 2>&1 | %filecheck %s
//
// NEGATIVE CONTROL. `merlin_iface.tensor` requires a `name` attribute -- leaf tensors are
// materialized deterministically BY NAME on both sides, so an unnamed tensor makes the golden
// unmatchable. The IRDL definition encodes that, and this test proves the encoding bites.
// If this test starts passing the module, the frozen grammar has stopped being enforced.

// CHECK: error
// CHECK-SAME: name
module {
  %0 = "merlin_iface.tensor"() {role = "weight"} : () -> tensor<4x4xi8>
}

