
// RUN: %mlir-opt --irdl-file=%iface-irdl %s | %filecheck %s
//
// The frozen `merlin_iface` grammar, verified by UPSTREAM mlir-opt against the repo's own IRDL
// definition. Nothing of ours is in the checking path: the constraints come from
// merlin/contract/merlin_iface.irdl.mlir, which IS the contract.
//
// Generic assembly syntax is required -- an IRDL-registered dialect has no custom parser.

// CHECK: merlin_iface.resident_pack
// CHECK: merlin_iface.matmul
// CHECK: merlin_iface.commit
module attributes {merlin_iface.version = "0.1", merlin_iface.abi_version = "0.1"} {
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<32x16xi8>
  %A0 = "merlin_iface.tensor"() {name = "A0", role = "input"} : () -> tensor<16x32xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<32x16xi8>) -> !merlin_iface.resident
  %acc = "merlin_iface.matmul"(%A0, %Wr) : (tensor<16x32xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = "merlin_iface.commit"(%acc) {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
  "merlin_iface.evict"(%Wr) : (!merlin_iface.resident) -> ()
}

