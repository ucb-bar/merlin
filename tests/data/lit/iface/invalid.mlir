// RUN: %mlir-opt --irdl-file=%iface-irdl --split-input-file -verify-diagnostics %s
//
// NEGATIVE CONTROLS for the frozen `merlin_iface` grammar, checked by UPSTREAM mlir-opt against
// merlin/contract/merlin_iface.irdl.mlir. Nothing of ours is in the checking path: the IRDL file IS
// the contract, and the interpreter evaluating it is LLVM's.
//
// **Why `-verify-diagnostics` rather than `not ... | FileCheck`.** These cases used to be three
// separate files, each running `not %mlir-opt ... | %filecheck`. That shape passes whenever an error
// appears ANYWHERE in the output — a typo elsewhere in the module, a parse failure two lines up, or a
// diagnostic about an entirely different op all satisfy it, so the test can go on passing while the
// constraint it names stops being enforced. `-verify-diagnostics` binds each expected diagnostic to a
// LINE and to its message, and fails on any unexpected diagnostic as well as any expectation that was
// not produced. Verified by moving one expectation off its op: the run then fails twice, once for
// "unexpected error" and once for "expected error ... was not produced".
//
// Generic assembly syntax throughout: an IRDL-registered dialect has no custom parser.
//
// What this file does NOT cover is in `unchecked_by_irdl.mlir` — three constraints the generator
// documents as inexpressible in IRDL, pinned there as ACCEPTED so nobody reads this suite as
// enforcing them.

// A leaf tensor is materialized deterministically BY NAME on both sides of the L0 comparison, so an
// unnamed tensor makes the golden unmatchable.
module {
  // expected-error @+1 {{'merlin_iface.tensor' op attribute "name" is expected but not provided}}
  %0 = "merlin_iface.tensor"() {role = "weight"} : () -> tensor<4x4xi8>
}

// -----

// `role` partitions leaves into inputs and weights, which is what decides who supplies the bytes.
module {
  // expected-error @+1 {{expected base attribute 'builtin.string' but got 'builtin.integer'}}
  %0 = "merlin_iface.tensor"() {name = "A", role = 3 : i64} : () -> tensor<4x4xi32>
}

// -----

// TYPE ARITY. `!merlin_iface.acc` carries the accumulator's element type; the bare spelling drops it.
// This constraint caught a real mistake: the positive test in this directory was first written with a
// bare `!merlin_iface.acc`. An accumulator whose width is unstated is exactly the ambiguity that makes
// a readout rule unstatable — see the 2026-09-05 readout entries in the devblog for what that costs.
module {
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi8>
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  // expected-error @+1 {{expected 1 type arguments, but had 0}}
  %acc = "merlin_iface.matmul"(%A, %Wr) : (tensor<4x4xi8>, !merlin_iface.resident) -> !merlin_iface.acc
}

// -----

// OPERAND TYPES, not just attribute presence. `evict` releases a resident handle; handing it the raw
// tensor means the residency lifetime was never established — the confusion behind "packed twice" and
// "evicted while still live", which otherwise surfaces as a wrong number many minutes later at the RTL
// tier. Caught here in milliseconds.
module {
  %0 = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  // expected-error @+1 {{expected base type 'merlin_iface.resident' but got 'builtin.tensor'}}
  "merlin_iface.evict"(%0) : (tensor<4x4xi8>) -> ()
}

// -----

// The same residency invariant from the other end: a matmul whose weight operand is a raw tensor never
// packed it. On a real target the packed layout is not the tensor's layout, so this compiles to a
// contraction over bytes in the wrong order — a wrong answer, not a crash.
module {
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi8>
  // expected-error @+1 {{expected base type 'merlin_iface.resident' but got 'builtin.tensor'}}
  %acc = "merlin_iface.matmul"(%A, %W) : (tensor<4x4xi8>, tensor<4x4xi8>) -> !merlin_iface.acc<i32>
}

// -----

// `layout` is what makes a resident handle mean anything; a pack with no declared layout is a promise
// with no content.
module {
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  // expected-error @+1 {{'merlin_iface.resident_pack' op attribute "layout" is expected but not provided}}
  %Wr = "merlin_iface.resident_pack"(%W) : (tensor<4x4xi8>) -> !merlin_iface.resident
}

// -----

// A commit reads the ACCUMULATOR, not a tensor. Committing a tensor skips the readout entirely, which
// is where narrowing and the epilogue live.
module {
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi32>
  // expected-error @+1 {{expected base type 'merlin_iface.acc' but got 'builtin.tensor'}}
  %Y = "merlin_iface.commit"(%A) {name = "Y", epilogue = [], output_dtype = "i32"} : (tensor<4x4xi32>) -> tensor<4x4xi32>
}

// -----

// `output_dtype` decides the readout width. Its ABSENCE is legal in the JSON schema and means "do not
// narrow"; its absence HERE is not, because the interface program is where a capsule states its intent.
module {
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi8>
  %acc = "merlin_iface.matmul"(%A, %Wr) : (tensor<4x4xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  // expected-error @+1 {{'merlin_iface.commit' op attribute "output_dtype" is expected but not provided}}
  %Y = "merlin_iface.commit"(%acc) {name = "Y", epilogue = []} : (!merlin_iface.acc<i32>) -> tensor<4x4xi32>
}

// -----

// ...and it is a dtype TOKEN, not a width. `32` and `"i32"` are not the same statement: one of them
// says nothing about signedness.
module {
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi8>
  %acc = "merlin_iface.matmul"(%A, %Wr) : (tensor<4x4xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  // expected-error @+1 {{expected base attribute 'builtin.string' but got 'builtin.integer'}}
  %Y = "merlin_iface.commit"(%acc) {name = "Y", epilogue = [], output_dtype = 32 : i64} : (!merlin_iface.acc<i32>) -> tensor<4x4xi32>
}

// -----

// The epilogue is an ordered LIST of stages. A bare string is one stage spelled as though it were the
// whole list, and the two disagree about ordering the moment a second stage is added.
module {
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi32>
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  %acc = "merlin_iface.matmul"(%A, %Wr) : (tensor<4x4xi32>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  // expected-error @+1 {{expected base attribute 'builtin.array' but got 'builtin.string'}}
  %Y = "merlin_iface.commit"(%acc) {name = "Y", epilogue = "relu", output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<4x4xi32>
}
