// RUN: %mlir-opt --irdl-file=%iface-irdl --split-input-file %s | %filecheck %s
//
// THE GAP, PINNED. Every module below is MALFORMED and every one of them is ACCEPTED by the frozen
// grammar. That is not a bug in the IRDL file — it is what the generator already documents at the top
// of merlin/contract/merlin_iface.irdl.mlir, listing three ODS constraints IRDL cannot express.
//
// This file exists because a documented limitation is easy to forget and a green suite is easy to
// over-read. `invalid.mlir` proves the grammar bites; without this file, nothing distinguishes "the
// grammar rejects everything malformed" from "the grammar rejects the malformed things we happened to
// write a test for". Each case here names the layer that DOES catch it, so the gap is a routing
// decision rather than a hole.
//
// The generator's own note explains why these are absent rather than present-but-inert: mlir-opt drops
// an `irdl.c_pred` from its enclosing `irdl.all_of` WITHOUT a diagnostic, so a constraint carried that
// way can never fail. A constraint that cannot fail reads as enforcement while providing none, which is
// strictly worse than an honest omission.
//
// If a case here starts FAILING, IRDL gained expressiveness (or the generator learned to emit it) —
// delete the case and add it to `invalid.mlir`.

// (1) "the tensor element type must not be a token". `irdl.parametric` reaches the parameters of
// IRDL-declared types only, not a builtin tensor's, and IRDL has no negation. A tensor of tokens has
// no bytes, so this is caught downstream when the capsule's DRAM map sizes the tensor and
// `quant_formats` has no width for the element type.
// CHECK: async.token
module {
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4x!async.token>
}

// -----

// (2) "each element of the array must be a string", on `commit`'s epilogue. IRDL has no element-wise
// constraint over a builtin ArrayAttr. Caught by the runtime engines, which look each stage up in
// BIAS_STAGES / the epilogue dispatch and raise on an unmodelled one rather than skipping it.
// CHECK: epilogue = [7
module {
  %A = "merlin_iface.tensor"() {name = "A", role = "input"} : () -> tensor<4x4xi32>
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<4x4xi8>) -> !merlin_iface.resident
  %acc = "merlin_iface.matmul"(%A, %Wr) : (tensor<4x4xi32>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y = "merlin_iface.commit"(%acc) {name = "Y", epilogue = [7 : i64], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<4x4xi32>
}

// -----

// (3) "each element of the array must be a signless i64", on conv2d's geometry. Same reason. A kernel
// extent that is not an integer cannot be multiplied out, so this is caught the moment the golden
// computes the im2col output extent from it.
// CHECK: kernel = ["three"
module {
  %I = "merlin_iface.tensor"() {name = "I", role = "input"} : () -> tensor<1x8x8x4xi8>
  %W = "merlin_iface.tensor"() {name = "W", role = "weight"} : () -> tensor<3x3x4x4xi8>
  %Wr = "merlin_iface.resident_pack"(%W) {layout = "packed_rhs"} : (tensor<3x3x4x4xi8>) -> !merlin_iface.resident
  %Y = "merlin_iface.conv2d"(%I, %Wr) {kernel = ["three", "three"], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y", epilogue = [], output_dtype = "i32", layout = "nhwc"} : (tensor<1x8x8x4xi8>, !merlin_iface.resident) -> tensor<1x6x6x4xi32>
}
