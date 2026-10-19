/-
Proposed independent frontend intent primitives, pending human review.
PyTorch v2.10.0: c10/util/TypeCast.h:64-82,189-192 (real/bool casts);
aten/src/ATen/native/Linear.cpp:50-81,132-142 (leading dimensions and weight transpose).
These are scalar/coordinate contracts, not a completed source-program transformer.
They introduce no floating-point algebra laws or claims about BLAS reduction order.
-/
import MerlinVerify.Sem.Support
import MerlinVerify.Check.Types

namespace MerlinVerify.Spec.FrontendIntent

/-- The intended f32 boolean cast uses the positive IEEE constants. -/
def boolToF32 (x : Option Bool) : Val :=
  match x with
  | none => .poison
  | some b => .some (.flt .f32 (if b then 0x3f800000 else 0))

/-- Both signed zeros are false; every other f32 encoding, including NaN, is true.
The sign bit is excluded before testing for zero. Poison stays poison. -/
def f32ToBool (x : Option (FBits .f32)) : Val :=
  match x with
  | none => .poison
  | some bits => .some (.bits 1 (if bits.toNat % 2^31 = 0 then 0 else 1))

/-- Tensor extension of the boolean-to-f32 contract. -/
def boolToF32Tensor (shape : List Nat) (x : Idx → Option Bool) : Tensor :=
  Tensor.canon shape (.float .f32) fun p => boolToF32 (x p)

/-- Tensor extension of the f32-to-boolean contract. -/
def f32ToBoolTensor (shape : List Nat) (x : Idx → Option (FBits .f32)) : Tensor :=
  Tensor.canon shape (.int 1) fun p => f32ToBool (x p)

/-- For input [leading..., K] and weight [N, K], linear returns [leading..., N]. -/
def linearOutputShape (leading : List Nat) (outputs : Nat) : List Nat :=
  leading ++ [outputs]

/-- Every leading coordinate selects an independent input row. -/
def linearInputIndex (leadingPoint : Idx) (k : Nat) : Idx := leadingPoint ++ [k]

/-- Torch's linear weight storage is output-channel first, before transposition for matmul. -/
def linearWeightIndex (outputChannel k : Nat) : Idx := [outputChannel, k]

/-- Proposed claim labels; these declarations do not assert that a repair is verified. -/
def boolToFloatClaim : ClaimType := .policy "frontend-bool-to-float"
def floatToBoolClaim : ClaimType := .policy "frontend-float-to-bool"
def linearClaim : ClaimType := .policy "frontend-linear"

#guard boolToF32 (some true) = .some (.flt .f32 0x3f800000)
#guard boolToF32 (some false) = .some (.flt .f32 0)
#guard boolToF32 none = .poison
#guard f32ToBool (some 0x80000000) = .some (.bits 1 0)
#guard f32ToBool (some 0x3f000000) = .some (.bits 1 1)
#guard f32ToBool (some 0x7fc00000) = .some (.bits 1 1)
#guard f32ToBool none = .poison

end MerlinVerify.Spec.FrontendIntent
