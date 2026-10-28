/- Actual scalar-body acceptance/refusals and ordered cell traces. These do not
establish the captured diagonal mask or complete attention SSA composition. -/
import MerlinVerify.Rules.W8A8ReductionTensor
import MerlinVerify.Rules.W8A8ReductionOrder

open MerlinVerify Rules W8A8ReductionBody

namespace W8A8ReductionControls

def renamed (name flags : String) (operands : List Nat) : Region :=
  .mk [.mk [{ id := 13, ty := .float .f32 }, { id := 21, ty := .float .f32 }]
    [.mk name [{ id := 99, ty := .float .f32 }] operands
       [("fastmath", .enum "arith" "fastmath" flags)] [] [] [],
     .mk "linalg.yield" [] [99] [] [] [] []]]

theorem actual_body_controls :
    Check.PolicyW8A8ReductionBody.check .maximum (region .maximum) = true ∧
    Check.PolicyW8A8ReductionBody.check .sum (region .sum) = true ∧
    Check.PolicyW8A8ReductionBody.check .maximum (renamed "arith.maximumf" "none" [13, 21]) = true ∧
    Check.PolicyW8A8ReductionBody.check .sum (renamed "arith.addf" "none" [13, 21]) = true := by cbv

theorem valid_but_wrong_body_refusals :
    Check.PolicyW8A8ReductionBody.check .maximum (renamed "arith.addf" "none" [13, 21]) = false ∧
    Check.PolicyW8A8ReductionBody.check .sum (renamed "arith.addf" "fast" [13, 21]) = false ∧
    Check.PolicyW8A8ReductionBody.check .maximum (renamed "arith.maximumf" "none" [21, 21]) = false ∧
    Check.PolicyW8A8ReductionBody.check .sum (renamed "arith.addf" "none" [13, 13]) = false := by cbv

theorem accepted_body_executes {kind : Kind} {actual : Region}
    (checked : Check.PolicyW8A8ReductionBody.check kind actual = true) :
    Check.PolicyW8A8ReductionBody.Correct kind actual :=
  Check.PolicyW8A8ReductionBody.check_sound checked

theorem nan_payload_order_remains_observable :
    W8A8ByteFloat.maximum (0x7fc00001 : FBits .f32) 0xffc00042 = 0x7fc00001 ∧
    W8A8ByteFloat.maximum (0xffc00042 : FBits .f32) 0x7fc00001 = 0xffc00042 ∧
    W8A8ByteFloat.maximum (0x7fc00001 : FBits .f32) 0xffc00042 ≠
      W8A8ByteFloat.maximum (0xffc00042 : FBits .f32) 0x7fc00001 := by decide

theorem signed_zero_maximum :
    W8A8ByteFloat.maximum (0x80000000 : FBits .f32) 0 = 0 ∧
    W8A8ByteFloat.maximum (0 : FBits .f32) 0x80000000 = 0 := by decide

theorem arbitrary_actual_maximum_fold {cap : Nat} (words : List (FBits .f32))
    (members : ∀ word ∈ words, W8A8MaskedMaximum.Admissible cap word) :
    W8A8ReductionCells.reduce .maximum words W8A8MaskedMaximum.negativeInfinity =
      W8A8SoftmaxRow.rowMaximum words :=
  W8A8ReductionOrder.maximum_fold words _ (Or.inl rfl) members

theorem ordered_sum_is_not_reassociated :
    W8A8ReductionCells.reduce .sum [0x4b800000, 0x3f800000, 0xcb800000] 0 = 0 ∧
    W8A8ReductionCells.reduce .sum [0x4b800000, 0xcb800000, 0x3f800000] 0 = 0x3f800000 := by decide

theorem empty_and_masked_rows_keep_negative_infinity :
    W8A8ReductionCells.reduce .maximum [] W8A8MaskedMaximum.negativeInfinity = W8A8MaskedMaximum.negativeInfinity ∧
    W8A8ReductionCells.reduce .maximum [W8A8MaskedMaximum.negativeInfinity]
      W8A8MaskedMaximum.negativeInfinity = W8A8MaskedMaximum.negativeInfinity := by decide

def initial (cell : Idx) : Val :=
  if cell = [0] ∨ cell = [1] then W8A8OrderedBody.value 0 else .poison

def input (point : Idx) : FBits .f32 :=
  if point[1]?.getD 0 = 0 then 0x3f800000 else 0x40000000

def points : List Idx := [[0, 0], [1, 0], [0, 1], [1, 1]]

theorem interleaved_cells_and_unobserved_poison :
    W8A8TensorTrace.accumulate (List.take 1) (W8A8ReductionCells.step .sum input) initial points [0] =
      W8A8OrderedBody.value 0x40400000 ∧
    W8A8TensorTrace.accumulate (List.take 1) (W8A8ReductionCells.step .sum input) initial points [1] =
      W8A8OrderedBody.value 0x40400000 ∧
    W8A8TensorTrace.accumulate (List.take 1) (W8A8ReductionCells.step .sum input) initial points [99] = .poison := by decide

#print axioms actual_body_controls
#print axioms valid_but_wrong_body_refusals
#print axioms accepted_body_executes
#print axioms nan_payload_order_remains_observable
#print axioms signed_zero_maximum
#print axioms arbitrary_actual_maximum_fold
#print axioms ordered_sum_is_not_reassociated
#print axioms empty_and_masked_rows_keep_negative_infinity
#print axioms interleaved_cells_and_unobserved_poison

end W8A8ReductionControls
