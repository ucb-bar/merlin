/- Exact mask bodies, wrong-predicate/branch controls and the conditional diagonal
theorem. This does not assert captured ramp/gather producer correspondence. -/
import MerlinVerify.Check.PolicyW8A8AttentionMaskBody
import MerlinVerify.Rules.W8A8AttentionMaskTensor

open MerlinVerify Rules W8A8AttentionMaskBody

namespace W8A8AttentionMaskControls

theorem canonical_bodies :
    Check.PolicyW8A8AttentionMaskBody.check .lessEqual (comparisonRegion .lessEqual) = true ∧
    Check.PolicyW8A8AttentionMaskBody.check .equal (comparisonRegion .equal) = true ∧
    Check.PolicyW8A8AttentionMaskBody.check .conjunction conjunctionRegion = true ∧
    Check.PolicyW8A8AttentionMaskBody.check .inversion inversionRegion = true ∧
    Check.PolicyW8A8AttentionMaskBody.check .copy copyRegion = true ∧
    Check.PolicyW8A8AttentionMaskBody.check .select selectRegion = true := by cbv

def strictComparison : Region :=
  .mk [.mk [⟨0, .int 64⟩, ⟨1, .int 64⟩, ⟨2, .int 1⟩]
    [.mk "arith.cmpi" [⟨3, .int 1⟩] [0, 1] [("predicate", .int 2 (.int 64))] [] [] [],
     .mk "linalg.yield" [] [3] [] [] [] []]]

def swappedSelection : Region :=
  .mk [.mk [⟨0, .int 1⟩, ⟨1, .float .f32⟩, ⟨2, .float .f32⟩, ⟨3, .float .f32⟩]
    [.mk "arith.select" [⟨4, .float .f32⟩] [0, 2, 1] [] [] [] [], .mk "linalg.yield" [] [4] [] [] [] []]]

theorem changed_meaning_is_refused :
    Check.PolicyW8A8AttentionMaskBody.check .lessEqual strictComparison = false ∧
    Check.PolicyW8A8AttentionMaskBody.check .select swappedSelection = false := by cbv

theorem diagonal_for_same_functions (cell : Idx) (column : Nat) (ramp : Nat → BitVec 64)
    (groups : Idx → Nat → BitVec 64) :
    W8A8AttentionMaskRows.mask cell.length ramp groups (cell ++ [column, column]) = false :=
  W8A8AttentionMaskRows.diagonal cell column ramp groups

theorem unrelated_gathers_can_mask_diagonal :
    !(W8A8AttentionMaskBody.compare .lessEqual (0 : BitVec 64) 0 &&
      W8A8AttentionMaskBody.compare .equal (0 : BitVec 64) 1) = true ∧
    W8A8AttentionMaskRows.selected (fun _ => true) (fun _ => 0) [0, 0] = W8A8MaskedMaximum.negativeInfinity := by decide

theorem selected_score_execution (ctx : Ctx) (call : CallSem) (env : Env) (bits : FBits .f32) :
    evalRegion W8A8SourceSemantics.opSem ctx call env selectRegion
      [.scalar (boolean false), .scalar (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity),
        .scalar (W8A8OrderedBody.value bits), .scalar .poison] =
      .ok [.scalar (W8A8OrderedBody.value bits)] := select_sound ctx call env false _ bits .poison

#print axioms canonical_bodies
#print axioms changed_meaning_is_refused
#print axioms diagonal_for_same_functions
#print axioms unrelated_gathers_can_mask_diagonal
#print axioms selected_score_execution
#print axioms W8A8AttentionMaskRows.finite_row
#print axioms W8A8AttentionMaskTensor.select_sound
#print axioms W8A8AttentionMaskTensor.inversion_sound

end W8A8AttentionMaskControls
