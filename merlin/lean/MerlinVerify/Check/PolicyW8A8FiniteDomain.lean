/- Model-independent input guards for a conditional numerical-domain theorem. Finite data and
normal positive scale are checked from actual bits. Primitive model laws remain explicit proof
premises; this checker neither proves those laws nor qualifies an entire function or pass. -/
import MerlinVerify.Rules.W8A8FiniteLaws
import MerlinVerify.Check.PolicyW8A8SourceDomain

namespace MerlinVerify.Check.PolicyW8A8FiniteDomain

open MerlinVerify MerlinVerify.Sem.Ops Rules

/-- No flag supplies arithmetic correctness: each admitted floating model must carry proofs
of the primitive laws. The input guard itself does not evaluate any floating operation. -/
def Correct (ib width : Nat) (shape : List Nat) (args : List RVal) : Prop :=
  ∀ base, W8A8FiniteLaws.Laws base → PolicyW8A8SourceDomain.Correct ib width base shape args

def pointCheck (value : Val) : Bool :=
  match Dequant.float32 value with
  | none => false
  | some bits => decide (W8A8FiniteLaws.Finite bits)

theorem pointCheck_sound {value : Val} (checked : pointCheck value = true) :
    value ≠ .poison ∧ ∀ x, Dequant.float32 value = some x → W8A8FiniteLaws.Finite x := by
  unfold pointCheck at checked
  cases decoded : Dequant.float32 value with
  | none => simp [decoded] at checked
  | some x =>
    have finite : W8A8FiniteLaws.Finite x := by simpa [decoded] using checked
    refine ⟨?_, ?_⟩
    · intro poison
      simp [poison, Dequant.float32] at decoded
    · intro y same
      have equal : x = y := Option.some.inj same
      simpa [equal] using finite

def pointsCheck (shape : List Nat) (input : Tensor) : Bool :=
  (allPoints shape).all fun point => pointCheck (input.get point)

theorem pointsCheck_sound {shape : List Nat} {input : Tensor} (shapeEq : input.shape = shape)
    (checked : pointsCheck shape input = true) :
    (RVal.tensor input).Defined ∧ ∀ point, Idx.InBounds shape point →
      ∀ x, Dequant.float32 (input.get point) = some x → W8A8FiniteLaws.Finite x := by
  have each := List.all_eq_true.1 checked
  constructor
  · intro point bounds
    have hp : Idx.InBounds shape point := shapeEq ▸ bounds
    exact (pointCheck_sound (each point ((mem_allPoints _ _).2 hp))).1
  · intro point bounds
    exact (pointCheck_sound (each point ((mem_allPoints _ _).2 bounds))).2

def check (width : Nat) (shape : List Nat) : List RVal → Bool
  | [.tensor input, .tensor scale, .tensor zero] =>
    if W8A8SourceQuantize.validInputs width shape input scale zero then
      match Dequant.float32 (scale.get []), Dequant.bits width (zero.get []) with
      | some s, some _ => W8A8FiniteLaws.scaleCheck s && pointsCheck shape input
      | _, _ => false
    else false
  | _ => false

theorem check_sound {ib width : Nat} {shape : List Nat} {args : List RVal}
    (checked : check width shape args = true) : Correct ib width shape args := by
  unfold check at checked
  split at checked
  · next input scale zero =>
    split at checked
    · next valid =>
      cases scaleDecoded : Dequant.float32 (scale.get []) with
      | none => simp [scaleDecoded] at checked
      | some s =>
        cases zeroDecoded : Dequant.bits width (zero.get []) with
        | none => simp [scaleDecoded, zeroDecoded] at checked
        | some z =>
          simp only [scaleDecoded, zeroDecoded, Bool.and_eq_true] at checked
          have data := valid
          simp only [W8A8SourceQuantize.validInputs, Bool.and_eq_true, Bool.or_eq_true,
            beq_iff_eq, decide_eq_true_eq] at data
          obtain ⟨⟨⟨⟨allowed, ⟨shapeEq, _, _⟩⟩, _⟩, _⟩, _⟩ := data
          obtain ⟨initialized, finite⟩ := pointsCheck_sound shapeEq checked.2
          intro base laws
          apply W8A8SourceQuantize.run_defined valid scaleDecoded zeroDecoded
            (W8A8FiniteLaws.scaleCheck_admissible checked.1) initialized
          intro point bounds x decoded
          exact W8A8FiniteLaws.shifted_nonNaN laws allowed (finite point bounds x decoded) checked.1
    · cases checked
  · cases checked

#print axioms pointCheck_sound
#print axioms pointsCheck_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8FiniteDomain
