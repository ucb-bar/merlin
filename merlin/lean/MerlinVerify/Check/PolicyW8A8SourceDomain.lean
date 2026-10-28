/- Executable numerical-domain checking for actual inputs of the uninstalled source quantizer.
Acceptance proves initialized execution of this instance, not a property of every program input.
The floating base model stays explicit; no IEEE law or policy approval is introduced. -/
import MerlinVerify.Rules.W8A8SourceQuantize

namespace MerlinVerify.Check.PolicyW8A8SourceDomain

open MerlinVerify MerlinVerify.Sem.Ops Rules

/-- The soundness contract is successful, typed, nonpoison execution of the actual candidate
entry point. No caller-supplied definedness or non-NaN flag is a premise. -/
def Correct (ib width : Nat) (base : FloatModel) (shape : List Nat) (args : List RVal) : Prop :=
  ∃ output, W8A8SourceQuantize.run width (W8A8ByteModel.extend base) shape args = .ok [.tensor output] ∧
    RVal.HasTy ib (.tensor output) (.tensor shape (.int 8)) ∧ (RVal.tensor output).Defined

def pointCheck (width : Nat) (base : FloatModel) (value : Val) (scale : FBits .f32)
    (zero : BitVec width) : Bool :=
  match Dequant.float32 value with
  | none => false
  | some x => !W8A8ByteFloat.isNaN (W8A8SourceQuantize.shifted width base x scale zero)

theorem pointCheck_sound {width : Nat} {base : FloatModel} {value : Val}
    {scale : FBits .f32} {zero : BitVec width} (checked : pointCheck width base value scale zero = true) :
    value ≠ .poison ∧ ∀ x, Dequant.float32 value = some x →
      W8A8ByteFloat.isNaN (W8A8SourceQuantize.shifted width base x scale zero) = false := by
  unfold pointCheck at checked
  cases decoded : Dequant.float32 value with
  | none => simp [decoded] at checked
  | some x =>
    have nonNaN : W8A8ByteFloat.isNaN (W8A8SourceQuantize.shifted width base x scale zero) = false := by
      simpa [decoded] using checked
    refine ⟨?_, ?_⟩
    · intro poison
      simp [poison, Dequant.float32] at decoded
    · intro y same
      have equal : x = y := Option.some.inj same
      simpa [equal] using nonNaN

def pointsCheck (width : Nat) (base : FloatModel) (shape : List Nat) (input : Tensor)
    (scale : FBits .f32) (zero : BitVec width) : Bool :=
  (allPoints shape).all fun point => pointCheck width base (input.get point) scale zero

theorem pointsCheck_sound {width : Nat} {base : FloatModel} {shape : List Nat} {input : Tensor}
    {scale : FBits .f32} {zero : BitVec width} (shapeEq : input.shape = shape)
    (checked : pointsCheck width base shape input scale zero = true) :
    (RVal.tensor input).Defined ∧ W8A8SourceQuantize.Domain width base shape input scale zero := by
  have each := List.all_eq_true.1 checked
  constructor
  · intro point bounds
    have hp : Idx.InBounds shape point := shapeEq ▸ bounds
    exact (pointCheck_sound (each point ((mem_allPoints _ _).2 hp))).1
  · intro point bounds
    exact (pointCheck_sound (each point ((mem_allPoints _ _).2 bounds))).2

def check (width : Nat) (base : FloatModel) (shape : List Nat) : List RVal → Bool
  | [.tensor input, .tensor scale, .tensor zero] =>
    if W8A8SourceQuantize.validInputs width shape input scale zero then
      match Dequant.float32 (scale.get []), Dequant.bits width (zero.get []) with
      | some s, some z => W8A8SourceQuantize.scaleAdmissible s && pointsCheck width base shape input s z
      | _, _ => false
    else false
  | _ => false

theorem check_sound {ib width : Nat} {base : FloatModel} {shape : List Nat} {args : List RVal}
    (checked : check width base shape args = true) : Correct ib width base shape args := by
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
          simp only [W8A8SourceQuantize.validInputs, Bool.and_eq_true, decide_eq_true_eq] at data
          obtain ⟨⟨⟨⟨_, ⟨shapeEq, _, _⟩⟩, _⟩, _⟩, _⟩ := data
          obtain ⟨initialized, domain⟩ := pointsCheck_sound shapeEq checked.2
          exact W8A8SourceQuantize.run_defined valid scaleDecoded zeroDecoded checked.1 initialized domain
    · cases checked
  · cases checked

#print axioms pointCheck_sound
#print axioms pointsCheck_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8SourceDomain
