/- Instance qualification of actual captured-source operations through the additive adapter.
Every accepted invocation has successful, typed, nonpoison candidate execution. This is not
frozen source dispatch, independent specification approval or all-input program definedness. -/
import MerlinVerify.Rules.W8A8SourceContract
import MerlinVerify.Check.PolicyW8A8SourceDomain

namespace MerlinVerify.Check.PolicyW8A8Source

open MerlinVerify MerlinVerify.Sem.Ops Rules
open W8A8SourceContract

def outputType (c : Config) : VTy :=
  .tensor c.shape (if c.kind = .quantize then .int 8 else .float .f32)

def Correct (ib : Nat) (types : List (ValueId × Ty)) (op : Op) (base : FloatModel)
    (args : List RVal) : Prop :=
  ∃ c output, inspect types op = some c ∧
    eval types op (W8A8ByteModel.extend base) args = .ok [.tensor output] ∧
    RVal.HasTy ib (.tensor output) (outputType c) ∧ (RVal.tensor output).Defined

def initialized (input : Tensor) : Bool :=
  (allPoints input.shape).all fun point => match input.get point with
    | .poison => false
    | .some _ => true

theorem initialized_sound {input : Tensor} (checked : initialized input = true) :
    (RVal.tensor input).Defined := by
  intro point bounds poison
  have tested := List.all_eq_true.1 checked point ((mem_allPoints _ _).2 bounds)
  simp [poison] at tested

def check (types : List (ValueId × Ty)) (op : Op) (base : FloatModel) (args : List RVal) : Bool :=
  match inspect types op with
  | none => false
  | some c => match c.kind with
    | .quantize => PolicyW8A8SourceDomain.check c.zeroWidth base c.shape args
    | .dequantize => match args with
      | [.tensor input, .tensor scale, .tensor zero] =>
        W8A8DequantWidth.validInputs c.zeroWidth ⟨false, 0, c.shape⟩ input scale zero &&
          initialized input && initialized scale && initialized zero
      | _ => false

theorem check_sound {ib : Nat} {types : List (ValueId × Ty)} {op : Op} {base : FloatModel}
    {args : List RVal} (checked : check types op base args = true) : Correct ib types op base args := by
  unfold check at checked
  cases accepted : inspect types op with
  | none => simp [accepted] at checked
  | some c =>
    simp only [accepted] at checked
    cases kind : c.kind with
    | quantize =>
      simp only [kind] at checked
      obtain ⟨output, executed, typed, defined⟩ := PolicyW8A8SourceDomain.check_sound (ib := ib) checked
      refine ⟨c, output, accepted, ?_, ?_, defined⟩
      · rw [eval_recognized accepted, W8A8SourceContract.run, kind]
        exact executed
      · simpa [outputType, kind] using typed
    | dequantize =>
      simp only [kind] at checked
      split at checked
      · next input scale zero =>
        simp only [Bool.and_eq_true] at checked
        obtain ⟨⟨⟨valid, inputInit⟩, scaleInit⟩, zeroInit⟩ := checked
        refine ⟨c, W8A8DequantWidth.result c.zeroWidth ⟨false, 0, c.shape⟩
          (W8A8ByteModel.extend base) input scale zero, accepted, ?_, ?_, ?_⟩
        · rw [eval_recognized accepted, W8A8SourceContract.run, kind]
          simp [W8A8DequantWidth.run, valid]
        · simpa [outputType, kind] using W8A8DequantWidth.result_typed ib c.zeroWidth
            ⟨false, 0, c.shape⟩ (W8A8ByteModel.extend base) input scale zero
        · exact W8A8DequantWidth.result_defined _ (Or.inl rfl) valid
            (initialized_sound inputInit) (initialized_sound scaleInit) (initialized_sound zeroInit)
      · cases checked

#print axioms initialized_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8Source
