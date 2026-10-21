/- Successful initialized execution under the provisional source interpretation. This book
adds general per-tensor source dequantization to the established ordinary leaf rules. It does
not infer quantizer domains from typed inputs or silently import frozen region meanings. -/
import MerlinVerify.Check.PolicyW8A8Dequant
import MerlinVerify.Check.DefinedLinalg

namespace MerlinVerify.Check.PolicyW8A8Defined

open MerlinVerify Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem

/-- Restrict the existing book to operations with no regions, where changing the evaluator's
region semantics has no effect. Source operations must use their own proved rule. -/
def ordinaryBook (ib : Nat) : Book sourceSem ib where
  rule Γ op fs := if W8A8SourceSemantics.isSource op.name = false ∧ op.regions = [] then
    (Definedness.book ib).rule Γ op fs else none
  sound Γ op fs rs checked ctx call env args width valid typed := by
    split at checked
    · next conditions =>
      rw [W8A8SourceSemantics.unchanged conditions.1]
      obtain ⟨values, executed, outputFacts⟩ :=
        (Definedness.book ib).sound Γ op fs rs checked ctx call env args width valid typed
      exact ⟨values, by simpa only [conditions.2, evalRegions] using executed, outputFacts⟩
    · cases checked

def dequantBook (ib : Nat) : Book sourceSem ib where
  rule _ op fs := do
    let results ← op.results.mapM fun result => vtyOf ib result.ty
    if fs.all (·.initialized) && PolicyW8A8Dequant.check op (fs.map (·.ty)) results then
      some (clean results) else none
  sound Γ op fs rs checked ctx call env args _width _valid typed := by
    cases resultTypes : op.results.mapM (fun result => vtyOf ib result.ty) with
    | none => simp [resultTypes] at checked
    | some results =>
      simp only [resultTypes, bind, Option.bind] at checked
      split at checked
      · next accepted =>
        cases checked
        simp only [Bool.and_eq_true] at accepted
        obtain ⟨initialized, dequant⟩ := accepted
        obtain ⟨values, executed, outputTyped, outputDefined⟩ :=
          PolicyW8A8Dequant.check_sound dequant ctx call env typed.typed
        exact ⟨values, executed, inputFacts outputTyped
          (outputDefined (typed.defined (List.all_eq_true.1 initialized)))⟩
      · cases checked

def book (ib : Nat) : Book sourceSem ib := (dequantBook ib).orElse (ordinaryBook ib)

def check (ib : Nat) (m : Module) (fn : String) (args results : List VTy) : Bool :=
  Definedness.check (book ib) m fn args results

/-- This is universal definedness, rather than successful execution of selected samples.
Floating bits may be NaN/Inf; initialized means no semantic poison, UB or stuck outcome. -/
theorem check_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (checked : check ib m fn argTys resTys = true) :
    FuncDefined sourceSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m fn :=
  Definedness.check_sound (book ib) checked

#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8Defined
