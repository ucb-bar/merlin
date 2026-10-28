/- Successful initialized execution under the provisional source interpretation. This book
adds general per-tensor source dequantization to established ordinary definedness rules. It does
not infer quantizer domains from typed inputs. The separate whole-module route reuses frozen
definedness only after proving equality of the two interpretations for every function. -/
import MerlinVerify.Check.PolicyW8A8Dequant
import MerlinVerify.Check.DefinedLinalg
import MerlinVerify.Check.DefinedExtended
import MerlinVerify.Check.DefinedProjection
import MerlinVerify.Check.DefinedGeneric
import MerlinVerify.Rules.W8A8SourceFree

namespace MerlinVerify.Check.PolicyW8A8Defined

open MerlinVerify Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem

/-- Reuse the existing book only after checking that this operation and all its regions have
the same meaning under both interpretations. Source operations use their own proved rule. -/
def ordinaryRefinedBook (ib : Nat) : Refined.Book IndexRanges.model sourceSem ib where
  rule Γ op fs := if W8A8SourceFree.operation op then
    (Bounds.book ib).rule Γ op fs else none
  sound Γ op fs rs checked ctx call env args width valid typed := by
    split at checked
    · next conditions =>
      simp only [W8A8SourceFree.operation, Bool.and_eq_true,
        Bool.not_eq_eq_eq_not, Bool.not_true] at conditions
      rw [W8A8SourceFree.regions_eq op.regions call env conditions.2,
        W8A8SourceSemantics.unchanged conditions.1]
      exact (Bounds.book ib).sound Γ op fs rs checked ctx call env args width valid typed
    · cases checked

def ordinaryBook (ib : Nat) : Book sourceSem ib :=
  Refined.Book.erase IndexRanges.model (ordinaryRefinedBook ib)

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

theorem genericAgreement : Structured.GenericAgreement sourceSem := by
  intro ctx call op args regions named
  apply W8A8SourceSemantics.unchanged
  simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]

def leafBook (ib : Nat) : Refined.Book IndexRanges.model sourceSem ib :=
  (Refined.Book.lift IndexRanges.model (dequantBook ib)).orElse (ordinaryRefinedBook ib)

/-- Bounds remain attached to actual bindings throughout each region. The recursive book
checks source operations under the same interpretation as the enclosing executor. -/
def nestedBook (ib : Nat) : Nat → Refined.Book IndexRanges.model sourceSem ib
  | 0 => leafBook ib
  | depth + 1 => {
      rule Γ op fs := ((Structured.genericBook IndexRanges.model genericAgreement ib
        (nestedBook ib depth)).orElse (leafBook ib)).rule Γ op fs
      sound Γ op fs rs checked ctx call env args width valid typed :=
        ((Structured.genericBook IndexRanges.model genericAgreement ib
          (nestedBook ib depth)).orElse (leafBook ib)).sound
            Γ op fs rs checked ctx call env args width valid typed }

def refinedBook (ib : Nat) : Refined.Book IndexRanges.model sourceSem ib where
  rule Γ op fs := (nestedBook ib (RegionDepth.op op)).rule Γ op fs
  sound Γ op fs rs checked ctx call env args width valid typed :=
    (nestedBook ib (RegionDepth.op op)).sound Γ op fs rs checked ctx call env args width valid typed

def book (ib : Nat) : Book sourceSem ib := Refined.Book.erase IndexRanges.model (refinedBook ib)

def check (ib : Nat) (m : Module) (fn : String) (args results : List VTy) : Bool :=
  Refined.check (refinedBook ib) m fn args results ||
    (W8A8SourceFree.moduleCheck m && DefinedExtended.check ib m fn args results)

/-- This is universal definedness, rather than successful execution of selected samples.
Floating bits may be NaN/Inf; initialized means no semantic poison, UB or stuck outcome. -/
theorem check_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (checked : check ib m fn argTys resTys = true) :
    FuncDefined sourceSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m fn := by
  simp only [check, Bool.or_eq_true, Bool.and_eq_true] at checked
  rcases checked with source | ⟨ordinary, frozen⟩
  · exact Refined.check_sound (refinedBook ib) source
  · exact (W8A8SourceFree.defined_iff ordinary).2 (DefinedExtended.check_sound frozen)

#print axioms check_sound
#print axioms genericAgreement

end MerlinVerify.Check.PolicyW8A8Defined
