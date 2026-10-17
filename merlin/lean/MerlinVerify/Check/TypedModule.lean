/- Module accounting and declared input domains for the typed rewrite checker. -/
import MerlinVerify.Check.TypedC1
import MerlinVerify.Check.Module
import MerlinVerify.Rules.TypedGeneralize

namespace MerlinVerify.Check.TypedModule

open MerlinVerify MerlinVerify.Check

/-- Decode the domain of a function. TypedC1 independently checks the actual block arguments and
source execution against it; a signature annotation alone is not proof of any runtime fact. -/
def signature? (ib : Nat) (m : Module) (fn : String) : Option (List VTy × List VTy) :=
  m.ops.findSome? fun op =>
    match op.asFunc?, op.attr? "function_type" with
    | some f, some (.type (.function ins outs)) =>
      if f.name == fn then do
        let a ← ins.mapM (vtyOf ib)
        let r ← outs.mapM (vtyOf ib)
        pure (a, r)
      else none
    | _, _ => none

def check (ib : Nat) (src tgt : Module) : Bool :=
  (Module.definedNames src).all fun fn =>
    match signature? ib src fn with
    | some (a, r) => c1Check (synthC1 src tgt fn) ||
        TypedC1.check Rules.TypedGeneralize.expansion ib src tgt fn a r
    | none => false

theorem check_sound {ib : Nat} {src tgt : Module} (h : check ib src tgt = true)
    {fn : String} (hf : fn ∈ Module.definedNames src) {argTys resTys : List VTy}
    (hs : signature? ib src fn = some (argTys, resTys)) :
    FuncRefinesC Sem.Ops.opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := by
  have hc := List.all_eq_true.1 h fn hf
  simp only [hs, Bool.or_eq_true] at hc
  rcases hc with hu | ht
  · have h := c1Check_sound (synthC1 src tgt fn) hu
    exact fun ctx _ args _ => h ctx trivial args trivial
  · exact TypedC1.check_sound Rules.TypedGeneralize.expansion ht

end MerlinVerify.Check.TypedModule
