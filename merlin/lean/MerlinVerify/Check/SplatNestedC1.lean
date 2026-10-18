/- Nested comparison uses the same exact checker, with a budget from actual function bodies. -/
import MerlinVerify.Check.SplatNested
import MerlinVerify.Check.RegionDepth

namespace MerlinVerify.Check.SplatNestedC1

open MerlinVerify MerlinVerify.Sem.Ops

def checkAt (depth ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  SplatNormalizer.check ib (SplatNested.normalizer ib depth) src tgt fn argTys resTys

theorem checkAt_sound {depth ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : checkAt depth ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := SplatNormalizer.check_sound h

/-- Only completeness depends on this structural budget. Every depth has the same soundness proof. -/
def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  match src.func? fn, tgt.func? fn with
  | some source, some target =>
    checkAt (max (RegionDepth.regions [source.body]) (RegionDepth.regions [target.body]))
      ib src tgt fn argTys resTys
  | _, _ => false

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := by
  unfold check at h
  split at h
  · exact checkAt_sound h
  · cases h

end MerlinVerify.Check.SplatNestedC1
