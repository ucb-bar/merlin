/- Top-level uniform-input comparison through the shared exact-normalizer checker. -/
import MerlinVerify.Check.SplatNormalizer

namespace MerlinVerify.Check.SplatC1

open MerlinVerify MerlinVerify.Sem.Ops

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  SplatNormalizer.check ib (SplatNormalizer.base ib) src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := SplatNormalizer.check_sound h

end MerlinVerify.Check.SplatC1
