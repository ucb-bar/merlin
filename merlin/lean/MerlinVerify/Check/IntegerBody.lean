/- Exact integer operand ordering inside supported structured regions, optionally composed with
uniform-input normalization. Actual source and target programs retain all their original bytes. -/
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Rules.IntegerOrder
import MerlinVerify.Rules.IntegerIdempotent
import MerlinVerify.Rules.IntegerCastNested

namespace MerlinVerify.Check.IntegerBody

open MerlinVerify MerlinVerify.Sem.Ops

/-- This local family is exact on arbitrary environments; the shared interface additionally
tracks the source types, region scopes and producer facts needed by composed families. -/
def integerOrder (ib : Nat) : SplatNormalizer.Normalizer ib where
  run _ _ _ ops _ := some (Rules.IntegerOrder.operations ops)
  sound _ _ _ ops normalized _ ctx call env _ hn _ _ _ _ := by
    cases hn
    exact Rules.IntegerOrder.operations_equal ctx call ops env

def base (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (Rules.IntegerIdempotent.normalizer ib)
    (BodyNormalize.sequential (Rules.IntegerCastNested.normalizer ib)
      (BodyNormalize.sequential (integerOrder ib) (SplatNormalizer.base ib)))

def checkAt (depth ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  SplatNormalizer.check ib (BodyNormalize.recursive (base ib) depth) src tgt fn argTys resTys

theorem checkAt_sound {depth ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : checkAt depth ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := SplatNormalizer.check_sound h

def functionDepth (m : Module) (fn : String) : Nat :=
  match m.func? fn with
  | some f => RegionDepth.regions [f.body]
  | none => 0

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  checkAt (max (functionDepth src fn) (functionDepth tgt fn)) ib src tgt fn argTys resTys

/-- Equality of both normalizations and the final alpha comparison establish refinement of the
actual programs. Target stuck/UB behavior cannot be hidden by a conditional normalization. -/
theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := checkAt_sound h

end MerlinVerify.Check.IntegerBody
