/- Compose independently proved exact normalizations of both actual programs before comparison.
Every intermediate program is checked, and execution remains exactly equal throughout. -/
import MerlinVerify.Check.TypedNormalizeC1
import MerlinVerify.Check.ReshapeNested
import MerlinVerify.Check.IntegerBody

namespace MerlinVerify.Check.SharedNormalize

open MerlinVerify MerlinVerify.Sem.Ops

/-- Adapt the exact body normalizers without assuming any producer facts at function entry.
The source and target syntax supply only a finite traversal budget. -/
def bodies : TypedNormalizer where
  run ib d Γ ss hints yields :=
    let depth := max (RegionDepth.ops ss) (RegionDepth.ops hints)
    (BodyNormalize.recursive (IntegerBody.base ib) depth).run d Γ [] ss yields
  sound ib d Γ ss hints ns yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (IntegerBody.base ib)
      (max (RegionDepth.ops ss) (RegionDepth.ops hints))).sound
      d Γ [] ss ns yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

/-- Local families preserve escaping bindings and prove exact deletion of total dead producers.
Adding a family requires its normalization proof, not a new whole-program walker. -/
def normalizer : TypedNormalizer :=
  bodies.sequential ReshapeNested.normalizer

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  TypedNormalizeC1.check normalizer ib src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := TypedNormalizeC1.check_sound normalizer h

end MerlinVerify.Check.SharedNormalize
