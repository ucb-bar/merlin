/- Compose independently proved exact normalizations of both actual programs before comparison.
Every intermediate program is checked, and execution remains exactly equal throughout. -/
import MerlinVerify.Check.TypedNormalizeC1
import MerlinVerify.Check.ReshapeNested
import MerlinVerify.Check.IntegerBody
import MerlinVerify.Check.ScalarCarrier
import MerlinVerify.Check.DenseExtract
import MerlinVerify.Check.FromElementsSelect
import MerlinVerify.Rules.IndexCompare
import MerlinVerify.Check.ConstantAlias
import MerlinVerify.Rules.IndexConstant
import MerlinVerify.Rules.CastCancel

namespace MerlinVerify.Check.SharedNormalize

open MerlinVerify MerlinVerify.Sem.Ops

/-- Local families share one recursive body traversal. The order lets materialization and
carrier aliases expose scalar rewrite opportunities before the integer stage. -/
def base (ib : Nat) (hints : List Op) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (FromElementsSelect.base ib)
    (BodyNormalize.sequential (DenseExtract.base ib (DenseExtract.hintOps hints))
      (BodyNormalize.sequential (ScalarCarrier.base ib)
        (BodyNormalize.sequential (BodyNormalize.sequential (IntegerBody.base ib) (Rules.CastCancel.base ib))
          (BodyNormalize.sequential
            (FactNormalize.base Rules.IndexCompare.family ib (DenseExtract.hintOps hints))
            (BodyNormalize.sequential
              (FactNormalize.base Rules.IndexConstant.family ib [])
              (ConstantAlias.base ib))))))

/-- Adapt the exact body normalizers without assuming any producer facts at function entry.
The source and target syntax supply only a finite traversal budget. -/
def bodies : TypedNormalizer where
  run ib d Γ ss hints yields :=
    let depth := max (RegionDepth.ops ss) (RegionDepth.ops hints)
    (BodyNormalize.recursive (base ib hints) depth).run d Γ [] ss yields
  sound ib d Γ ss hints ns yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (base ib hints)
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
