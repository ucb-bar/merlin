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
import MerlinVerify.Rules.IntegerCastPartial
import MerlinVerify.Check.InsertExtract
import MerlinVerify.Rules.IntegerNeutral
import MerlinVerify.Rules.IntegerCompareOrder
import MerlinVerify.Check.RegionExact
import MerlinVerify.Check.AlphaNormalize
import MerlinVerify.Check.ScalarConstantHoist
import MerlinVerify.Check.BitcastCancel
import MerlinVerify.Check.TensorEmptyDead
import MerlinVerify.Check.ParallelFusion
import MerlinVerify.Check.IdentityReturnComparison

namespace MerlinVerify.Check.SharedNormalize

open MerlinVerify MerlinVerify.Sem.Ops

/-- Local families share one recursive body traversal. The order lets materialization and
carrier aliases expose scalar rewrite opportunities before the integer stage. -/
def base (ib : Nat) (hints : List Op) : SplatNormalizer.Normalizer ib :=
  let selection := BodyNormalize.sequential (FromElementsSelect.base ib)
    (DenseExtract.base ib (DenseExtract.hintOps hints))
  let scalarAliases := BodyNormalize.sequential (ScalarCarrier.base ib)
    (BodyNormalize.sequential (IntegerBody.unorderedBase ib)
      (BodyNormalize.sequential (Rules.IntegerCastPartial.base ib) (BitcastCancel.base ib)))
  let indexFacts := BodyNormalize.sequential
    (FactNormalize.base Rules.IndexCompare.family ib (DenseExtract.hintOps hints))
    (FactNormalize.base Rules.IndexConstant.family ib [])
  let insertions := BodyNormalize.sequential (InsertExtract.base ib) (Rules.IntegerNeutral.base ib)
  let aliases := BodyNormalize.sequential selection
    (BodyNormalize.sequential scalarAliases (BodyNormalize.sequential indexFacts insertions))
  BodyNormalize.sequential (ScalarConstantHoist.base ib)
    (BodyNormalize.sequential aliases (TensorEmptyDead.base ib))

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

/-- Unconditional ordering families share one traversal through every original region, including
named-op bodies. Both families retain result bindings and prove equality for arbitrary environments. -/
def scalarOrdering : List Op → List Op :=
  Rules.IntegerCompareOrder.operations ∘ Rules.IntegerOrder.operations

theorem scalarOrdering_exact : RegionExact.Exact scalarOrdering :=
  RegionExact.compose
    (fun ctx call env ops => Rules.IntegerOrder.operations_equal ctx call ops env)
    (fun ctx call env ops => Rules.IntegerCompareOrder.operations_equal ctx call ops env)

/-- Fusion runs before scalar identities can erase its matched primitive. Local families then
preserve escaping bindings and delete only total dead producers. Checked lexical names make the
final numeric-ID ordering stable after aliases and scope motion. -/
def normalizer : TypedNormalizer :=
  ParallelFusion.normalizer.sequential
    (bodies.sequential (ReshapeNested.normalizer.sequential
      (AlphaNormalize.normalizer.sequential
        (RegionExact.normalizer scalarOrdering scalarOrdering_exact))))

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  TypedNormalizeC1.checkWith normalizer
    (TypedComparison.script.orElse IdentityReturn.comparison) ib src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := TypedNormalizeC1.checkWith_sound normalizer
        (TypedComparison.script.orElse IdentityReturn.comparison) h

end MerlinVerify.Check.SharedNormalize
