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

def scalarRewrites (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (Rules.IntegerIdempotent.normalizer ib)
    (Rules.IntegerCastNested.normalizer ib)

/-- Shared pipelines defer ordering until all alias families have finished. An alias can
replace a later SSA identifier with an earlier one and invalidate a previous operand order. -/
def unorderedBase (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (scalarRewrites ib) (SplatNormalizer.base ib)

/-- Preserve the standalone integer checker's existing order. Aggregate pipelines instead
use unorderedBase followed by one final integerOrder stage after their other rewrites. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (scalarRewrites ib)
    (BodyNormalize.sequential (integerOrder ib) (SplatNormalizer.base ib))

private theorem sequential_run_assoc {ib : Nat} (a b c : SplatNormalizer.Normalizer ib)
    (d : Nat) (Γ : TEnv) (facts : List UniformFacts.Fact) (ops : List Op) (yields : List VTy) :
    (BodyNormalize.sequential (BodyNormalize.sequential a b) c).run d Γ facts ops yields =
      (BodyNormalize.sequential a (BodyNormalize.sequential b c)).run d Γ facts ops yields := by
  cases hf : a.run d Γ facts ops yields with
  | none => simp [BodyNormalize.sequential, hf]
  | some middle =>
    by_cases hc : checkOps (merlinBook ib) d Γ middle yields = true
    · simp [BodyNormalize.sequential, hf, hc]
    · simp [BodyNormalize.sequential, hf, hc]

/-- Extraction changes no standalone acceptance or output, for every local input and budget. -/
theorem base_run_legacy (ib d : Nat) (Γ : TEnv) (facts : List UniformFacts.Fact)
    (ops : List Op) (yields : List VTy) :
    (base ib).run d Γ facts ops yields =
      (BodyNormalize.sequential (Rules.IntegerIdempotent.normalizer ib)
        (BodyNormalize.sequential (Rules.IntegerCastNested.normalizer ib)
          (BodyNormalize.sequential (integerOrder ib) (SplatNormalizer.base ib)))).run d Γ facts ops yields :=
  sequential_run_assoc _ _ _ _ _ _ _ _

/-- Run once after all alias-producing families, including parent-region capture renames.
Only operand ordering is revisited; fact learning and rewrite families are not repeated. -/
def ordering : TypedNormalizer where
  run ib d Γ ops hints yields :=
    (BodyNormalize.recursive (integerOrder ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).run
      d Γ [] ops yields
  sound ib d Γ ops hints normalized yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (integerOrder ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).sound
      d Γ [] ops normalized yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

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
