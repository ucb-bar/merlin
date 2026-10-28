/- Region inference with the checked iteration space of the actual enclosing operation.
Bounds constrain only the current body's loop coordinates, independently of numerical laws. -/
import MerlinVerify.Check.DefinedRegions

namespace MerlinVerify.Check.Definedness.Contextual

open MerlinVerify Refined

def PointConditions (allowed : Conditions) (ranges : List Nat) : Conditions :=
  fun ctx call => allowed ctx call ∧ Idx.InBounds ranges ctx.loopIndex

structure PointRegions {Info : Type} (M : Model Info) (sem : OpSem) (ib : Nat) (allowed : Conditions) where
  infer : Refined.FEnv Info → List Nat → Region → List (Refined.Fact Info) → Option (List (Refined.Fact Info))
  sound : ∀ Γ ranges region inputs outputs, infer Γ ranges region inputs = some outputs →
    ∀ ctx call env args, ctx.indexBits = ib → allowed ctx call → Idx.InBounds ranges ctx.loopIndex →
      Refined.EnvHolds M ib Γ env → Refined.FactsHold M ib args inputs → ∃ values,
        evalRegion sem ctx call env region args = .ok values ∧
        Refined.FactsHold M ib values outputs ∧ ResultsDefined values

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

def PointRegions.fromRegions (regions : Regions M sem ib allowed) : PointRegions M sem ib allowed where
  infer Γ _ body inputs := regions.infer Γ body inputs
  sound Γ _ body inputs outputs checked ctx call env args width permitted _ valid holds :=
    regions.sound Γ body inputs outputs checked ctx call env args width permitted valid holds

def PointRegions.operations (books : ∀ ranges, Book M sem ib (PointConditions allowed ranges)) :
    PointRegions M sem ib allowed where
  infer Γ ranges body inputs := inferRegion (books ranges) Γ body inputs
  sound _ ranges _ _ _ checked _ _ _ _ width permitted bounds valid holds :=
    inferRegion_sound (books ranges) checked width ⟨permitted, bounds⟩ valid holds

def PointRegions.orElse (first next : PointRegions M sem ib allowed) : PointRegions M sem ib allowed where
  infer Γ ranges body inputs := (first.infer Γ ranges body inputs).orElse fun _ => next.infer Γ ranges body inputs
  sound Γ ranges body inputs outputs checked ctx call env args width permitted bounds valid holds := by
    cases selected : first.infer Γ ranges body inputs with
    | none =>
      simp only [selected, Option.orElse_none] at checked
      exact next.sound Γ ranges body inputs outputs checked ctx call env args width permitted bounds valid holds
    | some results =>
      simp only [selected, Option.orElse_some, Option.some.injEq] at checked
      subst outputs
      exact first.sound Γ ranges body inputs results selected ctx call env args width permitted bounds valid holds

#print axioms PointRegions.fromRegions
#print axioms PointRegions.operations
#print axioms PointRegions.orElse

end MerlinVerify.Check.Definedness.Contextual
