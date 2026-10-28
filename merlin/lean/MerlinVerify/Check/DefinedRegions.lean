/- A checked whole-region contract can complement operation-by-operation inference. Both
routes prove execution under the same semantics, captures and explicit context/call premises. -/
import MerlinVerify.Check.DefinedContext

namespace MerlinVerify.Check.Definedness.Contextual

open MerlinVerify Refined

structure Regions {Info : Type} (M : Model Info) (sem : OpSem) (ib : Nat) (allowed : Conditions) where
  infer : Refined.FEnv Info → Region → List (Refined.Fact Info) → Option (List (Refined.Fact Info))
  sound : ∀ Γ region inputs outputs, infer Γ region inputs = some outputs →
    ∀ ctx call env args, ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
      Refined.FactsHold M ib args inputs → ∃ values,
        evalRegion sem ctx call env region args = .ok values ∧
        Refined.FactsHold M ib values outputs ∧ ResultsDefined values

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

def Regions.operations (book : Book M sem ib allowed) : Regions M sem ib allowed where
  infer := inferRegion book
  sound _ _ _ _ checked _ _ _ _ width permitted valid inputs :=
    inferRegion_sound book checked width permitted valid inputs

def Regions.empty : Regions M sem ib allowed where
  infer _ _ _ := none
  sound _ _ _ _ checked := by cases checked

def Regions.restrict {stronger : Conditions} (regions : Regions M sem ib allowed)
    (implies : ∀ ctx call, stronger ctx call → allowed ctx call) : Regions M sem ib stronger where
  infer := regions.infer
  sound Γ body inputs outputs checked ctx call env args width permitted valid holds :=
    regions.sound Γ body inputs outputs checked ctx call env args width (implies ctx call permitted) valid holds

def Regions.orElse (first next : Regions M sem ib allowed) : Regions M sem ib allowed where
  infer Γ region inputs := (first.infer Γ region inputs).orElse fun _ => next.infer Γ region inputs
  sound Γ region inputs outputs checked ctx call env args width permitted valid holds := by
    cases selected : first.infer Γ region inputs with
    | none =>
      simp only [selected, Option.orElse_none] at checked
      exact next.sound Γ region inputs outputs checked ctx call env args width permitted valid holds
    | some results =>
      simp only [selected, Option.orElse_some, Option.some.injEq] at checked
      subst outputs
      exact first.sound Γ region inputs results selected ctx call env args width permitted valid holds

#print axioms Regions.operations
#print axioms Regions.empty
#print axioms Regions.restrict
#print axioms Regions.orElse

end MerlinVerify.Check.Definedness.Contextual
