/- Signed nonnegative intervals for scalars and every accessible tensor element.
Input intervals are proof obligations; this module does not infer them from metadata. -/
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Rules.FrontendMaskProperties

namespace MerlinVerify.Rules.FrontendTensorRanges
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness
open Sem.Ops IndexRanges

/-- The same interval applies to a scalar, or to all accessible tensor elements. -/
def Holds (r : Interval) : RVal → Prop
  | .scalar v => r.Holds (.scalar v)
  | .tensor t => ∀ p, Idx.InBounds t.shape p → r.Holds (.scalar (t.get p))

def model : Refined.Model (Option Interval) where
  holds info v := match info with | none => True | some r => Holds r v
  empty := none
  empty_sound _ := trivial

abbrev Fact := Refined.Fact (Option Interval)
abbrev FEnv := Refined.FEnv (Option Interval)

/-- Old scalar rules receive no tensor-element evidence they cannot interpret. -/
def scalarFact (f : Fact) : Fact :=
  ⟨f.base, match f.base.ty with | .scalar _ => f.info | .tensor _ _ => none⟩

def scalarEnv (Γ : FEnv) : FEnv := Γ.map fun (x, f) => (x, scalarFact f)

theorem to_scalar {ib : Nat} {v : RVal} {f : Fact}
    (h : Refined.Fact.Holds model ib v f) :
    Refined.Fact.Holds IndexRanges.model ib v (scalarFact f) := by
  refine ⟨h.1, ?_⟩
  cases ht : f.base.ty with
  | tensor sh e => simp [scalarFact, ht, IndexRanges.model]
  | scalar ty =>
    cases v with
    | tensor _ => have hv := h.1.1; rw [ht] at hv; cases hv
    | scalar v =>
      cases hi : f.info <;> simpa [scalarFact, ht, model, Holds, IndexRanges.model, hi] using h.2

theorem from_scalar {ib : Nat} {v : RVal} {f : Fact}
    (h : Refined.Fact.Holds IndexRanges.model ib v f) :
    Refined.Fact.Holds model ib v f := by
  refine ⟨h.1, ?_⟩
  cases hi : f.info with
  | none => trivial
  | some r =>
    cases v with
    | scalar v => simpa [model, Holds, IndexRanges.model, hi] using h.2
    | tensor t =>
      have hr : r.Holds (.tensor t) := by simpa [IndexRanges.model, hi] using h.2
      obtain ⟨z, hz, _⟩ := hr
      cases hz

theorem facts_to_scalar {ib : Nat} {vs : List RVal} {fs : List Fact}
    (h : Refined.FactsHold model ib vs fs) :
    Refined.FactsHold IndexRanges.model ib vs (fs.map scalarFact) := by
  induction h with
  | nil => exact .nil
  | cons hv _ ih => exact .cons (to_scalar hv) ih

theorem facts_from_scalar {ib : Nat} {vs : List RVal} {fs : List Fact}
    (h : Refined.FactsHold IndexRanges.model ib vs fs) :
    Refined.FactsHold model ib vs fs := by
  induction h with
  | nil => exact .nil
  | cons hv _ ih => exact .cons (from_scalar hv) ih

theorem scalarEnv_lookup (Γ : FEnv) (x : ValueId) :
    (scalarEnv Γ).lookup x = (Γ.lookup x).map scalarFact := by
  induction Γ with
  | nil => rfl
  | cons p Γ ih =>
    obtain ⟨y, f⟩ := p
    simp only [scalarEnv, List.map_cons, List.lookup]
    split
    · rfl
    · simpa only [scalarEnv] using ih

theorem env_to_scalar {ib : Nat} {Γ : FEnv} {env : Env}
    (h : Refined.EnvHolds model ib Γ env) :
    Refined.EnvHolds IndexRanges.model ib (scalarEnv Γ) env := by
  intro x f hx
  rw [scalarEnv_lookup] at hx
  cases hf : Γ.lookup x with
  | none => simp [hf] at hx
  | some g =>
    simp only [hf, Option.map_some, Option.some.injEq] at hx
    subst f
    obtain ⟨v, hv, hg⟩ := h x g hf
    exact ⟨v, hv, to_scalar hg⟩

/-- Reuse proved scalar rules through a checked model projection. -/
def scalarBook {ib : Nat} (old : Refined.Book IndexRanges.model opSem ib) :
    Refined.Book model opSem ib where
  rule Γ op fs := old.rule (scalarEnv Γ) op (fs.map scalarFact)
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    obtain ⟨vs, hv, hf⟩ := old.sound _ op _ rs h ctx call env args hib
      (env_to_scalar hΓ) (facts_to_scalar ha)
    exact ⟨vs, hv, facts_from_scalar hf⟩

theorem reshape_holds {r : Interval} {t : Tensor} (new : List Nat) (elt : Ty)
    (ht : Holds r (.tensor t)) (hs : shapeSize t.shape = shapeSize new) :
    Holds r (.tensor (ReshapeIndex.reshape t new elt)) := by
  exact FrontendMaskProperties.reshape_elements (P := fun v => r.Holds (.scalar v)) new elt ht hs

#print axioms scalarBook
end MerlinVerify.Rules.FrontendTensorRanges
