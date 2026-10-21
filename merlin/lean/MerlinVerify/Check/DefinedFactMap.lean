/- Translate a proved operation book between refinement models. Projection must preserve
input evidence, and embedding must prove every reported output fact. -/
import MerlinVerify.Check.DefinedContext

namespace MerlinVerify.Check.Definedness.Refined

open MerlinVerify

structure FactMap {Source Target : Type} (source : Model Source) (target : Model Target) where
  project : Source → Target
  embed : Target → Source
  project_sound : ∀ info value, source.holds info value → target.holds (project info) value
  embed_sound : ∀ info value, target.holds info value → source.holds (embed info) value

variable {Source Target : Type} {source : Model Source} {target : Model Target}

def mapFact (map : Source → Target) (fact : Fact Source) : Fact Target := ⟨fact.base, map fact.info⟩

def mapEnv (map : Source → Target) (Γ : FEnv Source) : FEnv Target :=
  Γ.map fun (id, fact) => (id, mapFact map fact)

theorem mapEnv_lookup (map : Source → Target) (Γ : FEnv Source) (id : ValueId) :
    (mapEnv map Γ).lookup id = (Γ.lookup id).map (mapFact map) := by
  induction Γ with
  | nil => rfl
  | cons entry Γ ih =>
    obtain ⟨key, fact⟩ := entry
    simp only [mapEnv, List.map_cons, List.lookup]
    split
    · rfl
    · simpa only [mapEnv] using ih

theorem FactsHold.map {ib : Nat} (map : Source → Target)
    (sound : ∀ info value, source.holds info value → target.holds (map info) value)
    {values : List RVal} {facts : List (Fact Source)} (holds : FactsHold source ib values facts) :
    FactsHold target ib values (facts.map (mapFact map)) := by
  induction holds with
  | nil => exact .nil
  | cons head _ ih => exact .cons ⟨head.1, sound _ _ head.2⟩ ih

theorem EnvHolds.map {ib : Nat} (map : Source → Target)
    (sound : ∀ info value, source.holds info value → target.holds (map info) value)
    {Γ : FEnv Source} {env : Env} (holds : EnvHolds source ib Γ env) :
    EnvHolds target ib (mapEnv map Γ) env := by
  intro id fact found
  rw [mapEnv_lookup] at found
  cases original : Γ.lookup id with
  | none => simp [original] at found
  | some info =>
    simp only [original, Option.map_some, Option.some.injEq] at found
    subst fact
    obtain ⟨value, bound, valid⟩ := holds id info original
    exact ⟨value, bound, valid.1, sound _ _ valid.2⟩

def adaptBook {sem : OpSem} {ib : Nat} {allowed : Contextual.Conditions}
    (map : FactMap source target) (book : Contextual.Book target sem ib allowed) :
    Contextual.Book source sem ib allowed where
  rule Γ op facts := (book.rule (mapEnv map.project Γ) op (facts.map (mapFact map.project))).map
    (List.map (mapFact map.embed))
  sound Γ op facts results checked ctx call env args width permitted valid inputs := by
    cases selected : book.rule (mapEnv map.project Γ) op (facts.map (mapFact map.project)) with
    | none => simp [selected] at checked
    | some outputs =>
      simp only [selected, Option.map_some, Option.some.injEq] at checked
      subst results
      obtain ⟨values, executed, holds⟩ := book.sound _ op _ outputs selected ctx call env args width permitted
        (valid.map map.project map.project_sound) (inputs.map map.project map.project_sound)
      exact ⟨values, executed, holds.map map.embed map.embed_sound⟩

#print axioms mapEnv_lookup
#print axioms FactsHold.map
#print axioms EnvHolds.map
#print axioms adaptBook

end MerlinVerify.Check.Definedness.Refined
