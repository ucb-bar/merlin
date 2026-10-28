/- Refinement transport may inspect a fact's proved type. For example, an elementwise
tensor interval cannot be supplied to a rule expecting a scalar interval. -/
import MerlinVerify.Check.DefinedFactMap
import MerlinVerify.Check.DefinedRegions

namespace MerlinVerify.Check.Definedness.Refined

open MerlinVerify

structure FactTransport {Source Target : Type} (source : Model Source) (target : Model Target) (ib : Nat) where
  map : Fact Source → Fact Target
  sound : ∀ fact value, fact.Holds source ib value → (map fact).Holds target ib value

variable {Source Target : Type} {source : Model Source} {target : Model Target} {ib : Nat}

def FactTransport.env (transport : FactTransport source target ib) (Γ : FEnv Source) : FEnv Target :=
  Γ.map fun (id, fact) => (id, transport.map fact)

theorem FactTransport.lookup (transport : FactTransport source target ib) (Γ : FEnv Source) (id : ValueId) :
    (transport.env Γ).lookup id = (Γ.lookup id).map transport.map := by
  induction Γ with
  | nil => rfl
  | cons entry Γ ih =>
    obtain ⟨key, fact⟩ := entry
    simp only [FactTransport.env, List.map_cons, List.lookup]
    split
    · rfl
    · simpa only [FactTransport.env] using ih

theorem FactTransport.facts (transport : FactTransport source target ib)
    {values : List RVal} {facts : List (Fact Source)} (holds : FactsHold source ib values facts) :
    FactsHold target ib values (facts.map transport.map) := by
  induction holds with
  | nil => exact .nil
  | cons head _ ih => exact .cons (transport.sound _ _ head) ih

theorem FactTransport.environment (transport : FactTransport source target ib)
    {Γ : FEnv Source} {env : Env} (holds : EnvHolds source ib Γ env) :
    EnvHolds target ib (transport.env Γ) env := by
  intro id fact found
  rw [transport.lookup] at found
  cases original : Γ.lookup id with
  | none => simp [original] at found
  | some info =>
    simp only [original, Option.map_some, Option.some.injEq] at found
    subst fact
    obtain ⟨value, bound, valid⟩ := holds id info original
    exact ⟨value, bound, transport.sound _ _ valid⟩

def transportBook {sem : OpSem} {allowed : Contextual.Conditions}
    (project : FactTransport source target ib) (embed : FactTransport target source ib)
    (book : Contextual.Book target sem ib allowed) : Contextual.Book source sem ib allowed where
  rule Γ op facts := (book.rule (project.env Γ) op (facts.map project.map)).map (List.map embed.map)
  sound Γ op facts results checked ctx call env args width permitted valid inputs := by
    cases selected : book.rule (project.env Γ) op (facts.map project.map) with
    | none => simp [selected] at checked
    | some outputs =>
      simp only [selected, Option.map_some, Option.some.injEq] at checked
      subst results
      obtain ⟨values, executed, holds⟩ := book.sound _ op _ outputs selected ctx call env args width permitted
        (project.environment valid) (project.facts inputs)
      exact ⟨values, executed, embed.facts holds⟩

/-- Whole-region contracts can be reused with the same proved projection/embedding as
operation rules, including captured bindings and their first-binding lookup semantics. -/
def adaptRegions {sem : OpSem} {allowed : Contextual.Conditions}
    (map : FactMap source target) (regions : Contextual.Regions target sem ib allowed) :
    Contextual.Regions source sem ib allowed where
  infer Γ body facts := (regions.infer (mapEnv map.project Γ) body (facts.map (mapFact map.project))).map
    (List.map (mapFact map.embed))
  sound Γ body facts results checked ctx call env args width permitted valid inputs := by
    cases selected : regions.infer (mapEnv map.project Γ) body (facts.map (mapFact map.project)) with
    | none => simp [selected] at checked
    | some outputs =>
      simp only [selected, Option.map_some, Option.some.injEq] at checked
      subst results
      obtain ⟨values, executed, holds, defined⟩ := regions.sound _ body _ outputs selected ctx call env args
        width permitted (valid.map map.project map.project_sound) (inputs.map map.project map.project_sound)
      exact ⟨values, executed, holds.map map.embed map.embed_sound, defined⟩

#print axioms FactTransport.lookup
#print axioms FactTransport.facts
#print axioms FactTransport.environment
#print axioms transportBook
#print axioms adaptRegions

end MerlinVerify.Check.Definedness.Refined
