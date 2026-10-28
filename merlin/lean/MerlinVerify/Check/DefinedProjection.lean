/- Reuse a refined definedness book from ordinary initialization facts. Missing refinements
start empty and every returned refinement is erased. A regional rule can still derive bounds
inside its own body; no enclosing bound is invented or retained by this adapter. -/
import MerlinVerify.Check.DefinedRefinements

namespace MerlinVerify.Check.Definedness.Refined

open MerlinVerify

def emptyEnv {Info : Type} (M : Model Info) (env : Definedness.FEnv) : FEnv Info :=
  env.map fun (id, fact) => (id, emptyFact M fact)

theorem emptyEnv_lookup {Info : Type} (M : Model Info) (env : Definedness.FEnv) (id : ValueId) :
    (emptyEnv M env).lookup id = (env.lookup id).map (emptyFact M) := by
  induction env with
  | nil => rfl
  | cons binding rest ih =>
    obtain ⟨key, fact⟩ := binding
    simp only [emptyEnv, List.map_cons, List.lookup]
    split
    · rfl
    · exact ih

theorem EnvHolds.empty {Info : Type} (M : Model Info) {ib : Nat} {facts : Definedness.FEnv} {env : Env}
    (valid : Definedness.EnvHolds ib facts env) : EnvHolds M ib (emptyEnv M facts) env := by
  intro id fact found
  rw [emptyEnv_lookup] at found
  cases original : facts.lookup id with
  | none => simp [original] at found
  | some base =>
    simp only [original, Option.map_some, Option.some.injEq] at found
    subst fact
    obtain ⟨value, bound, holds⟩ := valid id base original
    exact ⟨value, bound, holds, M.empty_sound value⟩

def Book.erase {Info : Type} (M : Model Info) {sem : OpSem} {ib : Nat}
    (R : Book M sem ib) : Definedness.Book sem ib where
  rule facts op args := (R.rule (emptyEnv M facts) op (args.map (emptyFact M))).map (List.map (·.base))
  sound facts op args results checked ctx call env values width valid typed := by
    cases accepted : R.rule (emptyEnv M facts) op (args.map (emptyFact M)) with
    | none => simp [accepted] at checked
    | some refined =>
      simp only [accepted, Option.map_some, Option.some.injEq] at checked
      subst results
      obtain ⟨output, executed, holds⟩ := R.sound _ op _ refined accepted ctx call env values width
        (EnvHolds.empty M valid) (FactsHold.empty typed M)
      exact ⟨output, executed, holds.base⟩

#print axioms emptyEnv_lookup
#print axioms EnvHolds.empty
#print axioms Book.erase

end MerlinVerify.Check.Definedness.Refined
