/- Complete ordinary Code decomposition into ordered top-level steps. Region
steps still execute their actual lexical body, not an admitted loop summary.
This is syntax-length induction; runtime tripcounts are never expanded. -/
import Merlin.GraphLoopsShapeEpochScope

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem eval_aliases {Value World : Type} (meaning : Interpretation Value World) (code : Code)
    (first second : Aliases) (env : Env Value) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false first code env before outputs after ↔ Eval meaning false second code env before outputs after := by
  induction code generalizing first second env before outputs after with
  | returned values => rfl
  | cast origin result source target rest induction =>
    simp only [Eval, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr; intro value
    exact and_congr_right (fun _ => induction _ _ _ _ _ _)
  | dimension buffer axis result type axisIndex value rest induction =>
    simp only [Eval, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr; intro actual
    exact and_congr_right (fun _ => induction _ _ _ _ _ _)
  | constant result value rest induction => exact induction _ _ _ _ _ _
  | action key operands results rest induction =>
    simp only [Eval, readArguments, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr; intro values
    apply exists_congr; intro middle
    apply and_congr_right; intro _
    exact and_congr_right (fun _ => induction _ _ _ _ _ _)
  | region key operands parameters results body rest bodyInduction restInduction =>
    have bodySame : (fun inputs initial returned final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases first parameters) body (assign env parameters inputs) initial returned final) =
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases second parameters) body (assign env parameters inputs) initial returned final) := by
      funext inputs initial returned final
      apply propext
      exact and_congr_right (fun _ => bodyInduction _ _ _ _ _ _)
    simp only [Eval, readArguments, Bool.false_eq_true, ↓reduceIte] at bodySame ⊢
    rw [bodySame]
    apply exists_congr; intro values
    apply exists_congr; intro middle
    apply and_congr_right; intro _
    exact and_congr_right (fun _ => restInduction _ _ _ _ _ _)

inductive Atom where
  | constant (result : Binding) (value : Nat)
  | action (key : String) (operands results : List Binding)
  | region (key : String) (operands parameters results : List Binding) (body : Code)
  deriving DecidableEq, Repr

def Atom.code : Atom → Code → Code
  | .constant result value, rest => .constant result value rest
  | .action key operands results, rest => .action key operands results rest
  | .region key operands parameters results body, rest => .region key operands parameters results body rest

def Atom.step {Value World : Type} (meaning : Interpretation Value World) (atom : Atom)
    (env : Env Value) (before : World) (next : Env Value) (after : World) : Prop :=
  match atom with
  | .constant result value => next = put env result.name (.index value) ∧ after = before
  | .action key operands results => ∃ values,
      meaning.action key (readArguments false [] env operands) before values after ∧
      ValuesHold results values ∧ next = assign env results values
  | .region key operands parameters results body => ∃ values,
      meaning.region key (readArguments false [] env operands)
        (fun inputs initial outputs final => ValuesHold parameters inputs ∧
          Eval meaning false [] body (assign env parameters inputs) initial outputs final)
        before values after ∧ ValuesHold results values ∧ next = assign env results values

theorem Atom.eval_suffix {Value World : Type} (meaning : Interpretation Value World) (atom : Atom) (rest : Code)
    (env : Env Value) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false [] (atom.code rest) env before outputs after ↔
      ∃ next middle, atom.step meaning env before next middle ∧ Eval meaning false [] rest next middle outputs after := by
  cases atom with
  | constant result value =>
    rw [Atom.code, Eval, eval_aliases meaning rest _ []]
    constructor
    · intro run; exact ⟨_, before, ⟨rfl, rfl⟩, run⟩
    · rintro ⟨next, middle, ⟨rfl, rfl⟩, run⟩; exact run
  | action key operands results =>
    simp only [Atom.code, Eval, Atom.step]
    constructor
    · rintro ⟨values, middle, first, typed, suffix⟩
      exact ⟨_, middle, ⟨values, first, typed, rfl⟩,
        (eval_aliases meaning rest _ [] _ _ _ _).mp suffix⟩
    · rintro ⟨next, middle, ⟨values, first, typed, rfl⟩, suffix⟩
      exact ⟨values, middle, first, typed, (eval_aliases meaning rest _ [] _ _ _ _).mpr suffix⟩
  | region key operands parameters results body =>
    have bodySame : (fun inputs initial returned final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases [] parameters) body (assign env parameters inputs) initial returned final) =
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
        Eval meaning false [] body (assign env parameters inputs) initial returned final) := by
      funext inputs initial returned final
      apply propext
      exact and_congr_right (fun _ => eval_aliases meaning body _ [] _ _ _ _)
    simp only [Atom.code, Eval, Atom.step, bodySame]
    constructor
    · rintro ⟨values, middle, first, typed, suffix⟩
      exact ⟨_, middle, ⟨values, first, typed, rfl⟩,
        (eval_aliases meaning rest _ [] _ _ _ _).mp suffix⟩
    · rintro ⟨next, middle, ⟨values, first, typed, rfl⟩, suffix⟩
      exact ⟨values, middle, first, typed, (eval_aliases meaning rest _ [] _ _ _ _).mpr suffix⟩

def sequenceCode (atoms : List Atom) (returned : List Binding) : Code :=
  atoms.foldr Atom.code (.returned returned)

inductive SequenceRun {Value World : Type} (meaning : Interpretation Value World) :
    List Atom → Env Value → World → Env Value → World → Prop where
  | done (env : Env Value) (before : World) : SequenceRun meaning [] env before env before
  | more {atom : Atom} {rest : List Atom} {env middleEnv finalEnv : Env Value} {before middle after : World}
      (first : atom.step meaning env before middleEnv middle)
      (suffix : SequenceRun meaning rest middleEnv middle finalEnv after) :
      SequenceRun meaning (atom :: rest) env before finalEnv after

/-- Exact complete Code execution and all returned values/world are preserved.
The old epoch execution is not assumed by this independent decomposition. -/
theorem sequence_complete {Value World : Type} (meaning : Interpretation Value World) (atoms : List Atom)
    (returned : List Binding) (env : Env Value) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false [] (sequenceCode atoms returned) env before outputs after ↔
      ∃ finalEnv, SequenceRun meaning atoms env before finalEnv after ∧
        outputs = readArguments false [] finalEnv returned := by
  induction atoms generalizing env before with
  | nil =>
    constructor
    · rintro ⟨outputs, rfl⟩; exact ⟨env, .done env _, outputs⟩
    · rintro ⟨finalEnv, run, outputs⟩; cases run; exact ⟨outputs, rfl⟩
  | cons atom rest induction =>
    change Eval meaning false [] (atom.code (sequenceCode rest returned)) env before outputs after ↔ _
    rw [atom.eval_suffix]
    constructor
    · rintro ⟨middleEnv, middle, first, suffix⟩
      obtain ⟨finalEnv, suffix, outputs⟩ := (induction middleEnv middle).mp suffix
      exact ⟨finalEnv, .more first suffix, outputs⟩
    · rintro ⟨finalEnv, run, outputs⟩
      cases run with
      | more first suffix => exact ⟨_, _, first, (induction _ _).mpr ⟨finalEnv, suffix, outputs⟩⟩

end Merlin.GraphLoopsShape.Epoch
