/- Sound whole-program configuration elimination with a proved known-state
invariant. Exact whole-call omissions only; arbitrary initial configuration;
retained deterministic partial calls preserve terminal error/state as well as
successful state. Actual Kernel projection/admission remains an explicit host
boundary. This checker does not assume source success or initialized fields. -/
import Merlin.GraphStateElisionContract

namespace Merlin.GraphState.Elision
open StateEffects
set_option autoImplicit false

def canOmit (known : Known) (atom : Atom) : Bool :=
  match atom.effect with
  | none => false
  | some effect => pure atom effect && redundant known effect

def replay : Known → List Atom → List Atom → Bool
  | _,[],after => after == []
  | known,atom::rest,after =>
    compatible known atom.effect &&
      match after with
      | [] => canOmit known atom && replay (advance known atom.effect) rest []
      | actual::later => if atom == actual then replay (advance known atom.effect) rest later
        else canOmit known atom && replay (advance known atom.effect) rest after

def check (fields : List Nat) (before after : List Atom) : Bool :=
  decide fields.Nodup && before.all (validAtom fields) &&
    ((before.map (·.id)) == List.range before.length) && replay empty before after

theorem replay_sound {Value Error : Type} (fields : List Nat) (literal : Int → Value)
    (run : Nat → State Value → Except (Error × State Value) (State Value)) :
    ∀ before after known initial,
      (∀ atom ∈ before, validAtom fields atom = true) →
      (∀ atom ∈ before, Admitted fields literal run atom) →
      Knowledge fields literal known initial → replay known before after = true →
      execute run before initial = execute run after initial := by
  intro before
  induction before with
  | nil =>
    intro after known initial _ _ _ accepted
    have emptyAfter : after = [] := by simpa only [replay,beq_iff_eq] using accepted
    subst after
    rfl
  | cons atom rest ih =>
    intro after known initial valid models knowledge accepted
    have headValid := valid atom (by simp)
    have headModel := models atom (by simp)
    have tailValid : ∀ item ∈ rest, validAtom fields item = true := fun item member => valid item (by simp [member])
    have tailModel : ∀ item ∈ rest, Admitted fields literal run item := fun item member => models item (by simp [member])
    have eraseCase : ∀ after, canOmit known atom = true → replay (advance known atom.effect) rest after = true →
        execute run (atom::rest) initial = execute run after initial := by
      intro after canDrop remainder
      cases found : atom.effect with
      | none => simp [canOmit,found] at canDrop
      | some effect =>
        have parts : pure atom effect = true ∧ redundant known effect = true := by
          simpa only [canOmit,found,Bool.and_eq_true] using canDrop
        have identity := omitted_identity found headModel parts.1 parts.2 knowledge
        have next := advance_knowledge headValid headModel knowledge identity
        have equal := ih after (advance known atom.effect) initial tailValid tailModel next remainder
        simpa only [execute,identity,Except.bind] using equal
    cases after with
    | nil =>
      have parts : compatible known atom.effect = true ∧ canOmit known atom = true ∧
          replay (advance known atom.effect) rest [] = true := by
        simpa only [replay,Bool.and_eq_true] using accepted
      exact eraseCase [] parts.2.1 parts.2.2
    | cons actual later =>
      simp only [replay,Bool.and_eq_true] at accepted
      split at accepted
      · next same =>
        have equal : atom = actual := by simpa using same
        subst actual
        cases stepped : run atom.id initial with
        | error failure => simp only [execute,stepped,Except.bind]
        | ok final =>
          have next := advance_knowledge headValid headModel knowledge stepped
          have preserved := ih later (advance known atom.effect) final tailValid tailModel next accepted.2
          simpa only [execute,stepped,Except.bind] using preserved
      · have parts : canOmit known atom = true ∧ replay (advance known atom.effect) rest (actual::later) = true := by
          simpa only [Bool.and_eq_true] using accepted.2
        exact eraseCase (actual::later) parts.1 parts.2

/-- Acceptance proves equal complete outcome, for every arbitrary initial
configuration and admitted common call interpretation. Source success is a
consequence only in an independently proved concrete model/domain. -/
theorem check_sound {fields : List Nat} {before after : List Atom}
    (accepted : check fields before after = true) : Preserves fields before after := by
  have parts := accepted
  simp only [check,Bool.and_eq_true] at parts
  have valid : ∀ atom ∈ before, validAtom fields atom = true := by
    simpa only [List.all_eq_true] using parts.1.1.2
  intro Value Error literal run models initial
  exact replay_sound fields literal run before after empty initial valid models (initial_unknown fields literal initial) parts.2

theorem execute_append {Value Error : Type}
    (run : Nat → State Value → Except (Error × State Value) (State Value))
    (before after : List Atom) (initial : State Value) :
    execute run (before++after) initial = (execute run before initial).bind (execute run after) := by
  induction before generalizing initial with
  | nil => rfl
  | cons atom rest ih =>
    cases stepped : run atom.id initial with
    | error failure => simp [execute,stepped,Except.bind]
    | ok final => simpa only [List.cons_append,execute,stepped,Except.bind] using ih final

/-- Supported partial surrounding programs keep all observations, including
their state/error on failure, because the complete checked outcome is equal. -/
theorem context_preserved {fields : List Nat} {before after : List Atom}
    (accepted : check fields before after = true) {Value Error : Type} (literal : Int → Value)
    (run : Nat → State Value → Except (Error × State Value) (State Value))
    (models : ∀ atom ∈ before, Admitted fields literal run atom)
    (earlier later : List Atom) (initial : State Value) :
    execute run (earlier++before++later) initial = execute run (earlier++after++later) initial := by
  simp only [execute_append]
  cases outcome : execute run earlier initial with
  | error failure => rfl
  | ok state =>
    simp only [Except.bind]
    rw [check_sound accepted Value Error literal run models state]

#print axioms replay_sound
#print axioms check_sound
#print axioms context_preserved
end Merlin.GraphState.Elision
