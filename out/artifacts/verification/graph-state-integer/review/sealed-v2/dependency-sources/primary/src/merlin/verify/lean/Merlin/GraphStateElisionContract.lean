/- Complete Kernel configuration-elision domain. Initial configuration is
arbitrary. Qualified removable setters are TOTAL exact persistent overwrites
with no other observation/failure effect. Other retained instructions may be
deterministic partial steps; equality preserves BOTH failure and successful
complete state, including the terminal error and state on failure. No source-success/initialization/final-equality assumption is
used as a replacement for the known-field invariant. Actual admission and
Kernel-to-model correspondence remain separate explicit host premises. -/
import Merlin.GraphStateConfiguration
import Init.Data.List.Find

namespace Merlin.GraphState.Elision
open StateEffects
set_option autoImplicit false

abbrev Assignments := List (Nat × Int)
abbrev Known := Nat → Option Int
def empty : Known := fun _ => none

structure Effect where
  reads : Assignments
  writes : Assignments
  clobbers : List Nat
  ordered : Bool
  pureWrite : Bool
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

structure Atom where
  id : Nat
  token : String
  effect : Option Effect
  eraseAllowed : Bool
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

def assign {Value : Type} (literal : Int → Value) (writes : Assignments)
    (state : State Value) : State Value := fun key =>
  match writes.lookup key with
  | none => state key
  | some value => literal value

def advance (known : Known) : Option Effect → Known
  | none => empty
  | some effect => if effect.ordered then fun key =>
      match effect.writes.lookup key with
      | some value => some value
      | none => if key ∈ effect.clobbers then none else known key
    else empty

/-- Known means a fact justified by successful retained/identity-prefix steps,
not a source precondition that the requested value was initially installed. -/
def Knowledge {Value : Type} (fields : List Nat) (literal : Int → Value)
    (known : Known) (state : State Value) : Prop :=
  ∀ key value, known key = some value → key ∈ fields ∧ state key = literal value

theorem initial_unknown {Value : Type} (fields : List Nat) (literal : Int → Value) (state : State Value) :
    Knowledge fields literal empty state := by intro key value impossible; cases impossible

def validEffect (fields : List Nat) (effect : Effect) : Bool :=
  decide ((∀ pair ∈ effect.reads, pair.1 ∈ fields) ∧
    (∀ pair ∈ effect.writes, pair.1 ∈ fields) ∧
    (∀ key ∈ effect.clobbers, key ∈ fields) ∧
    (effect.reads.map (·.1)).Nodup ∧ (effect.writes.map (·.1)).Nodup ∧ effect.clobbers.Nodup) &&
  if effect.pureWrite then effect.ordered && (effect.reads == []) && (effect.clobbers == []) && (effect.writes != [])
  else true

def validAtom (fields : List Nat) (atom : Atom) : Bool :=
  match atom.effect with | none => true | some effect => validEffect fields effect

def pure (atom : Atom) (effect : Effect) : Bool := atom.eraseAllowed && effect.ordered && effect.pureWrite

def redundant (known : Known) (effect : Effect) : Bool :=
  effect.writes.all (fun pair => known pair.1 == some pair.2)

def compatible (known : Known) : Option Effect → Bool
  | none => true
  | some effect => if effect.ordered then effect.reads.all (fun pair =>
      match known pair.1 with | none => true | some value => value == pair.2)
    else true

def established (known : Known) (reads : Assignments) : Bool :=
  reads.all (fun pair => known pair.1 == some pair.2)

theorem lookup_member {assignments : Assignments} {key : Nat} {value : Int}
    (found : assignments.lookup key = some value) : (key,value) ∈ assignments := by
  induction assignments with
  | nil => cases found
  | cons pair rest ih =>
    rcases pair with ⟨index,assigned⟩
    simp only [List.lookup_cons] at found
    split at found
    · next same =>
      have keyEqual : key = index := by simpa using same
      have valueEqual : assigned = value := by simpa using found
      exact List.mem_cons.mpr (.inl (Prod.ext keyEqual valueEqual.symm))
    · exact List.mem_cons.mpr (.inr (ih found))

theorem redundant_values {known : Known} {effect : Effect}
    (checked : redundant known effect = true) :
    ∀ pair ∈ effect.writes, known pair.1 = some pair.2 := by
  simpa only [redundant,List.all_eq_true,beq_iff_eq] using checked

theorem established_reads {Value : Type} {fields : List Nat} {literal : Int → Value}
    {known : Known} {state : State Value} {reads : Assignments}
    (knowledge : Knowledge fields literal known state) (checked : established known reads = true) :
    ∀ pair ∈ reads, state pair.1 = literal pair.2 := by
  have facts : ∀ pair ∈ reads, known pair.1 = some pair.2 := by
    simpa only [established,List.all_eq_true,beq_iff_eq] using checked
  intro pair member
  exact (knowledge pair.1 pair.2 (facts pair member)).2

theorem redundant_assign_identity {Value : Type} {fields : List Nat} {literal : Int → Value}
    {known : Known} {state : State Value} {effect : Effect}
    (knowledge : Knowledge fields literal known state) (checked : redundant known effect = true) :
    assign literal effect.writes state = state := by
  funext key
  cases found : effect.writes.lookup key with
  | none => simp only [assign,found]
  | some value =>
    have installed := (knowledge key value (redundant_values checked (key,value) (lookup_member found))).2
    simpa only [assign,found] using installed.symm

/-- Reuse the existing exact overwrite meaning, without assuming that an
instruction name/flag establishes this operational equation. -/
theorem assign_as_put {Value : Type} (literal : Int → Value) (writes : Assignments) (state : State Value) :
    assign literal writes state = Configuration.put (writes.map (·.1))
      (fun key => literal ((writes.lookup key).getD 0)) state := by
  funext key
  cases found : writes.lookup key with
  | none =>
    have outside : key ∉ writes.map (·.1) := by
      intro inside
      obtain ⟨pair,member,equal⟩ := List.mem_map.mp inside
      have different : key ≠ pair.1 := by simpa only [bne_iff_ne] using List.lookup_eq_none_iff.mp found pair member
      exact different equal.symm
    simp only [assign,Configuration.put,found,ite_eq_right outside]
  | some value =>
    have inside : key ∈ writes.map (·.1) := List.mem_map.mpr ⟨(key,value),lookup_member found,rfl⟩
    simp only [assign,Configuration.put,found,ite_eq_left inside,Option.getD_some]

theorem assign_idempotent {Value : Type} (literal : Int → Value) (writes : Assignments) (state : State Value) :
    assign literal writes (assign literal writes state) = assign literal writes state := by
  simp only [assign_as_put]
  exact Configuration.put_idempotent _ _ _

theorem advance_redundant {known : Known} {effect : Effect}
    (ordered : effect.ordered = true) (noClobber : effect.clobbers = [])
    (same : redundant known effect = true) : advance known (some effect) = known := by
  funext key
  simp only [advance,ordered,ite_true]
  cases found : effect.writes.lookup key with
  | none => simp only [noClobber,List.not_mem_nil,ite_false]
  | some value =>
    simpa only [found] using (redundant_values same (key,value) (lookup_member found)).symm

/-- Arbitrary retained calls can fail and can change nonconfiguration state.
Only successful ordered effects constrain the tracked configuration fields.
The qualified pure law is the explicit stronger exact TOTAL overwrite meaning;
it cannot be inferred from a metadata flag or matching register value. -/
def Admitted {Value Error : Type} (fields : List Nat) (literal : Int → Value)
    (run : Nat → State Value → Except (Error × State Value) (State Value)) (atom : Atom) : Prop :=
  match atom.effect with
  | none => True
  | some effect =>
    (effect.ordered = true → ∀ initial final, run atom.id initial = .ok final →
      (∀ key value, effect.writes.lookup key = some value → final key = literal value) ∧
      (∀ key, key ∈ fields → effect.writes.lookup key = none → key ∉ effect.clobbers → final key = initial key) ∧
      (∀ pair ∈ effect.reads, initial pair.1 = literal pair.2)) ∧
    (pure atom effect = true → ∀ initial, run atom.id initial = .ok (assign literal effect.writes initial))

theorem advance_knowledge {Value Error : Type} {fields : List Nat} {literal : Int → Value}
    {known : Known} {initial final : State Value} {run : Nat → State Value → Except (Error × State Value) (State Value)} {atom : Atom}
    (valid : validAtom fields atom = true) (model : Admitted fields literal run atom)
    (knowledge : Knowledge fields literal known initial) (success : run atom.id initial = .ok final) :
    Knowledge fields literal (advance known atom.effect) final := by
  cases found : atom.effect with
  | none => simpa only [advance,found] using initial_unknown fields literal final
  | some effect =>
    by_cases ordered : effect.ordered = true
    · have facts := model
      simp only [Admitted,found] at facts
      have transition := facts.1 ordered initial final success
      have validity : (∀ pair ∈ effect.reads, pair.1 ∈ fields) ∧
          (∀ pair ∈ effect.writes, pair.1 ∈ fields) ∧
          (∀ key ∈ effect.clobbers, key ∈ fields) ∧
          (effect.reads.map (·.1)).Nodup ∧ (effect.writes.map (·.1)).Nodup ∧ effect.clobbers.Nodup := by
        have parts := valid
        simp only [validAtom,found,validEffect,Bool.and_eq_true] at parts
        exact of_decide_eq_true parts.1
      intro key value installed
      simp only [advance,ordered,ite_true] at installed
      cases written : effect.writes.lookup key with
      | some actual =>
        simp only [written,Option.some.injEq] at installed
        subst value
        exact ⟨validity.2.1 (key,actual) (lookup_member written),transition.1 key actual written⟩
      | none =>
        simp only [written] at installed
        split at installed
        · cases installed
        · next outside =>
          have old := knowledge key value installed
          exact ⟨old.1,(transition.2.1 key old.1 written outside).trans old.2⟩
    · have notOrdered : effect.ordered = false := Bool.eq_false_iff.mpr ordered
      simpa only [advance,found,notOrdered,Bool.false_eq_true,ite_false] using initial_unknown fields literal final

theorem omitted_identity {Value Error : Type} {fields : List Nat} {literal : Int → Value}
    {known : Known} {state : State Value} {run : Nat → State Value → Except (Error × State Value) (State Value)} {atom : Atom} {effect : Effect}
    (found : atom.effect = some effect) (model : Admitted fields literal run atom)
    (pureChecked : pure atom effect = true) (same : redundant known effect = true)
    (knowledge : Knowledge fields literal known state) : run atom.id state = .ok state := by
  have facts := model
  simp only [Admitted,found] at facts
  rw [facts.2 pureChecked state,redundant_assign_identity knowledge same]

def execute {Value Error : Type} (run : Nat → State Value → Except (Error × State Value) (State Value)) :
    List Atom → State Value → Except (Error × State Value) (State Value)
  | [],state => .ok state
  | atom::rest,state => (run atom.id state).bind (execute run rest)

/-- The full intended claim: one complete actual Kernel before/after boundary
has equal possible failure or complete success state, for arbitrary initial
configuration and every admitted common instruction interpretation. -/
def Preserves (fields : List Nat) (before after : List Atom) : Prop :=
  ∀ (Value Error : Type) (literal : Int → Value) (run : Nat → State Value → Except (Error × State Value) (State Value)),
    (∀ atom ∈ before, Admitted fields literal run atom) →
    ∀ initial, execute run before initial = execute run after initial

#print axioms initial_unknown
#print axioms redundant_assign_identity
#print axioms advance_knowledge
#print axioms omitted_identity
end Merlin.GraphState.Elision
