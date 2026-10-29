/- Complete straight-line entry-roster closure for configuration elision.
Reuses the existing exact whole-list checker, not the configuration hoist.
The module frame binds all declarations/outer structure through projection;
its string equality does not prove parser or hardware correspondence. -/
import Merlin.GraphStateElision

namespace Merlin.GraphState.Configuration.Entries
open StateEffects
open GraphState.Elision
set_option autoImplicit false

structure Entry where
  name : String
  header : String
  code : List Atom
  deriving DecidableEq, Repr

structure Program where
  frame : String
  entries : List Entry
  deriving DecidableEq, Repr

def checkEntries (fields : List Nat) : List Entry → List Entry → Bool
  | [], [] => true
  | before::rest, after::later =>
      (before.name == after.name) && (before.header == after.header) &&
      Elision.check fields before.code after.code && checkEntries fields rest later
  | _, _ => false

def check (fields : List Nat) (before after : Program) : Bool :=
  (before.frame == after.frame) && decide (before.entries.map (·.name)).Nodup &&
    checkEntries fields before.entries after.entries

def Related (fields : List Nat) (before after : Entry) : Prop :=
  before.name = after.name ∧ before.header = after.header ∧
    Elision.Preserves fields before.code after.code

inductive Corresponds (fields : List Nat) : List Entry → List Entry → Prop where
  | nil : Corresponds fields [] []
  | cons {first last : Entry} {rest later : List Entry} :
      Related fields first last → Corresponds fields rest later →
      Corresponds fields (first::rest) (last::later)

theorem checkEntries_sound {fields : List Nat} {before after : List Entry}
    (accepted : checkEntries fields before after = true) :
    Corresponds fields before after := by
  induction before generalizing after with
  | nil =>
    cases after with
    | nil => exact .nil
    | cons _ _ => cases accepted
  | cons first rest ih =>
    cases after with
    | nil => cases accepted
    | cons last later =>
      have parts : first.name = last.name ∧ first.header = last.header ∧
          Elision.check fields first.code last.code = true ∧
          checkEntries fields rest later = true := by
        simpa only [checkEntries, Bool.and_eq_true, beq_iff_eq, and_assoc] using accepted
      exact .cons ⟨parts.1, parts.2.1, Elision.check_sound parts.2.2.1⟩
        (ih parts.2.2.2)

def findEntry (name : String) : List Entry → Option Entry
  | [] => none
  | entry::rest => if entry.name = name then some entry else findEntry name rest

def invoke {Value Error : Type}
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (program : Program) (name : String) (initial : State Value) :
    Option (Except (Error × State Value) (State Value)) :=
  (findEntry name program.entries).map (fun entry => Elision.execute (run entry.name) entry.code initial)

def Admitted {Value Error : Type} (fields : List Nat) (literal : Int → Value)
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (program : Program) : Prop :=
  ∀ entry ∈ program.entries, ∀ atom ∈ entry.code,
    Elision.Admitted fields literal (run entry.name) atom

theorem related_invoke {fields : List Nat} {before after : List Entry}
    (related : Corresponds fields before after)
    {Value Error : Type} (literal : Int → Value)
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (models : ∀ entry ∈ before, ∀ atom ∈ entry.code,
      Elision.Admitted fields literal (run entry.name) atom)
    (name : String) (initial : State Value) :
    (findEntry name before).map (fun entry => Elision.execute (run entry.name) entry.code initial) =
      (findEntry name after).map (fun entry => Elision.execute (run entry.name) entry.code initial) := by
  induction related with
  | nil => rfl
  | @cons first last rest later head tail ih =>
    rcases head with ⟨names, _, preserved⟩
    by_cases chosen : first.name = name
    · have other : last.name = name := names.symm.trans chosen
      simp only [findEntry, chosen, other, ite_true, Option.map_some]
      simpa only [chosen] using congrArg some
        (preserved Value Error literal (run first.name) (models first (by simp)) initial)
    · have other : last.name ≠ name := by simpa only [←names] using chosen
      simp only [findEntry, chosen, other, ite_false]
      exact ih (fun entry member => models entry (by simp [member]))

/-- Every modeled defined entry, including missing-name results and complete
terminal errors/states, is preserved. Actual call-body/ISA interpretation remains
explicit in Admitted; declarations/headers are checked unchanged in the frame. -/
theorem check_sound {fields : List Nat} {before after : Program}
    (accepted : check fields before after = true)
    {Value Error : Type} (literal : Int → Value)
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (models : Admitted fields literal run before) (name : String) (initial : State Value) :
    invoke run before name initial = invoke run after name initial := by
  have parts : before.frame = after.frame ∧ (before.entries.map (·.name)).Nodup ∧
      checkEntries fields before.entries after.entries = true := by
    simpa only [check, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, and_assoc] using accepted
  exact related_invoke (checkEntries_sound parts.2.2) literal run models name initial

theorem checked_frame {fields : List Nat} {before after : Program}
    (accepted : check fields before after = true) : before.frame = after.frame := by
  have parts := accepted
  simp only [check, Bool.and_eq_true, beq_iff_eq] at parts
  exact parts.1.1

theorem related_headers {fields : List Nat} {before after : List Entry}
    (related : Corresponds fields before after) :
    before.map (fun entry => (entry.name,entry.header)) =
      after.map (fun entry => (entry.name,entry.header)) := by
  induction related with
  | nil => rfl
  | cons head _ ih => simp only [List.map_cons,head.1,head.2.1,ih]

theorem checked_headers {fields : List Nat} {before after : Program}
    (accepted : check fields before after = true) :
    before.entries.map (fun entry => (entry.name,entry.header)) =
      after.entries.map (fun entry => (entry.name,entry.header)) := by
  have parts := accepted
  simp only [check,Bool.and_eq_true] at parts
  exact related_headers (checkEntries_sound parts.2)

/-- The finite lookup relation exposes a proof for every defined entry, and
requires missing names to be missing on both sides. -/
def Selected (fields : List Nat) (before after : Option Entry) : Prop :=
  match before, after with
  | none, none => True
  | some first, some last => Related fields first last
  | _, _ => False

theorem related_selected {fields : List Nat} {before after : List Entry}
    (related : Corresponds fields before after) (name : String) :
    Selected fields (findEntry name before) (findEntry name after) := by
  induction related with
  | nil => trivial
  | @cons first last rest later head tail ih =>
    by_cases chosen : first.name = name
    · have other : last.name = name := head.1.symm.trans chosen
      simpa only [findEntry,chosen,other,ite_true,Selected] using head
    · have other : last.name ≠ name := by simpa only [←head.1] using chosen
      simpa only [findEntry,chosen,other,ite_false] using ih

theorem checked_selected {fields : List Nat} {before after : Program}
    (accepted : check fields before after = true) (name : String) :
    Selected fields (findEntry name before.entries) (findEntry name after.entries) := by
  have parts := accepted
  simp only [check,Bool.and_eq_true] at parts
  exact related_selected (checkEntries_sound parts.2) name

/-- Arbitrary observers of complete outcomes inherit equality; no observer may
inspect erased instruction events outside the admitted pure-write semantics. -/
theorem checked_observation {fields : List Nat} {before after : Program}
    (accepted : check fields before after = true)
    {Value Error Observation : Type} (literal : Int → Value)
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (models : Admitted fields literal run before) (name : String) (initial : State Value)
    (observe : Option (Except (Error × State Value) (State Value)) → Observation) :
    observe (invoke run before name initial) = observe (invoke run after name initial) :=
  congrArg observe (check_sound accepted literal run models name initial)

end Merlin.GraphState.Configuration.Entries
