/- Initial scoped counted-loop language and checked body effect frames.
Completion fields are retained explicitly and must be empty in the checked
source/target templates. No asynchronous/carried-token interpretation is added. -/
import Merlin.GraphStateConfigurationPartialHoist

namespace Merlin.GraphState.Configuration.PartialHoist
open StateEffects
set_option autoImplicit false

structure Loop where
  var : String
  extent : Nat
  body : List Nat
  metadata : String
  iterArgs : List (String × String)
  yields : List String
  results : List String
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

inductive Command where
  | call (id : Nat)
  | loop (descriptor : Loop)
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

structure Program where
  frame : String
  atoms : List Elision.Atom
  commands : List Command
  deriving DecidableEq, Repr

structure Move where
  earlier : List Nat
  setup : Nat
  body : List Nat
  later : List Nat
  var : String
  extent : Nat
  metadata : String
  deriving DecidableEq, Repr

def source (move : Move) : List Command :=
  move.earlier.map .call ++ [.loop ⟨move.var,move.extent,move.setup::move.body,move.metadata,[],[],[]⟩] ++
    move.later.map .call

def target (move : Move) : List Command :=
  move.earlier.map .call ++ [.call move.setup,.loop ⟨move.var,move.extent,move.body,move.metadata,[],[],[]⟩] ++
    move.later.map .call

def calls {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (index : Nat) : List Nat → State Value → Outcome Value Error
  | [],state => .ok state
  | id::rest,state => (run id index state).bind (calls run index rest)

def Models {Value Error : Type} (fields : List Nat) (literal : Int → Value)
    (atoms : List Elision.Atom) (run : Nat → Nat → State Value → Outcome Value Error) : Prop :=
  ∀ index id atom, atoms[id]? = some atom →
    Elision.Admitted fields literal (fun _ state => run id index state) atom

def safeBody (atoms : List Elision.Atom) (setupFields : List Nat) : List Nat → Bool
  | [] => true
  | id::rest => match atoms[id]? with
    | none => false
    | some atom => match atom.effect with
      | none => false
      | some effect => effect.ordered &&
          decide (∀ key ∈ setupFields, key ∉ effect.writes.map (·.1) ∧ key ∉ effect.clobbers) &&
          safeBody atoms setupFields rest

theorem safeBody_frame {Value Error : Type} (fields : List Nat) (literal : Int → Value)
    (atoms : List Elision.Atom) (run : Nat → Nat → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (setupFields : List Nat)
    (subset : ∀ key ∈ setupFields, key ∈ fields) (index : Nat) :
    ∀ ids, safeBody atoms setupFields ids = true → Frame setupFields (calls run index ids) := by
  intro ids
  induction ids with
  | nil =>
    intro _ initial final success key _
    have equal : initial = final := Except.ok.inj success
    exact congrFun equal.symm key
  | cons id rest ih =>
    intro checked
    cases found : atoms[id]? with
    | none => simp [safeBody,found] at checked
    | some atom =>
      cases effectFound : atom.effect with
      | none => simp [safeBody,found,effectFound] at checked
      | some effect =>
        have parts : effect.ordered = true ∧
            (∀ key ∈ setupFields, key ∉ effect.writes.map (·.1) ∧ key ∉ effect.clobbers) ∧
            safeBody atoms setupFields rest = true := by
          simpa only [safeBody,found,effectFound,Bool.and_eq_true,decide_eq_true_eq,and_assoc] using checked
        have admitted := models index id atom found
        simp only [Elision.Admitted,effectFound] at admitted
        intro initial final success key member
        cases step : run id index initial with
        | error failure => simp only [calls,step,Except.bind] at success; cases success
        | ok middle =>
          have tail : calls run index rest middle = .ok final := by
            simpa only [calls,step,Except.bind] using success
          have frame := (admitted.1 parts.1 initial middle step).2.1
          have notWritten : effect.writes.lookup key = none := by
            cases lookup : effect.writes.lookup key with
            | none => rfl
            | some value =>
              have written : key ∈ effect.writes.map (·.1) :=
                List.mem_map.mpr ⟨(key,value),Elision.lookup_member lookup,rfl⟩
              exact False.elim ((parts.2.1 key member).1 written)
          exact (ih parts.2.2 middle final tail key member).trans
            (frame key (subset key member) notWritten (parts.2.1 key member).2)

/-- A loop with the exact total pure setup call is represented by repeatSetup;
this equation is derived constructorwise even when a retained body call fails. -/
theorem loop_setup {Value Error : Type} (literal : Int → Value) (writes : Elision.Assignments)
    (run : Nat → Nat → State Value → Outcome Value Error) (setup : Nat) (body : List Nat)
    (binding : ∀ index state, run setup index state = .ok (Elision.assign literal writes state)) :
    ∀ count index state,
      repeatBody (fun index => calls run index (setup::body)) count index state =
        repeatSetup (Elision.assign literal writes) (fun index => calls run index body) count index state := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index state
    simp only [repeatBody,repeatSetup,calls,binding,Except.bind]
    cases step : calls run index body (Elision.assign literal writes state) with
    | error failure => rfl
    | ok final => simpa only [calls,binding,Except.bind] using ih (index+1) final

end Merlin.GraphState.Configuration.PartialHoist
