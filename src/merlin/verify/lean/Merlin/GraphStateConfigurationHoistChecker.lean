/- Complete scoped actual-kernel hoist checker. The admitted shape has exactly
one positive loop with explicit empty completion fields; arbitrary behavior for
other loops is kept as a parameter and is never used by an accepted rewrite. -/
import Merlin.GraphStateConfigurationHoistModel

namespace Merlin.GraphState.Configuration.PartialHoist
open StateEffects
set_option autoImplicit false

def ready (fields : List Nat) (atoms : List Elision.Atom) (move : Move) : Bool :=
  match atoms[move.setup]? with
  | none => false
  | some atom => match atom.effect with
    | none => false
    | some effect => Elision.validEffect fields effect && Elision.pure atom effect &&
        safeBody atoms (effect.writes.map (·.1)) move.body

def check (fields : List Nat) (before after : Program) (move : Move) : Bool :=
  decide (fields.Nodup ∧ before.frame = after.frame ∧ before.atoms = after.atoms ∧
    before.commands = source move ∧ after.commands = target move ∧
    move.extent > 0 ∧ move.body ≠ [] ∧
    before.atoms.map (·.id) = List.range before.atoms.length ∧
    move.earlier ++ [move.setup] ++ move.body ++ move.later = List.range before.atoms.length ∧
    before.atoms.all (Elision.validAtom fields) = true ∧ ready fields before.atoms move = true)

def command {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error) : Command → State Value → Outcome Value Error
  | .call id,state => run id 0 state
  | .loop descriptor,state =>
      if descriptor.iterArgs = [] ∧ descriptor.yields = [] ∧ descriptor.results = [] then
        repeatBody (fun index => calls run index descriptor.body) descriptor.extent 0 state
      else otherLoop descriptor state

def execute {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error) : List Command → State Value → Outcome Value Error
  | [],state => .ok state
  | first::rest,state => (command run otherLoop first state).bind (execute run otherLoop rest)

def program {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error) (code : Program) : State Value → Outcome Value Error :=
  execute run otherLoop code.commands

theorem execute_append {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error) (earlier later : List Command)
    (initial : State Value) :
    execute run otherLoop (earlier++later) initial =
      (execute run otherLoop earlier initial).bind (execute run otherLoop later) := by
  induction earlier generalizing initial with
  | nil => rfl
  | cons first rest ih =>
    cases step : command run otherLoop first initial with
    | error failure => simp [execute,step,Except.bind]
    | ok final => simpa only [List.cons_append,execute,step,Except.bind] using ih final

theorem valid_write_fields {fields : List Nat} {effect : Elision.Effect}
    (valid : Elision.validEffect fields effect = true) :
    ∀ key ∈ effect.writes.map (·.1), key ∈ fields := by
  have parts := valid
  simp only [Elision.validEffect,Bool.and_eq_true] at parts
  have facts := of_decide_eq_true parts.1
  intro key member
  obtain ⟨pair,inside,equal⟩ := List.mem_map.mp member
  exact equal ▸ facts.2.1 pair inside

/-- Pointwise admitted setup laws and constructor-derived successful body
frames suffice. No source execution, total body, or complete-run premise. -/
theorem ready_loop {fields : List Nat} {atoms : List Elision.Atom} {move : Move}
    (checked : ready fields atoms move = true) (positive : move.extent > 0)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (initial : State Value) :
    repeatBody (fun index => calls run index (move.setup::move.body)) move.extent 0 initial =
      (run move.setup 0 initial).bind
        (repeatBody (fun index => calls run index move.body) move.extent 0) := by
  cases found : atoms[move.setup]? with
  | none => simp [ready,found] at checked
  | some atom =>
    cases effectFound : atom.effect with
    | none => simp [ready,found,effectFound] at checked
    | some effect =>
      have parts : Elision.validEffect fields effect = true ∧ Elision.pure atom effect = true ∧
          safeBody atoms (effect.writes.map (·.1)) move.body = true := by
        simpa only [ready,found,effectFound,Bool.and_eq_true,and_assoc] using checked
      have binding : ∀ index state, run move.setup index state = .ok (Elision.assign literal effect.writes state) := by
        intro index state
        have admitted := models index move.setup atom found
        simp only [Elision.Admitted,effectFound] at admitted
        exact admitted.2 parts.2.1 state
      have frame : ∀ index, Frame (effect.writes.map (·.1)) (fun state => calls run index move.body state) := by
        intro index
        exact safeBody_frame fields literal atoms run models _ (valid_write_fields parts.1) index move.body parts.2.2
      rw [loop_setup literal effect.writes run move.setup move.body binding]
      rw [binding,Except.bind]
      exact hoist_nonempty_partial literal effect.writes _ frame move.extent positive 0 initial

/-- Complete accepted before/after kernel correspondence, including arbitrary
retained surrounding calls and first-error terminal state. Nonempty-token loops
are outside the accepted shape, irrespective of their supplied interpretation. -/
theorem check_sound {fields : List Nat} {before after : Program} {move : Move}
    (accepted : check fields before after move = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run) (initial : State Value) :
    program run otherLoop before initial = program run otherLoop after initial := by
  have facts := of_decide_eq_true accepted
  rcases facts with ⟨_,_,_,sourceShape,targetShape,positive,_,_,_,_,readyChecked⟩
  simp only [program,sourceShape,targetShape,source,target,execute_append]
  cases earlier : execute run otherLoop (move.earlier.map Command.call) initial with
  | error failure => rfl
  | ok state =>
    simp only [Except.bind,execute,command,List.not_mem_nil,ite_true]
    have core := ready_loop readyChecked positive literal run models state
    rw [core]
    cases step : run move.setup 0 state with
    | error failure => rfl
    | ok installed => rfl

theorem checked_observation {fields : List Nat} {before after : Program} {move : Move}
    (accepted : check fields before after move = true)
    {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run) (initial : State Value)
    (observe : Outcome Value Error → Observation) :
    observe (program run otherLoop before initial) = observe (program run otherLoop after initial) :=
  congrArg observe (check_sound accepted literal run otherLoop models initial)

/-- Any identical partial surrounding context observes the exact same complete
outcome; no local totality or successful-source-execution assumption is needed. -/
theorem context_preserved {fields : List Nat} {before after : Program} {move : Move}
    (accepted : check fields before after move = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run)
    (earlier later : List Command) (initial : State Value) :
    execute run otherLoop (earlier++before.commands++later) initial =
      execute run otherLoop (earlier++after.commands++later) initial := by
  simp only [execute_append]
  cases outcome : execute run otherLoop earlier initial with
  | error failure => rfl
  | ok state =>
    simp only [Except.bind]
    have equal := check_sound accepted literal run otherLoop models state
    simp only [program] at equal
    rw [equal]

end Merlin.GraphState.Configuration.PartialHoist
