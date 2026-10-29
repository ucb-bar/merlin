/- Complete flat Kernel contexts for independent pure-prefix hoists. Every
stage retains the full actual command list and global atom roster. Existing
all-index Models and complete partial failure/state outcomes are unchanged. -/
import Merlin.GraphStateConfigurationPrefixHoist

set_option autoImplicit false
namespace Merlin.GraphState.Configuration.PartialHoist.Focus
open StateEffects

structure Move where
  earlier : List Command
  move : Prefix.Move
  later : List Command
  deriving DecidableEq, Repr

def source (focus : Move) : List Command := focus.earlier ++ Prefix.source focus.move ++ focus.later

def target (focus : Move) : List Command := focus.earlier ++ Prefix.target focus.move ++ focus.later

def ids : List Command → List Nat
  | [] => []
  | .call id::rest => id::ids rest
  | .loop loop::rest => loop.body ++ ids rest

def ready (fields : List Nat) (atoms : List Elision.Atom)
    (before after : List Command) (focus : Move) : Bool :=
  decide (before = source focus ∧ after = target focus ∧
    focus.move.extent > 0 ∧ focus.move.setups ≠ [] ∧
    focus.move.earlier = [] ∧ focus.move.later = [] ∧
    Prefix.ready fields atoms focus.move = true)

theorem move_sound {fields : List Nat} {atoms : List Elision.Atom} {move : Prefix.Move}
    (checked : Prefix.ready fields atoms move = true) (positive : move.extent > 0)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (initial : State Value) :
    execute run otherLoop (Prefix.source move) initial =
      execute run otherLoop (Prefix.target move) initial := by
  simp only [Prefix.source,Prefix.target,execute_append]
  cases earlier : execute run otherLoop (move.earlier.map Command.call) initial with
  | error failure => rfl
  | ok state =>
    simp only [Except.bind,execute,command]
    have core := Prefix.ready_loop checked positive literal run models state
    rw [core,Prefix.execute_calls]
    cases step : calls run 0 move.setups state with
    | error failure => rfl
    | ok installed => rfl

theorem ready_sound {fields : List Nat} {atoms : List Elision.Atom}
    {before after : List Command} {focus : Move}
    (checked : ready fields atoms before after focus = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (initial : State Value) :
    execute run otherLoop before initial = execute run otherLoop after initial := by
  have facts := of_decide_eq_true checked
  rcases facts with ⟨sourceShape,targetShape,positive,_,_,_,core⟩
  simp only [sourceShape,targetShape,source,target,execute_append]
  cases earlier : execute run otherLoop focus.earlier initial with
  | error failure => rfl
  | ok state =>
    simp only [Except.bind]
    rw [move_sound core positive literal run otherLoop models state]

structure Stage where
  after : List Command
  focus : Move
  deriving DecidableEq, Repr

def chain (fields : List Nat) (atoms : List Elision.Atom) :
    List Command → List Stage → List Command → Bool
  | before,[],final => decide (before = final)
  | before,stage::rest,final => ready fields atoms before stage.after stage.focus &&
      chain fields atoms stage.after rest final

theorem chain_sound {fields : List Nat} {atoms : List Elision.Atom}
    (stages : List Stage) {before after : List Command}
    (checked : chain fields atoms before stages after = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (initial : State Value) :
    execute run otherLoop before initial = execute run otherLoop after initial := by
  induction stages generalizing before with
  | nil =>
    have equal : before = after := of_decide_eq_true checked
    rw [equal]
  | cons stage rest ih =>
    have parts : ready fields atoms before stage.after stage.focus = true ∧
        chain fields atoms stage.after rest after = true := by
      simpa only [chain,Bool.and_eq_true] using checked
    exact (ready_sound parts.1 literal run otherLoop models initial).trans (ih parts.2)

def check (fields : List Nat) (before after : Program) (stages : List Stage) : Bool :=
  decide (fields.Nodup ∧ before.frame = after.frame ∧ before.atoms = after.atoms ∧
    stages ≠ [] ∧ before.atoms.map (·.id) = List.range before.atoms.length ∧
    ids before.commands = List.range before.atoms.length ∧
    ids after.commands = List.range after.atoms.length ∧
    before.atoms.all (Elision.validAtom fields) = true ∧
    chain fields before.atoms before.commands stages after.commands = true)

theorem check_sound {fields : List Nat} {before after : Program} {stages : List Stage}
    (checked : check fields before after stages = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run) (initial : State Value) :
    program run otherLoop before initial = program run otherLoop after initial := by
  have facts := of_decide_eq_true checked
  rcases facts with ⟨_,_,_,_,_,_,_,_,steps⟩
  exact chain_sound stages steps literal run otherLoop models initial

theorem checked_observation {fields : List Nat} {before after : Program} {stages : List Stage}
    (checked : check fields before after stages = true)
    {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run) (initial : State Value)
    (observe : Outcome Value Error → Observation) :
    observe (program run otherLoop before initial) = observe (program run otherLoop after initial) :=
  congrArg observe (check_sound checked literal run otherLoop models initial)

theorem context_preserved {fields : List Nat} {before after : Program} {stages : List Stage}
    (checked : check fields before after stages = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run)
    (earlier later : List Command) (initial : State Value) :
    execute run otherLoop (earlier ++ before.commands ++ later) initial =
      execute run otherLoop (earlier ++ after.commands ++ later) initial := by
  simp only [execute_append]
  cases outcome : execute run otherLoop earlier initial with
  | error failure => rfl
  | ok state =>
    simp only [Except.bind]
    have same := check_sound checked literal run otherLoop models state
    simp only [program] at same
    rw [same]

end Merlin.GraphState.Configuration.PartialHoist.Focus
