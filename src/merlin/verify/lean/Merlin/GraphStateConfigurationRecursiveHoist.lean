/- Recursive configuration hoisting, without assumed simulation.
Recursive actual Kernel command scopes, retaining arbitrary partial call errors
and terminal state. Existing Prefix lemmas supply total setters and compatible
successful-state preservation; enclosing loops are interpreted, not opaque. -/
import Merlin.GraphStateConfigurationPrefixHoist

set_option autoImplicit false
namespace Merlin.GraphState.Configuration.PartialHoist.Recursive
open StateEffects

abbrev Env := String → Nat

def bindIndex (env : Env) (var : String) (index : Nat) : Env :=
  fun name => if name = var then index else env name

inductive Code where
  | done
  | call (id : Nat) (rest : Code)
  | loop (var : String) (extent : Nat) (metadata : String) (body rest : Code)
  deriving DecidableEq, Repr

def ids : Code → List Nat
  | .done => []
  | .call id rest => id :: ids rest
  | .loop _ _ _ body rest => ids body ++ ids rest

def prepend : List Nat → Code → Code
  | [], rest => rest
  | id :: tail, rest => .call id (prepend tail rest)

def execute {Value Error : Type}
    (run : Nat → Env → State Value → Outcome Value Error) (env : Env) :
    Code → State Value → Outcome Value Error
  | .done, state => .ok state
  | .call id rest, state => (run id env state).bind (execute run env rest)
  | .loop var extent _ body rest, state =>
      (repeatBody (fun index => execute run (bindIndex env var index) body) extent 0 state).bind
        (execute run env rest)

def Models {Value Error : Type} (fields : List Nat) (literal : Int → Value)
    (atoms : List Elision.Atom) (run : Nat → Env → State Value → Outcome Value Error) : Prop :=
  ∀ env id atom, atoms[id]? = some atom →
    Elision.Admitted fields literal (fun _ state => run id env state) atom

def runAt {Value Error : Type}
    (run : Nat → Env → State Value → Outcome Value Error) (env : Env) :
    Nat → Nat → State Value → Outcome Value Error := fun id _ => run id env

theorem models_at {Value Error : Type} {fields : List Nat} {literal : Int → Value}
    {atoms : List Elision.Atom} {run : Nat → Env → State Value → Outcome Value Error}
    (models : Models fields literal atoms run) (env : Env) :
    PartialHoist.Models fields literal atoms (runAt run env) := by
  intro _ id atom found
  exact models env id atom found

theorem execute_prepend {Value Error : Type}
    (run : Nat → Env → State Value → Outcome Value Error) (env : Env)
    (setups : List Nat) (body : Code) (initial : State Value) :
    execute run env (prepend setups body) initial =
      (calls (runAt run env) 0 setups initial).bind (execute run env body) := by
  induction setups generalizing initial with
  | nil => rfl
  | cons first rest ih =>
    cases step : run first env initial with
    | error failure => simp only [prepend,execute,calls,runAt,step,Except.bind]
    | ok state => simpa only [prepend,execute,calls,runAt,step,Except.bind] using ih state

def safe (atoms : List Elision.Atom) (writes : Elision.Assignments) : Code → Bool
  | .done => true
  | .call id rest => match atoms[id]? with
    | none => false
    | some atom => match atom.effect with
      | none => false
      | some effect => Prefix.compatible writes effect && safe atoms writes rest
  | .loop _ _ _ body rest => safe atoms writes body && safe atoms writes rest

theorem repeat_fixed {Value Error : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (body : Nat → State Value → Outcome Value Error)
    (preserves : ∀ index initial final, Elision.assign literal writes initial = initial →
      body index initial = .ok final → Elision.assign literal writes final = final) :
    ∀ count index initial final, Elision.assign literal writes initial = initial →
      repeatBody body count index initial = .ok final → Elision.assign literal writes final = final := by
  intro count
  induction count with
  | zero =>
    intro index initial final fixed success
    have equal : initial = final := Except.ok.inj success
    simpa only [equal] using fixed
  | succ count ih =>
    intro index initial final fixed success
    cases step : body index initial with
    | error failure => simp only [repeatBody,step,Except.bind] at success; cases success
    | ok middle =>
      have tail : repeatBody body count (index+1) middle = .ok final := by
        simpa only [repeatBody,step,Except.bind] using success
      exact ih (index+1) middle final (preserves index initial middle fixed step) tail

theorem safe_fixed {Value Error : Type} {fields : List Nat} (literal : Int → Value)
    (writes : Elision.Assignments)
    (run : Nat → Env → State Value → Outcome Value Error) (atoms : List Elision.Atom)
    (models : Models fields literal atoms run)
    (keys : ∀ key ∈ writes.map (·.1), key ∈ fields) :
    ∀ code, safe atoms writes code = true →
      ∀ env initial final, Elision.assign literal writes initial = initial →
        execute run env code initial = .ok final → Elision.assign literal writes final = final := by
  intro code
  induction code with
  | done =>
    intro _ env initial final fixed success
    have equal : initial = final := Except.ok.inj success
    simpa only [equal] using fixed
  | call id rest ih =>
    intro checked env initial final fixed success
    cases found : atoms[id]? with
    | none => simp [safe,found] at checked
    | some atom =>
      cases effectFound : atom.effect with
      | none => simp [safe,found,effectFound] at checked
      | some effect =>
        have parts : Prefix.compatible writes effect = true ∧ safe atoms writes rest = true := by
          simpa only [safe,found,effectFound,Bool.and_eq_true] using checked
        cases step : run id env initial with
        | error failure => simp only [execute,step,Except.bind] at success; cases success
        | ok middle =>
          have middleFixed := Prefix.atom_fixed literal writes (runAt run env) (models_at models env)
            keys found effectFound parts.1 0 initial middle fixed step
          have tail : execute run env rest middle = .ok final := by
            simpa only [execute,step,Except.bind] using success
          exact ih parts.2 env middle final middleFixed tail
  | loop var extent metadata body rest bodyIH restIH =>
    intro checked env initial final fixed success
    have parts : safe atoms writes body = true ∧ safe atoms writes rest = true := by
      simpa only [safe,Bool.and_eq_true] using checked
    have loopFixed := repeat_fixed literal writes
      (fun index => execute run (bindIndex env var index) body)
      (fun index => bodyIH parts.1 (bindIndex env var index))
    cases step : repeatBody (fun index => execute run (bindIndex env var index) body) extent 0 initial with
    | error failure => simp only [execute,step,Except.bind] at success; cases success
    | ok middle =>
      have tail : execute run env rest middle = .ok final := by
        simpa only [execute,step,Except.bind] using success
      exact restIH parts.2 env middle final (loopFixed extent 0 initial middle fixed step) tail

theorem loop_prefix {Value Error : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (run : Nat → Env → State Value → Outcome Value Error)
    (setups : List Nat) (body : Code) (var : String) (env : Env)
    (binding : ∀ env state, calls (runAt run env) 0 setups state = .ok (Elision.assign literal writes state)) :
    ∀ count index state,
      repeatBody (fun index => execute run (bindIndex env var index) (prepend setups body)) count index state =
        repeatSetup (Elision.assign literal writes)
          (fun index => execute run (bindIndex env var index) body) count index state := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index state
    simp only [repeatBody,repeatSetup,execute_prepend,binding,Except.bind]
    cases step : execute run (bindIndex env var index) body (Elision.assign literal writes state) with
    | error failure => rfl
    | ok final => simpa only [step,Except.bind] using ih (index+1) final

structure Move where
  setups : List Nat
  body : Code
  rest : Code
  var : String
  extent : Nat
  metadata : String
  deriving DecidableEq, Repr

def source (move : Move) : Code :=
  .loop move.var move.extent move.metadata (prepend move.setups move.body) move.rest

def target (move : Move) : Code :=
  prepend move.setups (.loop move.var move.extent move.metadata move.body move.rest)

def movable (fields : List Nat) (atoms : List Elision.Atom) (move : Move) : Bool :=
  match Prefix.collect fields atoms move.setups with
  | none => false
  | some writes => safe atoms writes move.body

theorem movable_sound {fields : List Nat} {atoms : List Elision.Atom} {move : Move}
    (checked : movable fields atoms move = true) (positive : move.extent > 0)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Env → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (env : Env) (initial : State Value) :
    execute run env (source move) initial = execute run env (target move) initial := by
  cases collected : Prefix.collect fields atoms move.setups with
  | none => simp [movable,collected] at checked
  | some writes =>
    have bodySafe : safe atoms writes move.body = true := by
      simpa only [movable,collected] using checked
    have keys := (Prefix.collect_sound collected literal (runAt run env) (models_at models env)).1
    have binding : ∀ env state, calls (runAt run env) 0 move.setups state =
        .ok (Elision.assign literal writes state) := by
      intro selected state
      exact (Prefix.collect_sound collected literal (runAt run selected) (models_at models selected)).2 0 state
    have preserves : ∀ index before after, Elision.assign literal writes before = before →
        execute run (bindIndex env move.var index) move.body before = .ok after →
        Elision.assign literal writes after = after := by
      intro index
      exact safe_fixed literal writes run atoms models keys move.body bodySafe (bindIndex env move.var index)
    have core : repeatBody (fun index => execute run (bindIndex env move.var index)
          (prepend move.setups move.body)) move.extent 0 initial =
        repeatBody (fun index => execute run (bindIndex env move.var index) move.body)
          move.extent 0 (Elision.assign literal writes initial) := by
      rw [loop_prefix literal writes run move.setups move.body move.var env binding]
      exact Prefix.hoist_nonempty_fixed literal writes _ preserves move.extent positive 0 initial
    change (repeatBody (fun index => execute run (bindIndex env move.var index)
        (prepend move.setups move.body)) move.extent 0 initial).bind (execute run env move.rest) =
      execute run env (prepend move.setups
        (.loop move.var move.extent move.metadata move.body move.rest)) initial
    calc
      _ = (repeatBody (fun index => execute run (bindIndex env move.var index) move.body)
          move.extent 0 (Elision.assign literal writes initial)).bind (execute run env move.rest) :=
        congrArg (fun outcome => outcome.bind (execute run env move.rest)) core
      _ = _ := by
        rw [execute_prepend,binding]
        rfl

theorem repeat_congr {Value Error : Type}
    (before after : Nat → State Value → Outcome Value Error)
    (same : ∀ index state, before index state = after index state) :
    ∀ count index state, repeatBody before count index state = repeatBody after count index state := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index state
    simp only [repeatBody,same]
    cases step : after index state with
    | error failure => rfl
    | ok final => simpa only [step,Except.bind] using ih (index+1) final

inductive Context where
  | hole
  | call (id : Nat) (rest : Context)
  | loopBody (var : String) (extent : Nat) (metadata : String) (body : Context) (rest : Code)
  | loopRest (var : String) (extent : Nat) (metadata : String) (body : Code) (rest : Context)
  deriving DecidableEq, Repr

def plug : Context → Code → Code
  | .hole, code => code
  | .call id rest, code => .call id (plug rest code)
  | .loopBody var extent metadata body rest, code => .loop var extent metadata (plug body code) rest
  | .loopRest var extent metadata body rest, code => .loop var extent metadata body (plug rest code)

theorem context_congr {Value Error : Type}
    (run : Nat → Env → State Value → Outcome Value Error) (before after : Code)
    (same : ∀ env state, execute run env before state = execute run env after state) :
    ∀ context env state, execute run env (plug context before) state = execute run env (plug context after) state := by
  intro context
  induction context with
  | hole => exact same
  | call id rest ih =>
    intro env state
    simp only [plug,execute]
    cases step : run id env state with
    | error failure => rfl
    | ok final => simpa only [Except.bind] using ih env final
  | loopBody var extent metadata body rest ih =>
    intro env state
    simp only [plug,execute]
    rw [repeat_congr _ _ (fun index => ih (bindIndex env var index)) extent 0 state]
  | loopRest var extent metadata body rest ih =>
    intro env state
    simp only [plug,execute]
    cases step : repeatBody (fun index => execute run (bindIndex env var index) body) extent 0 state with
    | error failure => rfl
    | ok final => simpa only [Except.bind] using ih env final

structure Stage where
  after : Code
  context : Context
  move : Move
  deriving DecidableEq, Repr

def ready (fields : List Nat) (atoms : List Elision.Atom) (before : Code) (stage : Stage) : Bool :=
  decide (before = plug stage.context (source stage.move) ∧
    stage.after = plug stage.context (target stage.move) ∧
    stage.move.extent > 0 ∧ stage.move.setups ≠ [] ∧ movable fields atoms stage.move = true)

theorem ready_sound {fields : List Nat} {atoms : List Elision.Atom} {before : Code} {stage : Stage}
    (checked : ready fields atoms before stage = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Env → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (env : Env) (initial : State Value) :
    execute run env before initial = execute run env stage.after initial := by
  rcases of_decide_eq_true checked with ⟨sourceShape,targetShape,positive,_,core⟩
  rw [sourceShape,targetShape]
  exact context_congr run _ _ (movable_sound core positive literal run models) stage.context env initial

def chain (fields : List Nat) (atoms : List Elision.Atom) : Code → List Stage → Code → Bool
  | before, [], final => decide (before = final)
  | before, stage :: rest, final => ready fields atoms before stage && chain fields atoms stage.after rest final

theorem chain_sound {fields : List Nat} {atoms : List Elision.Atom}
    (stages : List Stage) {before after : Code} (checked : chain fields atoms before stages after = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Env → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (env : Env) (initial : State Value) :
    execute run env before initial = execute run env after initial := by
  induction stages generalizing before with
  | nil =>
    have equal : before = after := of_decide_eq_true checked
    rw [equal]
  | cons stage rest ih =>
    have parts : ready fields atoms before stage = true ∧ chain fields atoms stage.after rest after = true := by
      simpa only [chain,Bool.and_eq_true] using checked
    exact (ready_sound parts.1 literal run models env initial).trans (ih parts.2)

structure Program where
  frame : String
  atoms : List Elision.Atom
  code : Code
  deriving DecidableEq, Repr

def check (fields : List Nat) (before after : Program) (stages : List Stage) : Bool :=
  decide (fields.Nodup ∧ before.frame = after.frame ∧ before.atoms = after.atoms ∧
    stages ≠ [] ∧ before.atoms.map (·.id) = List.range before.atoms.length ∧
    ids before.code = List.range before.atoms.length ∧ ids after.code = List.range after.atoms.length ∧
    before.atoms.all (Elision.validAtom fields) = true ∧ chain fields before.atoms before.code stages after.code = true)

theorem check_sound {fields : List Nat} {before after : Program} {stages : List Stage}
    (checked : check fields before after stages = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Env → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run) (env : Env) (initial : State Value) :
    execute run env before.code initial = execute run env after.code initial := by
  rcases of_decide_eq_true checked with ⟨_,_,_,_,_,_,_,_,steps⟩
  exact chain_sound stages steps literal run models env initial

theorem checked_observation {fields : List Nat} {before after : Program} {stages : List Stage}
    (checked : check fields before after stages = true)
    {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → Env → State Value → Outcome Value Error)
    (models : Models fields literal before.atoms run) (env : Env) (initial : State Value)
    (observe : Outcome Value Error → Observation) :
    observe (execute run env before.code initial) = observe (execute run env after.code initial) :=
  congrArg observe (check_sound checked literal run models env initial)

end Merlin.GraphState.Configuration.PartialHoist.Recursive
