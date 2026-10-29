import Merlin.GraphStateConfigurationHoistChecker
set_option autoImplicit false
namespace Merlin.GraphState.Configuration.PartialHoist.Prefix
open StateEffects

theorem assign_append {Value : Type} (literal : Int → Value)
    (earlier later : Elision.Assignments) (state : State Value) :
    Elision.assign literal (later ++ earlier) state =
      Elision.assign literal later (Elision.assign literal earlier state) := by
  funext key
  simp only [Elision.assign, List.lookup_append]
  cases later.lookup key <;> rfl


def collect (fields : List Nat) (atoms : List Elision.Atom) : List Nat → Option Elision.Assignments
  | [] => some []
  | id::rest => match atoms[id]? with
    | none => none
    | some atom => match atom.effect with
      | none => none
      | some effect =>
          if Elision.validEffect fields effect && Elision.pure atom effect then
            (collect fields atoms rest).map (fun tail => tail ++ effect.writes)
          else none

theorem collect_sound {fields : List Nat} {atoms : List Elision.Atom} {ids : List Nat}
    {writes : Elision.Assignments} (checked : collect fields atoms ids = some writes)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error) (models : Models fields literal atoms run) :
    (∀ key ∈ writes.map (·.1), key ∈ fields) ∧
    ∀ index state, calls run index ids state = .ok (Elision.assign literal writes state) := by
  induction ids generalizing writes with
  | nil =>
    have equal : ([] : Elision.Assignments) = writes := by simpa only [collect,Option.some.injEq] using checked
    subst writes
    constructor
    · intro key member; cases member
    · intro index state; rfl
  | cons id rest ih =>
    cases found : atoms[id]? with
    | none => simp [collect,found] at checked
    | some atom =>
      cases effectFound : atom.effect with
      | none => simp [collect,found,effectFound] at checked
      | some effect =>
        by_cases valid : (Elision.validEffect fields effect && Elision.pure atom effect) = true
        · cases tail : collect fields atoms rest with
          | none => simp [collect,found,effectFound,valid,tail] at checked
          | some tailWrites =>
            have equal : tailWrites ++ effect.writes = writes := by
              simpa only [collect,found,effectFound,valid,ite_true,tail,Option.map_some,Option.some.injEq] using checked
            subst writes
            obtain ⟨tailKeys,tailRuns⟩ := ih tail
            have validParts : Elision.validEffect fields effect = true ∧ Elision.pure atom effect = true := by
              simpa only [Bool.and_eq_true] using valid
            have headKeys := valid_write_fields validParts.1
            constructor
            · intro key member
              simp only [List.map_append,List.mem_append] at member
              exact member.elim (tailKeys key) (headKeys key)
            · intro index state
              have admitted := models index id atom found
              simp only [Elision.Admitted,effectFound] at admitted
              have headRun := admitted.2 validParts.2 state
              simp only [calls,headRun,Except.bind]
              rw [tailRuns,assign_append]
        · simp [collect,found,effectFound,valid] at checked


theorem calls_append {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (index : Nat) (earlier later : List Nat) (initial : State Value) :
    calls run index (earlier++later) initial =
      (calls run index earlier initial).bind (calls run index later) := by
  induction earlier generalizing initial with
  | nil => rfl
  | cons first rest ih =>
    cases step : run first index initial with
    | error failure => simp [calls,step,Except.bind]
    | ok final => simpa only [List.cons_append,calls,step,Except.bind] using ih final

theorem execute_calls {Value Error : Type} (run : Nat → Nat → State Value → Outcome Value Error)
    (otherLoop : Loop → State Value → Outcome Value Error) (ids : List Nat) (initial : State Value) :
    execute run otherLoop (ids.map Command.call) initial = calls run 0 ids initial := by
  induction ids generalizing initial with
  | nil => rfl
  | cons first rest ih =>
    cases step : run first 0 initial with
    | error failure => simp [execute,command,calls,step,Except.bind]
    | ok final => simpa only [List.map_cons,execute,command,calls,step,Except.bind] using ih final

theorem loop_prefix {Value Error : Type} (literal : Int → Value) (writes : Elision.Assignments)
    (run : Nat → Nat → State Value → Outcome Value Error) (setups body : List Nat)
    (binding : ∀ index state, calls run index setups state = .ok (Elision.assign literal writes state)) :
    ∀ count index state,
      repeatBody (fun index => calls run index (setups++body)) count index state =
        repeatSetup (Elision.assign literal writes) (fun index => calls run index body) count index state := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index state
    simp only [repeatBody,repeatSetup,calls_append,binding,Except.bind]
    cases step : calls run index body (Elision.assign literal writes state) with
    | error failure => rfl
    | ok final => simpa only [step,Except.bind] using ih (index+1) final

theorem hoist_prefix {fields : List Nat} {atoms : List Elision.Atom} {setups body : List Nat}
    {writes : Elision.Assignments} (checked : collect fields atoms setups = some writes)
    (safe : safeBody atoms (writes.map (·.1)) body = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error) (models : Models fields literal atoms run)
    (count : Nat) (positive : count > 0) (initial : State Value) :
    repeatBody (fun index => calls run index (setups++body)) count 0 initial =
      (calls run 0 setups initial).bind (repeatBody (fun index => calls run index body) count 0) := by
  obtain ⟨keys,binding⟩ := collect_sound checked literal run models
  have frame : ∀ index, Frame (writes.map (·.1)) (fun state => calls run index body state) := by
    intro index
    exact safeBody_frame fields literal atoms run models _ keys index body safe
  rw [loop_prefix literal writes run setups body binding,binding,Except.bind]
  exact hoist_nonempty_partial literal writes _ frame count positive 0 initial


def sameWritten (writes : Elision.Assignments) (effect : Elision.Effect) (key : Nat) : Bool :=
  match effect.writes.lookup key with | none => true | some value => decide (writes.lookup key = some value)

def compatible (writes : Elision.Assignments) (effect : Elision.Effect) : Bool :=
  effect.ordered && (writes.map (·.1)).all (fun key => decide (key ∉ effect.clobbers) && sameWritten writes effect key)

def safeFixedBody (atoms : List Elision.Atom) (writes : Elision.Assignments) : List Nat → Bool
  | [] => true
  | id::rest => match atoms[id]? with
    | none => false
    | some atom => match atom.effect with
      | none => false
      | some effect => compatible writes effect && safeFixedBody atoms writes rest

theorem atom_fixed {Value Error : Type} {fields : List Nat} (literal : Int → Value)
    (writes : Elision.Assignments) (run : Nat → Nat → State Value → Outcome Value Error)
    {atoms : List Elision.Atom} (models : Models fields literal atoms run)
    (keys : ∀ key ∈ writes.map (·.1), key ∈ fields)
    {id : Nat} {atom : Elision.Atom} (found : atoms[id]? = some atom)
    {effect : Elision.Effect} (effectFound : atom.effect = some effect)
    (checked : compatible writes effect = true) (index : Nat) (initial final : State Value)
    (fixed : Elision.assign literal writes initial = initial) (success : run id index initial = .ok final) :
    Elision.assign literal writes final = final := by
  have parts : effect.ordered = true ∧
      (writes.map (·.1)).all (fun key => decide (key ∉ effect.clobbers) && sameWritten writes effect key) = true := by
    simpa only [compatible,Bool.and_eq_true] using checked
  have conditions : ∀ key ∈ writes.map (·.1), key ∉ effect.clobbers ∧ sameWritten writes effect key = true := by
    simpa only [List.all_eq_true,Bool.and_eq_true,decide_eq_true_eq] using parts.2
  have admitted := models index id atom found
  simp only [Elision.Admitted,effectFound] at admitted
  have transition := admitted.1 parts.1 initial final success
  funext key
  cases lookup : writes.lookup key with
  | none => simp only [Elision.assign,lookup]
  | some value =>
    have member : key ∈ writes.map (·.1) := List.mem_map.mpr ⟨(key,value),Elision.lookup_member lookup,rfl⟩
    have condition := conditions key member
    simp only [Elision.assign,lookup]
    cases written : effect.writes.lookup key with
    | none =>
      have frame := transition.2.1 key (keys key member) written condition.1
      have installed := congrFun fixed key
      simp only [Elision.assign,lookup] at installed
      exact installed.trans frame.symm
    | some other =>
      have equalLookup : writes.lookup key = some other := by simpa only [sameWritten,written,decide_eq_true_eq] using condition.2
      have equal : value = other := Option.some.inj (lookup.symm.trans equalLookup)
      exact (congrArg literal equal).trans (transition.1 key other written).symm

theorem safeFixedBody_preserves {Value Error : Type} {fields : List Nat} (literal : Int → Value)
    (writes : Elision.Assignments) (run : Nat → Nat → State Value → Outcome Value Error)
    (atoms : List Elision.Atom) (models : Models fields literal atoms run)
    (keys : ∀ key ∈ writes.map (·.1), key ∈ fields) (index : Nat) :
    ∀ ids, safeFixedBody atoms writes ids = true →
      ∀ initial final, Elision.assign literal writes initial = initial →
        calls run index ids initial = .ok final → Elision.assign literal writes final = final := by
  intro ids
  induction ids with
  | nil =>
    intro _ initial final fixed success
    have equal : initial = final := Except.ok.inj success
    simpa only [equal] using fixed
  | cons id rest ih =>
    intro checked initial final fixed success
    cases found : atoms[id]? with
    | none => simp [safeFixedBody,found] at checked
    | some atom =>
      cases effectFound : atom.effect with
      | none => simp [safeFixedBody,found,effectFound] at checked
      | some effect =>
        have parts : compatible writes effect = true ∧ safeFixedBody atoms writes rest = true := by
          simpa only [safeFixedBody,found,effectFound,Bool.and_eq_true] using checked
        cases step : run id index initial with
        | error failure => simp only [calls,step,Except.bind] at success; cases success
        | ok middle =>
          have middleFixed := atom_fixed literal writes run models keys found effectFound parts.1 index initial middle fixed step
          have tail : calls run index rest middle = .ok final := by simpa only [calls,step,Except.bind] using success
          exact ih parts.2 middle final middleFixed tail

theorem remove_fixed_prefixes {Value Error : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (body : Nat → State Value → Outcome Value Error)
    (preserves : ∀ index initial final, Elision.assign literal writes initial = initial →
      body index initial = .ok final → Elision.assign literal writes final = final) :
    ∀ count index initial, Elision.assign literal writes initial = initial →
      repeatSetup (Elision.assign literal writes) body count index initial = repeatBody body count index initial := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index initial fixed
    simp only [repeatSetup,repeatBody,fixed]
    cases step : body index initial with
    | error failure => rfl
    | ok final =>
      simp only [Except.bind]
      exact ih (index+1) final (preserves index initial final fixed step)

theorem hoist_nonempty_fixed {Value Error : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (body : Nat → State Value → Outcome Value Error)
    (preserves : ∀ index initial final, Elision.assign literal writes initial = initial →
      body index initial = .ok final → Elision.assign literal writes final = final)
    (count : Nat) (positive : count > 0) (index : Nat) (initial : State Value) :
    repeatSetup (Elision.assign literal writes) body count index initial =
      repeatBody body count index (Elision.assign literal writes initial) := by
  cases count with
  | zero => cases positive
  | succ count =>
    simp only [repeatSetup,repeatBody]
    cases step : body index (Elision.assign literal writes initial) with
    | error failure => rfl
    | ok final =>
      simp only [Except.bind]
      exact remove_fixed_prefixes literal writes body preserves count (index+1) final
        (preserves index _ final (Elision.assign_idempotent literal writes initial) step)

theorem hoist_prefix_fixed {fields : List Nat} {atoms : List Elision.Atom} {setups body : List Nat}
    {writes : Elision.Assignments} (checked : collect fields atoms setups = some writes)
    (safe : safeFixedBody atoms writes body = true)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error) (models : Models fields literal atoms run)
    (count : Nat) (positive : count > 0) (initial : State Value) :
    repeatBody (fun index => calls run index (setups++body)) count 0 initial =
      (calls run 0 setups initial).bind (repeatBody (fun index => calls run index body) count 0) := by
  obtain ⟨keys,binding⟩ := collect_sound checked literal run models
  have preserves : ∀ index initial final, Elision.assign literal writes initial = initial →
      calls run index body initial = .ok final → Elision.assign literal writes final = final := by
    intro index
    exact safeFixedBody_preserves literal writes run atoms models keys index body safe
  rw [loop_prefix literal writes run setups body binding,binding,Except.bind]
  exact hoist_nonempty_fixed literal writes _ preserves count positive 0 initial


structure Move where
  earlier : List Nat
  setups : List Nat
  body : List Nat
  later : List Nat
  var : String
  extent : Nat
  metadata : String
  deriving DecidableEq, Repr

def source (move : Move) : List Command :=
  move.earlier.map .call ++ [.loop ⟨move.var,move.extent,move.setups++move.body,move.metadata,[],[],[]⟩] ++
    move.later.map .call

def target (move : Move) : List Command :=
  move.earlier.map .call ++ move.setups.map .call ++
    [.loop ⟨move.var,move.extent,move.body,move.metadata,[],[],[]⟩] ++ move.later.map .call

def ready (fields : List Nat) (atoms : List Elision.Atom) (move : Move) : Bool :=
  match collect fields atoms move.setups with
  | none => false
  | some writes => safeFixedBody atoms writes move.body

def check (fields : List Nat) (before after : Program) (move : Move) : Bool :=
  decide (fields.Nodup ∧ before.frame = after.frame ∧ before.atoms = after.atoms ∧
    before.commands = source move ∧ after.commands = target move ∧
    move.extent > 0 ∧ move.setups ≠ [] ∧
    before.atoms.map (·.id) = List.range before.atoms.length ∧
    move.earlier ++ move.setups ++ move.body ++ move.later = List.range before.atoms.length ∧
    before.atoms.all (Elision.validAtom fields) = true ∧ ready fields before.atoms move = true)

theorem ready_loop {fields : List Nat} {atoms : List Elision.Atom} {move : Move}
    (checked : ready fields atoms move = true) (positive : move.extent > 0)
    {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → Outcome Value Error)
    (models : Models fields literal atoms run) (initial : State Value) :
    repeatBody (fun index => calls run index (move.setups++move.body)) move.extent 0 initial =
      (calls run 0 move.setups initial).bind (repeatBody (fun index => calls run index move.body) move.extent 0) := by
  cases collected : collect fields atoms move.setups with
  | none => simp [ready,collected] at checked
  | some writes =>
    have safe : safeFixedBody atoms writes move.body = true := by
      simpa only [ready,collected] using checked
    exact hoist_prefix_fixed collected safe literal run models move.extent positive initial

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
    simp only [Except.bind,execute,command]
    have core := ready_loop readyChecked positive literal run models state
    rw [core,execute_calls]
    cases step : calls run 0 move.setups state with
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

end Merlin.GraphState.Configuration.PartialHoist.Prefix
