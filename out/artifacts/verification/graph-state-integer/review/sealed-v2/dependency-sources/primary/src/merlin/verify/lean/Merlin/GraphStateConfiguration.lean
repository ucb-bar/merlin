/- One supported actual persistent-configuration prefix hoist. This is a
specialization of the existing total sequential Action domain, not universal
loop semantics or a hardware effect admission. A pure setup is EXACTLY an
overwrite of its declared fields, with invocation-invariant values. -/
import Merlin.GraphState

namespace Merlin.GraphState.Configuration
open StateEffects
set_option autoImplicit false

def put {Value : Type} (fields : List Nat) (values initial : State Value) : State Value :=
  fun key => if key ∈ fields then values key else initial key

def putAction {Value : Type} (fields : List Nat) (values : State Value) :
    Action ⟨[],fields⟩ Value where
  step := put fields values
  frame := by intro initial key outside; simp [put,outside]
  locality := by intro first second _ key inside; simp [put,inside]

theorem put_idempotent {Value : Type} (fields : List Nat) (values initial : State Value) :
    put fields values (put fields values initial) = put fields values initial := by
  funext key
  simp only [put]
  split <;> rfl

theorem put_fixed_after {Value : Type} (fields : List Nat) (values initial : State Value)
    (body : State Value → State Value)
    (frame : ∀ state key, key ∈ fields → body state key = state key)
    (fixed : put fields values initial = initial) :
    put fields values (body initial) = body initial := by
  funext key
  by_cases inside : key ∈ fields
  · have equal := congrFun fixed key
    simpa only [put,ite_eq_left inside,frame initial key inside] using equal
  · simp only [put,ite_eq_right inside]

def repeatBody {Value : Type} (body : Nat → State Value → State Value) :
    Nat → Nat → State Value → State Value
  | 0,_,state => state
  | count+1,index,state => repeatBody body count (index+1) (body index state)

def repeatSetup {Value : Type} (setup : State Value → State Value)
    (body : Nat → State Value → State Value) : Nat → Nat → State Value → State Value
  | 0,_,state => state
  | count+1,index,state => repeatSetup setup body count (index+1) (body index (setup state))

theorem remove_fixed_setups {Value : Type} (fields : List Nat) (values : State Value)
    (body : Nat → State Value → State Value)
    (frame : ∀ index state key, key ∈ fields → body index state key = state key) :
    ∀ count index initial, put fields values initial = initial →
      repeatSetup (put fields values) body count index initial = repeatBody body count index initial := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index initial fixed
    simp only [repeatSetup,repeatBody,fixed]
    exact ih (index+1) (body index initial) (put_fixed_after fields values initial (body index) (frame index) fixed)

/-- A zero-trip loop must retain its setup inside the skipped path. -/
theorem hoist_nonempty {Value : Type} (fields : List Nat) (values : State Value)
    (body : Nat → State Value → State Value)
    (frame : ∀ index state key, key ∈ fields → body index state key = state key)
    (count : Nat) (nonempty : count > 0) (initial : State Value) :
    repeatSetup (put fields values) body count 0 initial =
      repeatBody body count 0 (put fields values initial) := by
  cases count with
  | zero => cases nonempty
  | succ count =>
    simp only [repeatSetup,repeatBody]
    exact remove_fixed_setups fields values body frame count 1 _
      (put_fixed_after fields values _ (body 0) (frame 0) (put_idempotent fields values initial))

inductive Command where
  | call (index : Nat)
  | loop (extent : Nat) (calls : List Nat)
  deriving DecidableEq, Repr

structure Hoist where
  earlier : List Nat
  setup : Nat
  body : List Nat
  extent : Nat
  later : List Nat
  fields : List Nat
  deriving DecidableEq, Repr

def source (move : Hoist) : List Command :=
  move.earlier.map .call ++ [.loop move.extent (move.setup::move.body)] ++ move.later.map .call

def target (move : Hoist) : List Command :=
  move.earlier.map .call ++ [.call move.setup,.loop move.extent move.body] ++ move.later.map .call

def safeBody (footprints : List Footprint) (fields : List Nat) : List Nat → Bool
  | [] => true
  | index::rest => match footprints[index]? with
    | none => false
    | some effects => decide (Disjoint fields effects.writes) && safeBody footprints fields rest

def check (footprints : List Footprint) (before after : List Command) (move : Hoist) : Bool :=
  decide (before = source move ∧ after = target move ∧ move.extent > 0 ∧
    move.fields ≠ [] ∧
    footprints[move.setup]? = some ⟨[],move.fields⟩ ∧
    move.earlier ++ [move.setup] ++ move.body ++ move.later = List.range footprints.length ∧
    safeBody footprints move.fields move.body = true)

def calls {Value : Type} (run : Nat → Nat → State Value → State Value) (ids : List Nat)
    (index : Nat) : State Value → State Value := execute (ids.map (fun id => run id index))

def command {Value : Type} (run : Nat → Nat → State Value → State Value) :
    Command → State Value → State Value
  | .call id => run id 0
  | .loop extent ids => repeatBody (calls run ids) extent 0

def program {Value : Type} (run : Nat → Nat → State Value → State Value)
    (code : List Command) : State Value → State Value := execute (code.map (command run))

theorem safeBody_frame {Value : Type} {footprints : List Footprint} {fields ids : List Nat}
    (run : Nat → State Value → State Value) (model : Models footprints run)
    (safe : safeBody footprints fields ids = true) :
    ∀ initial key, key ∈ fields → execute (ids.map run) initial key = initial key := by
  induction ids with
  | nil => intros; rfl
  | cons id rest ih =>
    cases found : footprints[id]? with
    | none => simp [safeBody,found] at safe
    | some effects =>
      have parts : Disjoint fields effects.writes ∧ safeBody footprints fields rest = true := by
        simpa only [safeBody,found,Bool.and_eq_true,decide_eq_true_eq] using safe
      obtain ⟨action,step⟩ := model id effects found
      intro initial key inside
      rw [List.map_cons,execute,List.foldl_cons]
      exact (ih parts.2 (run id initial) key inside).trans
        (by rw [←step]; exact action.frame initial key (parts.1 key inside))

theorem loop_with_setup {Value : Type} (run : Nat → Nat → State Value → State Value)
    (setup : Nat) (ids : List Nat) (fields : List Nat) (values : State Value)
    (binding : ∀ index, run setup index = put fields values) :
    ∀ count index initial,
      repeatBody (calls run (setup::ids)) count index initial =
        repeatSetup (put fields values) (calls run ids) count index initial := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index initial
    simp only [repeatBody,repeatSetup,calls,List.map_cons,execute,List.foldl_cons,binding]
    exact ih (index+1) _

/-- Complete checked source/target command correspondence, under the original
Action models and the stronger EXACT overwrite meaning of the moved setup.
The equality preserves every state cell and every surrounding observer. -/
theorem check_sound {footprints : List Footprint} {before after : List Command} {move : Hoist}
    (accepted : check footprints before after move = true)
    {Value : Type} (run : Nat → Nat → State Value → State Value)
    (model : ∀ index, Models footprints (fun id => run id index)) (values : State Value)
    (binding : ∀ index, run move.setup index = put move.fields values)
    (initial : State Value) : program run before initial = program run after initial := by
  have facts := of_decide_eq_true accepted
  rcases facts with ⟨rfl,rfl,nonempty,_,_,_,safe⟩
  have frame : ∀ index state key, key ∈ move.fields → calls run move.body index state key = state key := by
    intro index
    exact safeBody_frame (fun id => run id index) (model index) safe
  have equal := hoist_nonempty move.fields values (calls run move.body) frame move.extent nonempty
  simp only [program,source,target,List.map_append,List.map_cons,List.map_nil,
    execute,List.foldl_append,List.foldl_cons,List.foldl_nil,command]
  rw [loop_with_setup run move.setup move.body move.fields values binding,binding 0,equal]

#print axioms check_sound
#print axioms hoist_nonempty
end Merlin.GraphState.Configuration
