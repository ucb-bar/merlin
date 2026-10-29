/- Whole-module complete outcomes, including internal calls. Call depth bounds
finite derivations only; loop/body execution is not tensor-volume unrolled.
External operations and region handlers retain ONE partial interpretation. -/
import Merlin.GraphLoopsShapeSound

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

structure Function where
  name : String
  parameters : List Binding
  returned : List Kind
  code : Code
  deriving DecidableEq, Repr

def Function.Valid (bits : Nat) (function : Function) : Prop :=
  Fresh [] function.parameters ∧ Plain function.parameters ∧
  GraphLoopsShape.Valid bits function.returned (extendContext [] function.parameters) (extendAliases [] function.parameters) function.code

instance (bits : Nat) (function : Function) : Decidable (function.Valid bits) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def emptyEnv {Value : Type} : Env Value := fun _ => .index 0

def Function.invoke {Value World : Type} (function : Function) (meaning : Interpretation Value World) (mode : Bool) :
    RegionMeaning Value World :=
  fun inputs before outputs after => ValuesHold function.parameters inputs ∧
    Eval meaning mode (extendAliases [] function.parameters) function.code
      (assign emptyEnv function.parameters inputs) before outputs after

theorem Function.preservation {Value World : Type} (function : Function) (bits : Nat) (valid : function.Valid bits)
    (meaning : Interpretation Value World) : function.invoke meaning false = function.invoke meaning true := by
  funext inputs before outputs after
  apply propext
  have emptyScope : Scope [] [] := ⟨List.nodup_nil, by simp⟩
  have emptySame : Agreement [] [] (emptyEnv (Value := Value)) emptyEnv := ⟨by simp [Typed], by simp⟩
  constructor
  · rintro ⟨typed, run⟩
    exact ⟨typed, (GraphLoopsShape.preservation meaning bits function.code _ _ _ valid.2.2 _ _
      (emptySame.define_many emptyScope typed valid.1) before outputs after).mp run⟩
  · rintro ⟨typed, run⟩
    exact ⟨typed, (GraphLoopsShape.preservation meaning bits function.code _ _ _ valid.2.2 _ _
      (emptySame.define_many emptyScope typed valid.1) before outputs after).mpr run⟩

structure Module where
  functions : List Function
  calls : List (String × String)
  deriving DecidableEq, Repr

def functionAt (module : Module) (name : String) : Option Function := module.functions.find? (fun function => function.name == name)

def Module.Valid (bits : Nat) (module : Module) : Prop :=
  (bits = 32 ∨ bits = 64) ∧ (module.functions.map Function.name).Nodup ∧
  (∀ function ∈ module.functions, function.Valid bits) ∧
  ∀ first ∈ module.calls, ∀ second ∈ module.calls, first.1 = second.1 → first.2 = second.2

instance (bits : Nat) (module : Module) : Decidable (module.Valid bits) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def callTarget : List (String × String) → String → Option String
  | [], _ => none
  | (key, target) :: rest, actual => if actual = key then some target else callTarget rest actual

/-- Only actual internal-call keys are redirected. External calls and all other
partial operations keep the exact shared primitive meaning. -/
def withCalls {Value World : Type} (base : Interpretation Value World) (module : Module)
    (calls : String → RegionMeaning Value World) : Interpretation Value World where
  action := fun key inputs before outputs after =>
    match callTarget module.calls key with
    | some name => if (functionAt module name).isSome then calls name inputs before outputs after else base.action key inputs before outputs after
    | none => base.action key inputs before outputs after
  region := base.region

/-- This is a declared operational call interpretation. Zero fuel admits no
internal function entry; successor unfolds one function with smaller call depth.
It is not a claimed bridge to a native/frozen evaluator or runtime stack. -/
def invoke {Value World : Type} (base : Interpretation Value World) (module : Module) (mode : Bool) :
    Nat → String → RegionMeaning Value World
  | 0, _ => fun _ _ _ _ => False
  | depth + 1, name =>
    match functionAt module name with
    | none => fun _ _ _ _ => False
    | some function => function.invoke (withCalls base module (invoke base module mode depth)) mode

theorem Module.checked_calls {Value World : Type} (module : Module) (bits : Nat) (valid : module.Valid bits)
    (base : Interpretation Value World) : ∀ depth, invoke base module false depth = invoke base module true depth := by
  intro depth
  induction depth with
  | zero => rfl
  | succ depth induction =>
    funext name
    unfold invoke
    split
    · rfl
    · rename_i function found
      have member : function ∈ module.functions := List.mem_of_find?_eq_some found
      have functionValid := valid.2.2.1 function member
      rw [induction]
      exact Function.preservation function bits functionValid _

def complete {Value World : Type} (base : Interpretation Value World) (module : Module) (mode : Bool)
    (name : String) (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) : Prop :=
  ∃ depth, invoke base module mode depth name inputs before outputs after

/-- Complete returned values AND the entire modeled heap/public world agree.
No source success, primitive totality, error-code or divergence identity follows. -/
theorem Module.checked_complete {Value World : Type} (module : Module) (bits : Nat) (valid : module.Valid bits)
    (base : Interpretation Value World) (name : String) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete base module false name inputs before outputs after ↔ complete base module true name inputs before outputs after := by
  unfold complete
  apply exists_congr
  intro depth
  rw [Module.checked_calls module bits valid base depth]

end Merlin.GraphLoopsShape
