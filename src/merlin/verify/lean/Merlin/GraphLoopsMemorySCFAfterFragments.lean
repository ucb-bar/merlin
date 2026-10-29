/- UNCOMPILED. Ordered expanded-target fragments include identity casts at
elided allocation sites and empty fragments at elided releases. They execute
ordinary typed Code; the old epoch sequence/evaluator is not modified.
-/
import Merlin.GraphLoopsMemorySCFAfterAllocation
import Merlin.GraphLoopsShapeEpochSequence

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

inductive Fragment where
  | atom (instruction : Epoch.Atom)
  | alias (origin result : Binding) (type : ViewType)
  | empty
  deriving DecidableEq, Repr

def Fragment.code : Fragment → Code → Code
  | .atom instruction, rest => instruction.code rest
  | .alias origin result type, rest => .cast origin result type type rest
  | .empty, rest => rest

def Fragment.step {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World) (fragment : Fragment)
    (env : GraphLoopsShape.Env Value) (before : World) (next : GraphLoopsShape.Env Value) (after : World) : Prop :=
  match fragment with
  | .atom instruction => instruction.step meaning env before next after
  | .alias origin result type => ∃ value, castResult type type (env origin.name) = some value ∧
      next = GraphLoopsShape.put env result.name value ∧ after = before
  | .empty => next = env ∧ after = before

/-- The actual scoped alias reads one existing descriptor and defines one
name. Unrelated registers (including generated temporaries) are unrestricted.
-/
theorem alias_actual_iff {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (origin result : Binding) (type : ViewType) (env : GraphLoopsShape.Env Value)
    (value : Runtime Value) (read : env origin.name = value) (typed : value.kind = .view type)
    (before : World) (next : GraphLoopsShape.Env Value) (after : World) :
    (Fragment.alias origin result type).step meaning env before next after ↔
      next = GraphLoopsShape.put env result.name value ∧ after = before := by
  have cast := cast_result typed
  simp only [Fragment.step, read, cast, Option.some.injEq]
  constructor
  · rintro ⟨actual, equal, next, final⟩
    cases equal
    exact ⟨next, final⟩
  · rintro ⟨next, final⟩
    exact ⟨value, rfl, next, final⟩

theorem Fragment.eval_suffix {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (fragment : Fragment) (rest : Code) (env : GraphLoopsShape.Env Value)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false [] (fragment.code rest) env before outputs after ↔
      ∃ next middle, fragment.step meaning env before next middle ∧
        Eval meaning false [] rest next middle outputs after := by
  cases fragment with
  | atom instruction => exact instruction.eval_suffix meaning rest env before outputs after
  | alias origin result type =>
    simp only [Fragment.code, Eval, Bool.false_eq_true, ↓reduceIte, Fragment.step]
    constructor
    · rintro ⟨value, cast, suffix⟩
      exact ⟨_, before, ⟨value, cast, rfl, rfl⟩,
        (Epoch.eval_aliases meaning rest _ [] _ _ _ _).mp suffix⟩
    · rintro ⟨next, middle, ⟨value, cast, rfl, rfl⟩, suffix⟩
      exact ⟨value, cast, (Epoch.eval_aliases meaning rest _ [] _ _ _ _).mpr suffix⟩
  | empty =>
    constructor
    · intro run; exact ⟨env, before, ⟨rfl, rfl⟩, run⟩
    · rintro ⟨next, middle, ⟨rfl, rfl⟩, run⟩; exact run

def fragmentsCode (fragments : List Fragment) (returned : List Binding) : Code :=
  fragments.foldr Fragment.code (.returned returned)

inductive FragmentsRun {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World) :
    List Fragment → GraphLoopsShape.Env Value → World → GraphLoopsShape.Env Value → World → Prop where
  | done (env : GraphLoopsShape.Env Value) (before : World) : FragmentsRun meaning [] env before env before
  | more {fragment : Fragment} {rest : List Fragment} {env middleEnv finalEnv : GraphLoopsShape.Env Value}
      {before middle after : World}
      (first : fragment.step meaning env before middleEnv middle)
      (suffix : FragmentsRun meaning rest middleEnv middle finalEnv after) :
      FragmentsRun meaning (fragment :: rest) env before finalEnv after

/-- Every fragment, suffix and returned value participates in ordinary Code
execution. A later epoch simulation must discharge each actual fragment step.
-/
theorem fragments_complete {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (fragments : List Fragment) (returned : List Binding) (env : GraphLoopsShape.Env Value)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false [] (fragmentsCode fragments returned) env before outputs after ↔
      ∃ finalEnv, FragmentsRun meaning fragments env before finalEnv after ∧
        outputs = readArguments false [] finalEnv returned := by
  induction fragments generalizing env before with
  | nil =>
    constructor
    · rintro ⟨outputs, rfl⟩; exact ⟨env, .done env _, outputs⟩
    · rintro ⟨finalEnv, run, outputs⟩; cases run; exact ⟨outputs, rfl⟩
  | cons fragment rest induction =>
    change Eval meaning false [] (fragment.code (fragmentsCode rest returned)) env before outputs after ↔ _
    rw [fragment.eval_suffix]
    constructor
    · rintro ⟨middleEnv, middle, first, suffix⟩
      obtain ⟨finalEnv, suffix, outputs⟩ := (induction middleEnv middle).mp suffix
      exact ⟨finalEnv, .more first suffix, outputs⟩
    · rintro ⟨finalEnv, run, outputs⟩
      cases run with
      | more first suffix => exact ⟨_, _, first, (induction _ _).mpr ⟨finalEnv, suffix, outputs⟩⟩

/-- Elided allocation introduces a typed alias to its retained owner. This
changes only the new SSA binding; the actual descriptor/heap/caller is preserved.
Slot equality and both logical bounds are explicit finite plan obligations.
-/
theorem alias_slot {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (objects : GraphLoopsReuse.Objects) (owner object : Nat)
    (ownerInside : owner < objects.length) (objectInside : object < objects.length)
    (sameSlot : GraphLoopsReuse.slotOf objects owner = GraphLoopsReuse.slotOf objects object)
    (origin result : Binding) (shape : List Nat) (element : String)
    (env : GraphLoopsShape.Env Value) (ownerValue : env origin.name = Epoch.view owner shape element)
    (before : World) (next : GraphLoopsShape.Env Value) (after : World) :
    (Fragment.alias origin result (Epoch.type shape element)).step meaning (slotEnv objects env)
      before next after ↔
      next = slotEnv objects (GraphLoopsShape.put env result.name (Epoch.view object shape element)) ∧ after = before := by
  have read : slotEnv objects env origin.name = Epoch.view (GraphLoopsReuse.slotOf objects object) shape element := by
    change slotRuntime objects (env origin.name) = _
    rw [ownerValue, slot_private_view objects owner ownerInside, sameSlot]
  have cast : castResult (Epoch.type shape element) (Epoch.type shape element)
      (slotEnv objects env origin.name) = some (slotRuntime objects (Epoch.view object shape element)) := by
    rw [read, slot_private_view objects object objectInside]
    exact cast_result rfl
  simp only [Fragment.step, cast, Option.some.injEq]
  rw [slot_put]
  constructor
  · rintro ⟨value, same, next, final⟩
    cases same
    exact ⟨next, final⟩
  · rintro ⟨next, final⟩
    exact ⟨_, rfl, next, final⟩

end Merlin.GraphLoopsMemorySCF.After
