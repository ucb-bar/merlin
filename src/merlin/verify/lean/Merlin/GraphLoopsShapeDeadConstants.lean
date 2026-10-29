/- Plan-parametric deletion of dead intrinsic index constants in ordinary Code.
Every action/call/cast/dimension and region boundary is retained.
Both sides use ordinary modefalse and the same arbitrary partial interpretation.
No source success, local simulation, normalization-mode or whole-run law is a
premise. Actual producer/IR decoding and closed source/target binding stay with
the original compiler owner. -/
import Merlin.GraphLoopsShapeEpochScope
import Merlin.GraphLoopsShapeEpochSequence
import Merlin.GraphLoopsShapeAfter

set_option autoImplicit false
namespace Merlin.GraphLoopsShape.DeadConstants

/-- Recursively delete only dead intrinsic index literals. Scalar constants
are ordinary actions and are never deleted. Only selected dead names are
removed. Region callbacks are related by
the structural proof below, without assuming a handler is total or pure. -/
def pruneConstants (removals : List Nat) : Code → Code
  | .returned values => .returned values
  | .cast origin result source target rest =>
      .cast origin result source target (pruneConstants removals rest)
  | .dimension buffer axis result type axisIndex value rest =>
      .dimension buffer axis result type axisIndex value (pruneConstants removals rest)
  | .constant result value rest =>
      let remaining := pruneConstants removals rest
      if result.name ∈ removals ∧ result.name ∉ Epoch.freeReads remaining then remaining
      else .constant result value remaining
  | .action key operands results rest => .action key operands results (pruneConstants removals rest)
  | .region key operands parameters results body rest =>
      .region key operands parameters results (pruneConstants removals body) (pruneConstants removals rest)

/-- A dropped intrinsic index assignment cannot change any remaining free read,
including captures used by an otherwise arbitrary nested region handler. -/
theorem absent_put_agrees {Value : Type} (code : Code) (env : Env Value) (name : Nat)
    (value : Runtime Value) (absent : name ∉ Epoch.freeReads code) :
    Epoch.AgreeReads (Epoch.freeReads code) (put env name value) env := by
  intro current member
  have different : current ≠ name := fun equal => absent (equal ▸ member)
  exact ite_eq_right different

/-- Equality is derived for every interpretation/environment/world and complete
returned list. Partial actions and region bodies are evaluated in their original
order; unused actions and their failure behavior are not removed. -/
theorem prune_eval {Value World : Type} (meaning : Interpretation Value World) (removals : List Nat) (code : Code)
    (aliases : Aliases) (env : Env Value) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases code env before outputs after ↔
      Eval meaning false aliases (pruneConstants removals code) env before outputs after := by
  induction code generalizing aliases env before outputs after with
  | returned values => rfl
  | cast origin result source target rest induction =>
    simp only [pruneConstants, Eval, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr
    intro value
    exact and_congr_right (fun _ => induction _ _ _ _ _)
  | dimension buffer axis result type axisIndex value rest induction =>
    simp only [pruneConstants, Eval, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr
    intro actual
    exact and_congr_right (fun _ => induction _ _ _ _ _)
  | constant result value rest induction =>
    by_cases dropped : result.name ∈ removals ∧ result.name ∉ Epoch.freeReads (pruneConstants removals rest)
    · simp only [pruneConstants, ite_eq_left dropped]
      change Eval meaning false ((result.name, result.name) :: aliases) rest
        (put env result.name (.index value)) before outputs after ↔ _
      exact (induction _ _ _ _ _).trans
        ((Epoch.eval_free_reads meaning (pruneConstants removals rest) ((result.name, result.name) :: aliases)
          (put env result.name (.index value)) env
          (absent_put_agrees (pruneConstants removals rest) env result.name (.index value) dropped.2)
          before outputs after).trans
          (Epoch.eval_aliases meaning (pruneConstants removals rest)
            ((result.name, result.name) :: aliases) aliases env before outputs after))
    · simpa only [pruneConstants, ite_eq_right dropped, Eval] using
        induction ((result.name, result.name) :: aliases) (put env result.name (.index value)) before outputs after
  | action key operands results rest induction =>
    simp only [pruneConstants, Eval]
    apply exists_congr
    intro values
    apply exists_congr
    intro middle
    apply and_congr_right
    intro _
    apply and_congr_right
    intro _
    exact induction _ _ _ _ _
  | region key operands parameters results body rest bodyInduction restInduction =>
    have bodySame : (fun inputs initial outputs final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases aliases parameters) body (assign env parameters inputs)
          initial outputs final) =
        (fun inputs initial outputs final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases aliases parameters) (pruneConstants removals body) (assign env parameters inputs)
          initial outputs final) := by
      funext inputs initial returned final
      apply propext
      exact and_congr_right (fun _ => bodyInduction _ _ _ _ _)
    simp only [pruneConstants, Eval, bodySame]
    apply exists_congr
    intro values
    apply exists_congr
    intro middle
    apply and_congr_right
    intro _
    apply and_congr_right
    intro _
    exact restInduction _ _ _ _ _

theorem prune_eval_equal {Value World : Type} (meaning : Interpretation Value World) (removals : List Nat) (code : Code)
    (aliases : Aliases) (env : Env Value) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases code env before outputs after =
      Eval meaning false aliases (pruneConstants removals code) env before outputs after :=
  propext (prune_eval meaning removals code aliases env before outputs after)

/-- Parameter and return declarations, function order/names, call-key map and
all retained actions remain attached to their original actual entry.
The explicit name plan can select literals inside nested regions as well. -/
def pruneFunction (removals : List Nat) (function : Function) : Function :=
  { function with code := pruneConstants removals function.code }

def pruneModule (removals : List Nat) (module : Module) : Module :=
  ⟨module.functions.map (pruneFunction removals), module.calls⟩

theorem function_pruned {Value World : Type} (removals : List Nat) (function : Function) (meaning : Interpretation Value World) :
    function.invoke meaning false = (pruneFunction removals function).invoke meaning false := by
  funext inputs before outputs after
  apply propext
  simp only [Function.invoke, pruneFunction]
  exact and_congr_right (fun _ => prune_eval meaning removals function.code _ _ _ _ _)

theorem functionAt_pruned (removals : List Nat) (module : Module) (name : String) :
    functionAt (pruneModule removals module) name = (functionAt module name).map (pruneFunction removals) := by
  unfold functionAt pruneModule
  induction module.functions with
  | nil => rfl
  | cons function rest induction =>
    simp only [List.map_cons, List.find?_cons, pruneFunction]
    by_cases equal : function.name == name
    · simp [equal, pruneFunction]
    · simpa only [equal, Bool.false_eq_true, ↓reduceIte] using induction

theorem withCalls_pruned {Value World : Type} (meaning : Interpretation Value World) (removals : List Nat)
    (module : Module) (calls : String → RegionMeaning Value World) :
    withCalls meaning (pruneModule removals module) calls = withCalls meaning module calls := by
  unfold withCalls
  congr 1
  funext key inputs before outputs after
  change (match callTarget module.calls key with
    | some name => if (functionAt (pruneModule removals module) name).isSome then
        calls name inputs before outputs after else meaning.action key inputs before outputs after
    | none => meaning.action key inputs before outputs after) = _
  cases target : callTarget module.calls key with
  | none => rfl
  | some name =>
    simp only [functionAt_pruned]
    cases found : functionAt module name <;> rfl

/-- Both modules use one shared arbitrary partial base meaning. Each successor
call depth invokes the correspondingly pruned function, with unchanged headers. -/
theorem invoke_pruned {Value World : Type} (meaning : Interpretation Value World) (removals : List Nat) (module : Module) :
    ∀ depth, invoke meaning module false depth = invoke meaning (pruneModule removals module) false depth := by
  intro depth
  induction depth with
  | zero => rfl
  | succ depth induction =>
    funext name
    simp only [invoke, functionAt_pruned]
    cases found : functionAt module name with
    | none => rfl
    | some function =>
      simp only [Option.map_some]
      rw [induction, withCalls_pruned]
      exact function_pruned removals function _

/-- All entries preserve the entire modeled returned-values/world relation,
without assuming any source execution exists. No complete-error or divergence
classification beyond this underlying successful-evaluation model is claimed. -/
theorem prune_complete {Value World : Type} (meaning : Interpretation Value World) (removals : List Nat) (module : Module)
    (name : String) (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning module false name inputs before outputs after ↔
      complete meaning (pruneModule removals module) false name inputs before outputs after := by
  unfold complete
  apply exists_congr
  intro depth
  rw [invoke_pruned]

/-- A closed compiler query proves this equation for independently decoded
actual modules. It never substitutes a constructed target for the actual output.
Frame equality and actual source/target Valid64 are separate closed facts. -/
theorem checked_complete {Value World : Type} (actualBefore actualAfter : Module) (removals : List Nat) (swaps : List (Nat × Nat))
    (bound : After.permuteModule swaps actualAfter = pruneModule removals actualBefore)
    (meaning : Interpretation Value World) (name : String) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning actualBefore false name inputs before outputs after ↔
      complete meaning actualAfter false name inputs before outputs after := by
  have pruned := prune_complete meaning removals actualBefore name inputs before outputs after
  have renamed := After.complete_permuted swaps meaning actualAfter name inputs before outputs after
  rw [bound] at renamed
  exact pruned.trans renamed.symm

end Merlin.GraphLoopsShape.DeadConstants
