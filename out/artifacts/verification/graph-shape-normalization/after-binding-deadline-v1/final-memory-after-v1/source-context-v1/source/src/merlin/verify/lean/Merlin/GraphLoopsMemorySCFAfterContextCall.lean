/- UNCOMPILED successor. Exact one-call entry decomposition. The actual callee
runs at the parent's smaller call depth; the suffix has no calls, so complete
composition needs neither a guessed common depth nor a fuel-monotonicity premise.
-/
import Merlin.GraphLoopsMemorySCFAfterContextSuffix

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

structure CallerContext where
  function : Function
  key : String
  callee : String
  arguments : List Binding
  results : List Binding
  suffix : Code
  deriving DecidableEq, Repr

def CallerContext.Facts (wrapper : CallerContext) (module : Module) : Prop :=
  functionAt module wrapper.function.name = some wrapper.function ∧
    wrapper.function.code = .action wrapper.key wrapper.arguments wrapper.results wrapper.suffix ∧
    callTarget module.calls wrapper.key = some wrapper.callee ∧
    (functionAt module wrapper.callee).isSome = true ∧ NoSuffixCalls module wrapper.suffix

instance (wrapper : CallerContext) (module : Module) : Decidable (wrapper.Facts module) := by
  unfold CallerContext.Facts
  infer_instance

theorem CallerContext.complete_iff {Value World : Type} (wrapper : CallerContext)
    (module : Module) (facts : wrapper.Facts module) (meaning : Interpretation Value World)
    (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning module false wrapper.function.name inputs before outputs after ↔
      ValuesHold wrapper.function.parameters inputs ∧
      ∃ returned middle,
        complete meaning module false wrapper.callee
          (readArguments false [] (assign emptyEnv wrapper.function.parameters inputs) wrapper.arguments)
          before returned middle ∧ ValuesHold wrapper.results returned ∧
        Eval meaning false (extendAliases (extendAliases [] wrapper.function.parameters) wrapper.results)
          wrapper.suffix (assign (assign emptyEnv wrapper.function.parameters inputs) wrapper.results returned)
          middle outputs after := by
  constructor
  · rintro ⟨depth, run⟩
    cases depth with
    | zero => exact False.elim run
    | succ depth =>
      simp only [GraphLoopsShape.invoke, facts.1, Function.invoke, facts.2.1] at run
      obtain ⟨typed, returned, middle, call, returnTyped, suffix⟩ := run
      simp only [withCalls, facts.2.2.1, facts.2.2.2.1, ↓reduceIte] at call
      exact ⟨typed, returned, middle, ⟨depth, call⟩, returnTyped,
        (suffix_dispatch meaning module depth wrapper.suffix facts.2.2.2.2 _ _ _ _ _).mp suffix⟩
  · rintro ⟨typed, returned, middle, ⟨depth, call⟩, returnTyped, suffix⟩
    refine ⟨depth + 1, ?_⟩
    simp only [GraphLoopsShape.invoke, facts.1, Function.invoke, facts.2.1]
    refine ⟨typed, returned, middle, ?_, returnTyped,
      (suffix_dispatch meaning module depth wrapper.suffix facts.2.2.2.2 _ _ _ _ _).mpr suffix⟩
    simpa only [withCalls, facts.2.2.1, facts.2.2.2.1, ↓reduceIte] using call

/-- Derive entry register typing from typed arguments and checked fresh formals.
No global intermediate-environment typing assumption is supplied.
-/
theorem caller_entry_typed {Value : Type} (parameters : List Binding) (inputs : List (Runtime Value))
    (fresh : Fresh [] parameters) (typed : ValuesHold parameters inputs) :
    Typed (extendContext [] parameters) (assign emptyEnv parameters inputs) := by
  have scope : Scope [] [] := ⟨List.nodup_nil, by simp⟩
  have same : Agreement [] [] (emptyEnv (Value := Value)) emptyEnv := ⟨by simp [Typed], by simp⟩
  exact (same.define_many scope typed fresh).1

theorem arguments_typed {Value : Type} (context : Context) (env : GraphLoopsShape.Env Value)
    (typed : Typed context env) (parameters arguments : List Binding)
    (kinds : parameters.map Binding.kind = arguments.map Binding.kind) (plain : Plain parameters)
    (bound : ∀ binding ∈ arguments, binding ∈ context) :
    ValuesHold parameters (readArguments false [] env arguments) := by
  induction parameters generalizing arguments with
  | nil =>
    cases arguments with
    | nil => exact .nil
    | cons => simp only [List.map_nil, List.map_cons] at kinds
  | cons parameter rest induction =>
    cases arguments with
    | nil => simp only [List.map_nil, List.map_cons] at kinds
    | cons argument tail =>
      have equal := List.cons.inj kinds
      refine .cons (Binding.holds_plain parameter _ ?_ (plain parameter List.mem_cons_self)) ?_
      · exact (typed argument (bound argument List.mem_cons_self)).1.trans equal.1.symm
      · exact induction tail equal.2 (fun item member => plain item (List.mem_cons_of_mem _ member))
          (fun item member => bound item (List.mem_cons_of_mem _ member))

def CallerContext.Arguments (wrapper : CallerContext) (kernel : Epoch.Kernel) : Prop :=
  wrapper.callee = kernel.function.name ∧ Fresh [] wrapper.function.parameters ∧ Plain kernel.function.parameters ∧
    kernel.function.parameters.map Binding.kind = wrapper.arguments.map Binding.kind ∧
    (∀ binding ∈ wrapper.arguments, binding ∈ extendContext [] wrapper.function.parameters) ∧
    NoViewBindings wrapper.results

instance (wrapper : CallerContext) (kernel : Epoch.Kernel) : Decidable (wrapper.Arguments kernel) := by
  unfold CallerContext.Arguments
  infer_instance

def CallerContext.Domain {Value : Type} (wrapper : CallerContext) (plan : Epoch.PlanContext)
    (kernel : Epoch.Kernel) (inputs : List (Runtime Value)) : Prop :=
  ValuesHold wrapper.function.parameters inputs ∧ ExternalValues plan.objects.length inputs ∧
    Epoch.CallersValid plan.objects.length plan.callers
      (assign emptyEnv kernel.function.parameters
        (readArguments false [] (assign emptyEnv wrapper.function.parameters inputs) wrapper.arguments))

theorem CallerContext.typed_arguments {Value : Type} (wrapper : CallerContext) (kernel : Epoch.Kernel)
    (facts : wrapper.Arguments kernel) (inputs : List (Runtime Value)) (typed : ValuesHold wrapper.function.parameters inputs) :
    ValuesHold kernel.function.parameters
      (readArguments false [] (assign emptyEnv wrapper.function.parameters inputs) wrapper.arguments) :=
  arguments_typed _ _ (caller_entry_typed wrapper.function.parameters inputs facts.2.1 typed) _ _
    facts.2.2.2.1 facts.2.2.1 facts.2.2.2.2.1

end Merlin.GraphLoopsMemorySCF.After
