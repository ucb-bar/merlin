/- Whole changed-boundary correspondence for pure scalar CSE/DCE/LICM.
Independent source/result SSA numbers are normalized under checked lexical
bindings. Equal returned/persistent outcomes follow operationally. -/
import Merlin.GraphStateScalarSound
import Merlin.GraphStateScalarTyping
import Merlin.GraphStateScalarInitialized

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

def check (registry : List Signature) (inputs outputs : List Ty) (before after : Program) : Bool :=
  checkTypes registry inputs outputs before && checkTypes registry inputs outputs after &&
    match normalize registry (vars inputs.length) before,normalize registry (vars inputs.length) after with
    | some first,some second => first == second
    | _,_ => false

theorem check_typed {registry : List Signature} {inputs outputs : List Ty} {before after : Program}
    (accepted : check registry inputs outputs before after = true) :
    Typed registry inputs outputs before ∧ Typed registry inputs outputs after := by
  have parts := accepted
  simp only [check,Bool.and_eq_true] at parts
  exact ⟨checkTypes_sound before inputs outputs parts.1.1,checkTypes_sound after inputs outputs parts.1.2⟩

theorem check_sound {registry : List Signature} {inputs outputs : List Ty} {before after : Program}
    (accepted : check registry inputs outputs before after = true) : Preserves registry inputs.length before after := by
  have parts := accepted
  simp only [check,Bool.and_eq_true] at parts
  cases first : normalize registry (vars inputs.length) before with
  | none => simp [first] at parts
  | some code =>
    cases second : normalize registry (vars inputs.length) after with
    | none => simp [first,second] at parts
    | some other =>
      have equal : code = other := by simpa only [first,second,beq_iff_eq] using parts.2
      subst other
      intro Value World Error model arguments world arity
      have binding := eval_vars model.default model.primitive arguments []
      simp only [List.append_nil] at binding
      rw [arity] at binding
      have left := normalize_sound model before (vars inputs.length) code first arguments world
      have right := normalize_sound model after (vars inputs.length) code second arguments world
      rw [binding] at left right
      exact left.trans right.symm

theorem preserves_trans {registry : List Signature} {count : Nat} {before middle after : Program}
    (first : Preserves registry count before middle) (second : Preserves registry count middle after) :
    Preserves registry count before after := by
  intro Value World Error model inputs world arity
  exact (first Value World Error model inputs world arity).trans (second Value World Error model inputs world arity)

theorem check_composes {registry : List Signature} {inputs outputs : List Ty} {before middle after : Program}
    (first : check registry inputs outputs before middle = true)
    (second : check registry inputs outputs middle after = true) : Preserves registry inputs.length before after :=
  preserves_trans (check_sound first) (check_sound second)

/-- Static typing discharges SSA initialization/result-count obligations in
the model using LOCAL opaque arity laws. This is not source success/no poison. -/
theorem check_initialized {registry : List Signature} {inputs outputs : List Ty} {before after : Program}
    (accepted : check registry inputs outputs before after = true) {Value World Error : Type}
    (model : Model Value World Error registry) (values : List Value) (world : World)
    (arity : values.length = inputs.length) :
    Shaped outputs.length (evalProgram model before values world) ∧
    Shaped outputs.length (evalProgram model after values world) :=
  ⟨typed_execution_shaped (check_typed accepted).1 model values world arity,
   typed_execution_shaped (check_typed accepted).2 model values world arity⟩

theorem check_operands_initialized {registry : List Signature} {inputs outputs : List Ty} {before after : Program}
    (accepted : check registry inputs outputs before after = true) {Value World Error : Type}
    (model : Model Value World Error registry) (values : List Value) (world : World)
    (arity : values.length = inputs.length) :
    Initialized model before values world ∧ Initialized model after values world :=
  ⟨typed_operands_initialized (check_typed accepted).1 model values world arity,
   typed_operands_initialized (check_typed accepted).2 model values world arity⟩

#print axioms check_sound
#print axioms check_typed
#print axioms check_composes
#print axioms check_initialized
#print axioms check_operands_initialized
end Merlin.GraphState.Scalar
