/- Extensional lexical input dependence for ordinary typed Code. Preseeding
future private descriptor ghosts cannot affect a program before their definition.
The relation concerns only genuine free reads, including nested captures. -/
import Merlin.GraphLoopsShapeEpochNormalize

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def without (reads definitions : List Nat) : List Nat := reads.filter (fun name => name ∉ definitions)

def freeReads : Code → List Nat
  | .returned values => names values
  | .cast origin result _ _ rest => origin.name :: without (freeReads rest) [result.name]
  | .dimension buffer axis result _ _ _ rest => buffer.name :: axis.name :: without (freeReads rest) [result.name]
  | .constant result _ rest => without (freeReads rest) [result.name]
  | .action _ operands results rest => names operands ++ without (freeReads rest) (names results)
  | .region _ operands parameters results body rest => names operands ++
      without (freeReads body) (names parameters) ++ without (freeReads rest) (names results)

def AgreeReads {Value : Type} (reads : List Nat) (first second : Env Value) : Prop :=
  ∀ name ∈ reads, first name = second name

theorem agree_put {Value : Type} (reads : List Nat) (first second : Env Value) (name : Nat)
    (value : Runtime Value) (same : AgreeReads (without reads [name]) first second) :
    AgreeReads reads (put first name value) (put second name value) := by
  intro current member
  by_cases equal : current = name
  · simp only [put, equal, ↓reduceIte]
  · simp only [put, equal, ↓reduceIte]
    exact same current (by simpa [without] using And.intro member equal)

theorem assign_at_other {Value : Type} (first : Env Value) (bindings : List Binding)
    (values : List (Runtime Value)) (name : Nat) (fresh : name ∉ names bindings) :
    assign first bindings values name = first name := by
  induction bindings generalizing first values with
  | nil => rfl
  | cons binding rest induction =>
    cases values with
    | nil => rfl
    | cons value tail =>
      have different : name ≠ binding.name := fun equal => fresh (by simp [names, equal])
      have restFresh : name ∉ names rest := fun member => fresh (by simp only [names, List.map_cons, List.mem_cons]; exact Or.inr member)
      exact (induction (put first binding.name value) tail restFresh).trans (ite_eq_right different)

theorem assign_agree {Value : Type} (reads : List Nat) (first second : Env Value)
    (bindings : List Binding) (values : List (Runtime Value)) (typed : ValuesHold bindings values)
    (same : AgreeReads (without reads (names bindings)) first second) :
    AgreeReads reads (assign first bindings values) (assign second bindings values) := by
  induction typed generalizing reads first second with
  | nil =>
    intro name member
    exact same name (by simpa [without, names] using member)
  | @cons binding value rest remaining holds typed induction =>
    apply induction
    apply agree_put
    intro name member
    apply same
    simp only [without, List.mem_filter, decide_eq_true_eq] at member ⊢
    rcases member with ⟨⟨read, restFresh⟩, fresh⟩
    refine ⟨read, ?_⟩
    simpa only [names, List.map_cons, List.mem_cons, not_or] using
      And.intro (by simpa using fresh) restFresh

theorem agree_arguments {Value : Type} (aliases : Aliases) (operands : List Binding) (first second : Env Value)
    (same : AgreeReads (names operands) first second) :
    readArguments false aliases first operands = readArguments false aliases second operands := by
  apply List.map_congr_left
  intro binding member
  exact same binding.name (List.mem_map_of_mem member)

/-- No assumption on retained operations is needed: they see the same explicit
arguments/world and extensionally equal nested region meanings. -/
theorem eval_free_reads {Value World : Type} (meaning : Interpretation Value World) (code : Code)
    (aliases : Aliases) (first second : Env Value) (same : AgreeReads (freeReads code) first second)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases code first before outputs after ↔ Eval meaning false aliases code second before outputs after := by
  induction code generalizing aliases first second before outputs after with
  | returned values =>
    simp only [Eval, agree_arguments aliases values first second same]
  | cast origin result source target rest induction =>
    have read := same origin.name (by simp [freeReads])
    have remaining : AgreeReads (without (freeReads rest) [result.name]) first second :=
      fun name member => same name (List.mem_cons_of_mem _ member)
    simp only [Eval, Bool.false_eq_true, ↓reduceIte, read]
    apply exists_congr
    intro value
    exact and_congr_right (fun _ => induction _ _ _ (agree_put _ _ _ _ value remaining) _ _ _)
  | dimension buffer axis result type axisIndex value rest induction =>
    have bufferRead := same buffer.name (by simp [freeReads])
    have axisRead := same axis.name (by simp [freeReads])
    have remaining : AgreeReads (without (freeReads rest) [result.name]) first second :=
      fun name member => same name (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member))
    simp only [Eval, Bool.false_eq_true, ↓reduceIte, bufferRead, axisRead]
    apply exists_congr
    intro actual
    exact and_congr_right (fun _ => induction _ _ _ (agree_put _ _ _ _ (.index actual) remaining) _ _ _)
  | constant result value rest induction =>
    exact induction _ _ _ (agree_put _ _ _ _ (.index value) same) _ _ _
  | action key operands results rest induction =>
    have arguments := agree_arguments aliases operands first second
      (fun name member => same name (List.mem_append_left _ member))
    have remaining : AgreeReads (without (freeReads rest) (names results)) first second :=
      fun name member => same name (List.mem_append_right _ member)
    simp only [Eval, arguments]
    apply exists_congr; intro values
    apply exists_congr; intro middle
    apply and_congr_right; intro _
    apply and_congr_right; intro typed
    exact induction _ _ _ (assign_agree _ _ _ _ _ typed remaining) _ _ _
  | region key operands parameters results body rest bodyInduction restInduction =>
    have arguments := agree_arguments aliases operands first second (fun name member =>
      same name (List.mem_append_left _ (List.mem_append_left _ member)))
    have captured : AgreeReads (without (freeReads body) (names parameters)) first second := fun name member =>
      same name (List.mem_append_left _ (List.mem_append_right _ member))
    have remaining : AgreeReads (without (freeReads rest) (names results)) first second :=
      fun name member => same name (List.mem_append_right _ member)
    have bodySame : (fun inputs initial outputs final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases aliases parameters) body (assign first parameters inputs) initial outputs final) =
        (fun inputs initial outputs final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases aliases parameters) body (assign second parameters inputs) initial outputs final) := by
      funext inputs initial returned final
      apply propext
      apply and_congr_right; intro typed
      exact bodyInduction _ _ _ (assign_agree _ _ _ _ _ typed captured) _ _ _
    simp only [Eval, arguments, bodySame]
    apply exists_congr; intro values
    apply exists_congr; intro middle
    apply and_congr_right; intro _
    apply and_congr_right; intro typed
    exact restInduction _ _ _ (assign_agree _ _ _ _ _ typed remaining) _ _ _

end Merlin.GraphLoopsShape.Epoch
