import Merlin.GraphStateScalarSound

namespace Merlin.GraphState.Pipeline.Context
set_option autoImplicit false
open Scalar

def EquivalentAt {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (count : Nat) (left right : Program) : Prop :=
  ∀ environment world, environment.length = count →
    evalProgram model left environment world = evalProgram model right environment world

theorem refl {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (count : Nat) (program : Program) :
    EquivalentAt model count program program := by
  intro environment world size
  rfl

theorem trans {Value World Error : Type} {registry : List Signature}
    {model : Model Value World Error registry} {count : Nat} {first middle last : Program}
    (left : EquivalentAt model count first middle) (right : EquivalentAt model count middle last) :
    EquivalentAt model count first last := by
  intro environment world size
  exact (left environment world size).trans (right environment world size)

theorem pure_congr {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (count : Nat) (op : Prim) (arguments : List Nat)
    {left right : Program} (same : EquivalentAt model (count+1) left right) :
    EquivalentAt model count (.pure op arguments left) (.pure op arguments right) := by
  intro environment world size
  apply same
  simp only [List.length_cons,size]

theorem opaque_congr {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (count identity : Nat)
    (arguments : List Nat) (outputs : List Ty) {left right : Program}
    (same : EquivalentAt model ((signature registry identity).outputs.length+count) left right) :
    EquivalentAt model count (.opaque identity arguments outputs left) (.opaque identity arguments outputs right) := by
  intro environment world size
  simp only [evalProgram]
  cases run : opaqueStep model identity (reads model.default environment arguments) world with
  | error failure => rfl
  | ok result =>
    rcases result with ⟨values,final⟩
    simp only [Except.bind]
    apply same
    rw [List.length_append,opaqueStep_arity model run,size]

theorem loop_congr {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (count lower upper stride : Nat)
    (initial : List Nat) (carries : List Ty) {leftBody rightBody leftNext rightNext : Program}
    (body : EquivalentAt model (carries.length+count+1) leftBody rightBody)
    (next : EquivalentAt model (carries.length+count) leftNext rightNext) :
    EquivalentAt model count (.loop lower upper stride initial carries leftBody leftNext)
      (.loop lower upper stride initial carries rightBody rightNext) := by
  intro environment world size
  let indices := model.iterations (environment[lower]?.getD model.default)
    (environment[upper]?.getD model.default) (environment[stride]?.getD model.default)
  let first := fun index values state =>
    evalProgram model leftBody (model.index index::values++environment) state
  let second := fun index values state =>
    evalProgram model rightBody (model.index index::values++environment) state
  have bodies : ∀ index values state, values.length = carries.length →
      first index values state = second index values state := by
    intro index values state arity
    apply body
    simp only [List.length_cons,List.length_append,arity,size]
    omega
  have loops := loopEval_congr carries.length first second bodies indices
    (reads model.default environment initial) world
  change (loopEval carries.length first indices (reads model.default environment initial) world).bind _ =
    (loopEval carries.length second indices (reads model.default environment initial) world).bind _
  rw [loops]
  cases run : loopEval carries.length second indices (reads model.default environment initial) world with
  | error failure => rfl
  | ok result =>
    rcases result with ⟨values,final⟩
    simp only [Except.bind]
    apply next
    rw [List.length_append,loopEval_arity carries.length second indices _ world run,size]

end Merlin.GraphState.Pipeline.Context

namespace Merlin.GraphState.Pipeline.Context
set_option autoImplicit false
open Scalar

/-- Add only the continuation spine; nested loop bodies retain their own yields. -/
def plug : Program → Program → Program
  | .ret _, next => next
  | .pure op arguments later, next => .pure op arguments (plug later next)
  | .opaque identity arguments outputs later, next => .opaque identity arguments outputs (plug later next)
  | .loop lower upper stride initial carries body later, next =>
      .loop lower upper stride initial carries body (plug later next)

def returnSlots : Program → List Nat
  | .ret values => values
  | .pure _ _ next | .opaque _ _ _ next | .loop _ _ _ _ _ _ next => returnSlots next

/-- Only top-level result bindings grow the enclosing private SSA frame. -/
def frameGrowth (registry : List Signature) : Program → Nat
  | .ret _ => 0
  | .pure _ _ next => 1 + frameGrowth registry next
  | .opaque identity _ _ next => (signature registry identity).outputs.length + frameGrowth registry next
  | .loop _ _ _ _ carries _ next => carries.length + frameGrowth registry next

/-- Checked suffix capture permits erasing every other private segment cell.
The pointwise suffix premise is discharged by Scope.check_sound in actual queries;
this theorem does not assume that the segment or any external body succeeds. -/
theorem segment_factor {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (segment suffix observer : Program)
    (count : Nat) (environment : List Value) (world : World)
    (size : environment.length = count)
    (capture : ∀ final state, final.length = count + frameGrowth registry segment →
      evalProgram model suffix final state =
        evalProgram model observer (reads model.default final (returnSlots segment)) state) :
    evalProgram model (plug segment suffix) environment world =
      (evalProgram model segment environment world).bind
        (fun result => evalProgram model observer result.1 result.2) := by
  induction segment generalizing count environment world with
  | ret values =>
    simpa only [plug,evalProgram,Except.bind,returnSlots] using
      capture environment world (by simpa only [frameGrowth,Nat.add_zero] using size)
  | pure op arguments later ih =>
    simp only [plug,evalProgram]
    apply ih (count+1) _ world
    · simpa only [List.length_cons,size]
    · intro final state arity
      apply capture final state
      simpa only [frameGrowth,Nat.add_assoc] using arity
  | «opaque» identity arguments outputs later ih =>
    simp only [plug,evalProgram]
    cases executed : opaqueStep model identity (reads model.default environment arguments) world with
    | error failure => rfl
    | ok result =>
      rcases result with ⟨values,terminal⟩
      simp only [Except.bind]
      apply ih (count+(signature registry identity).outputs.length) (values++environment) terminal
      · rw [List.length_append,opaqueStep_arity model executed,size,Nat.add_comm]
      · intro final state arity
        apply capture final state
        simpa only [frameGrowth,Nat.add_assoc] using arity
  | loop lower upper stride initial carries body later ihBody ihNext =>
    simp only [plug,evalProgram]
    cases executed : loopEval carries.length
        (fun index values state => evalProgram model body (model.index index::values++environment) state)
        (model.iterations (environment[lower]?.getD model.default)
          (environment[upper]?.getD model.default) (environment[stride]?.getD model.default))
        (reads model.default environment initial) world with
    | error failure => rfl
    | ok result =>
      rcases result with ⟨values,terminal⟩
      simp only [Except.bind]
      apply ihNext (count+carries.length) (values++environment) terminal
      · rw [List.length_append,loopEval_arity _ _ _ _ _ executed,size,Nat.add_comm]
      · intro final state arity
        apply capture final state
        simpa only [frameGrowth,Nat.add_assoc] using arity

end Merlin.GraphState.Pipeline.Context
