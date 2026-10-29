/- Source-only structural witness helpers, specialized by emitted actual-body
equations. These are not new operational semantics or successful-body axioms. -/
import Merlin.GraphStatePipelineTail

namespace Merlin.GraphState.Pipeline.Tail
set_option autoImplicit false
open Scalar

theorem progression_length (lower step count : Nat) :
    (progression lower step count).length = count := by
  induction count generalizing lower with
  | zero => rfl
  | succ count ih => simp only [progression, List.length_cons, ih]

theorem progression_member {lower step count index : Nat}
    (member : index ∈ progression lower step count) :
    ∃ ordinal, ordinal < count ∧ index = lower + step * ordinal := by
  induction count generalizing lower with
  | zero => cases member
  | succ count ih =>
    simp only [progression,List.mem_cons] at member
    rcases member with same | later
    · exact ⟨0,by omega,by simpa using same⟩
    · obtain ⟨ordinal,inside,equal⟩ := ih later
      refine ⟨ordinal+1,by omega,?_⟩
      rw [equal,Nat.mul_add,Nat.mul_one]
      omega

/-- A structural success witness, with a local equation closed from the
instantiated actual body. No trip-sized literal evaluation is required. -/
theorem constant_carry_run {Error : Type} (arity : Nat) (values : List Int)
    (body : Int → List Int → Nat → Outcome Int Nat Error)
    (size : values.length = arity) (indices : List Int)
    (bodyStep : ∀ i w, i ∈ indices → body i values w = .ok (values,w+1)) (world : Nat) :
    loopEval arity body indices values world = .ok (values,world+indices.length) := by
  induction indices generalizing world with
  | nil => simp only [loopEval,size,↓reduceIte,List.length_nil,Nat.add_zero]
  | cons index rest ih =>
    simp only [loopEval,size,↓reduceIte,bodyStep index world (by simp),Except.bind]
    rw [ih (fun i w member => bodyStep i w (by simp [member]))]
    simp only [List.length_cons]
    congr 1
    omega

/-- First fault after a successful prefix, with no later call or observation.
The prefix equation is discharged structurally in generated witness queries. -/
theorem fault_after_prefix {Error : Type} (arity : Nat) (values : List Int)
    (body : Int → List Int → Nat → Outcome Int Nat Error)
    (size : values.length = arity) (prefix rest : List Int) (faultIndex : Int)
    (bodyStep : ∀ i w, i ∈ prefix → body i values w = .ok (values,w+1))
    (failure : Fault Error) (fault : ∀ w, body faultIndex values w = .error (failure,w+1))
    (world : Nat) :
    loopEval arity body (prefix ++ faultIndex::rest) values world =
      .error (failure,world+prefix.length+1) := by
  rw [loop_append,constant_carry_run arity values body size prefix bodyStep world]
  simp only [Except.bind,loopEval,size,↓reduceIte,fault]
  rfl

end Merlin.GraphState.Pipeline.Tail
