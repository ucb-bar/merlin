/- Connect an actual literal-zero initializer to an accepted abs/max reduction. Positive
zero is checked by bits; no floating identity or numerical law is introduced. -/
import MerlinVerify.Check.PolicyW8A8Fill
import MerlinVerify.Check.PolicyW8A8AbsMax

namespace MerlinVerify.Check.PolicyW8A8InitializedAbsMax

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Rules W8A8Scalar W8A8TensorOperands
open PolicyW8A8Geometry

def literal : Attr := .float 0 (.float .f32)
def zero : Val := .some (.flt .f32 0#32)

def ZeroCorrect (shape : List Nat) (actual : List Op) (destination : Nat) : Prop :=
  ∀ ctx call env env', execOps Sem.Ops.opSem ctx call env actual = .ok env' →
    env'.get? destination = some (.tensor (Tensor.full shape (.float .f32) zero))

def checkZero (shape : List Nat) (actual : List Op) (destination : Nat) : Bool :=
  PolicyW8A8Fill.check shape (.float .f32) literal actual destination

theorem zero_binding {shape : List Nat} {actual : List Op} {destination : Nat}
    (h : checkZero shape actual destination = true) : ZeroCorrect shape actual destination := by
  intro ctx call env env' executed
  exact PolicyW8A8Fill.check_binding_sound h ctx call env env' (.flt .f32 0#32) rfl executed

def canonicalInitial (shape : List Nat) (cell : Idx) : Option (FBits .f32) :=
  if Idx.InBounds shape cell then some 0#32 else none

private theorem initial_tensor (shape : List Nat) :
    W8A8TensorTrace.pack shape (.float .f32) floatValue (canonicalInitial shape) =
      Tensor.full shape (.float .f32) zero := by
  unfold W8A8TensorTrace.pack Tensor.full Tensor.canon
  congr 1
  funext cell
  by_cases hc : Idx.InBounds shape cell
  all_goals simp [canonicalInitial, floatValue, hc, zero]

def reference (M : FloatModel) (shape kept : List Nat) (get : Idx → Option (FBits .f32)) : Tensor :=
  Tensor.canon (parameterShape shape kept) (.float .f32) (fun cell => floatValue
    (W8A8TensorTrace.accumulate
      (coordinate (parameterShape shape kept) (parameterMap shape.length kept))
      (W8A8AbsMax.step M get) (fun _ => some 0#32) (allPoints shape) cell))

private theorem reference_initial (M : FloatModel) (shape kept : List Nat)
    (get : Idx → Option (FBits .f32)) :
    Tensor.canon (parameterShape shape kept) (.float .f32) (fun cell => floatValue
      (W8A8TensorTrace.accumulate
        (coordinate (parameterShape shape kept) (parameterMap shape.length kept))
        (W8A8AbsMax.step M get) (canonicalInitial (parameterShape shape kept))
        (allPoints shape) cell)) = reference M shape kept get := by
  apply Tensor.canon_ext
  intro cell hc
  simp only [W8A8TensorTrace.accumulate_cell]
  simp [canonicalInitial, hc]

/-- Both the producer and reduction are checked. The actual destination lookup is established
by execution of that producer, rather than supplied as a zero-tensor premise. -/
theorem checked_destination {shape kept : List Nat} {maps : List AffineMap}
    {iters : List IterKind} {actual : Region} {initializers : List Op} {destination : Nat}
    (hi : checkZero (parameterShape shape kept) initializers destination = true)
    (hc : PolicyW8A8AbsMax.check shape kept maps iters actual = true)
    (ctx : Ctx) (call : CallSem) (env env' : Env)
    (executed : execOps Sem.Ops.opSem ctx call env initializers = .ok env')
    (get : Idx → Option (FBits .f32)) :
    ∃ output : RVal, env'.get? destination = some output ∧
      genericSem ctx maps iters [(W8A8FloatTensor.floatInput shape get).value] [output]
        (fun c => evalRegion Sem.Ops.opSem c call env' actual) =
        .ok [.tensor (reference ctx.M shape kept get)] := by
  refine ⟨.tensor (Tensor.full (parameterShape shape kept) (.float .f32) zero),
    zero_binding hi ctx call env env' executed, ?_⟩
  rw [← initial_tensor, PolicyW8A8AbsMax.check_sound hc ctx call env', reference_initial]

#print axioms zero_binding
#print axioms checked_destination

end MerlinVerify.Check.PolicyW8A8InitializedAbsMax
