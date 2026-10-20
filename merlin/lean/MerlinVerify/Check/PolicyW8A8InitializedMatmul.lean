/- Connect the checked real initializer to the tensor MAC component, including canonical
out-of-bounds poison. Source operands and the surrounding policy transformation remain separate. -/
import MerlinVerify.Check.PolicyW8A8Zero
import MerlinVerify.Check.PolicyW8A8Matmul

namespace MerlinVerify.Check.PolicyW8A8InitializedMatmul

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Rules

def canonicalInitial (m n : Nat) (cell : Idx) : Option (BitVec 32) :=
  if Idx.InBounds [m, n] cell then some 0 else none

private theorem initial_tensor (m n : Nat) :
    W8A8IntegerTensor.tensor 32 [m, n] (canonicalInitial m n) =
      Tensor.full [m, n] (.int 32) W8A8Zero.zero := by
  unfold W8A8IntegerTensor.tensor W8A8TensorTrace.pack Tensor.full Tensor.canon
  congr 1
  funext cell
  by_cases hc : Idx.InBounds [m, n] cell
  all_goals simp [canonicalInitial, W8A8IntegerBody.value, hc, W8A8Zero.zero_eq]

private theorem reference_initial (m n k : Nat) (left right : Idx → Option (BitVec 8)) :
    PolicyW8A8Matmul.reference m n k left right (canonicalInitial m n) =
      PolicyW8A8Matmul.reference m n k left right (fun _ => some 0) := by
  apply Tensor.canon_ext
  intro cell hc
  simp only [W8A8TensorTrace.accumulate_cell]
  simp [canonicalInitial, hc]

/-- An accepted tensor contraction computes the same result from the canonical zero tensor
proved by the initializer checker as from the functional zero destination of its exactness proof. -/
theorem canonical_zero {m n k : Nat} {maps : List AffineMap} {iters : List IterKind}
    {actual : Region} (h : PolicyW8A8Matmul.check m n k maps iters actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (left right : Idx → Option (BitVec 8)) :
    genericSem ctx maps iters
      [.tensor (W8A8IntegerTensor.tensor 8 [m, k] left),
        .tensor (W8A8IntegerTensor.tensor 8 [k, n] right)]
      [.tensor (Tensor.full [m, n] (.int 32) W8A8Zero.zero)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
    genericSem ctx maps iters
      [.tensor (W8A8IntegerTensor.tensor 8 [m, k] left),
        .tensor (W8A8IntegerTensor.tensor 8 [k, n] right)]
      [.tensor (W8A8IntegerTensor.tensor 32 [m, n] (fun _ => some 0))]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) := by
  rw [← initial_tensor, PolicyW8A8Matmul.check_sound h ctx call env,
    PolicyW8A8Matmul.check_sound h ctx call env, reference_initial]

/-- Prefix execution supplies the actual destination value. No zero-initialization premise is
left to the caller: both the producer and its selected SSA binding are checked. -/
theorem checked_destination {m n k : Nat} {maps : List AffineMap} {iters : List IterKind}
    {actual : Region} {initializers : List Op} {destination : Nat}
    (hi : PolicyW8A8Zero.check m n initializers destination = true)
    (hc : PolicyW8A8Matmul.check m n k maps iters actual = true)
    (ctx : Ctx) (call : CallSem) (env env' : Env)
    (executed : execOps Sem.Ops.opSem ctx call env initializers = .ok env')
    (left right : Idx → Option (BitVec 8)) :
    ∃ output : RVal, env'.get? destination = some output ∧
      genericSem ctx maps iters
        [.tensor (W8A8IntegerTensor.tensor 8 [m, k] left),
          .tensor (W8A8IntegerTensor.tensor 8 [k, n] right)] [output]
        (fun c => evalRegion Sem.Ops.opSem c call env' actual) =
        .ok [.tensor (PolicyW8A8Matmul.reference m n k left right (fun _ => some 0))] := by
  refine ⟨.tensor (Tensor.full [m, n] (.int 32) W8A8Zero.zero),
    PolicyW8A8Zero.check_binding_sound hi ctx call env env' executed, ?_⟩
  rw [canonical_zero hc, PolicyW8A8Matmul.check_sound hc ctx call env']

#print axioms canonical_zero
#print axioms checked_destination

end MerlinVerify.Check.PolicyW8A8InitializedMatmul
