/- The local numerical domain contains initialized operands at every admitted shape and format.
This is a mathematical witness, independent of sample tensors and the optimization output. -/
import MerlinVerify.Rules.Bf16F32AccTensor
import MerlinVerify.Sem.Defined

namespace MerlinVerify.Rules.Bf16F32AccNonvacuity

open MerlinVerify MerlinVerify.Check

def zeros (shape : List Nat) (kind : FKind) : RVal :=
  .tensor (Tensor.full shape (.float kind) (.some (.flt kind 0)))

theorem zeros_typed (ib : Nat) (shape : List Nat) (kind : FKind) :
    RVal.HasTy ib (zeros shape kind) (.tensor shape (.float kind)) := by
  exact hasTy_full (by simp [Val.HasTy, Scalar.conforms])

theorem zeros_defined (shape : List Nat) (kind : FKind) : (zeros shape kind).Defined := by
  intro cell inside
  simp only [zeros, Tensor.full, Tensor.canon] at inside ⊢
  simp [inside]

/-- Universal qualification of a contraction's typed initialized-input domain cannot succeed
only because that domain is empty, including shapes with a zero extent. -/
theorem operand_domain_nonempty (ib : Nat) (left right : List Nat) (kind : FKind) :
    ∃ args, HasTys ib args [.tensor left (.float kind), .tensor right (.float kind)] ∧
      ResultsDefined args := by
  refine ⟨[zeros left kind, zeros right kind],
    .cons (zeros_typed ib left kind) (.cons (zeros_typed ib right kind) .nil), ?_⟩
  intro value member
  rcases List.mem_cons.1 member with rfl | member
  · exact zeros_defined left kind
  · rcases List.mem_cons.1 member with rfl | member
    · exact zeros_defined right kind
    · cases member

#print axioms operand_domain_nonempty

end MerlinVerify.Rules.Bf16F32AccNonvacuity
