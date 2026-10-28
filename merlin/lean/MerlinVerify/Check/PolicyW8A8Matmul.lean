/- Checked tensor contraction component. The enclosing policy checker must still establish
source correspondence, actual operand types and zero initialization from the real prefix. -/
import MerlinVerify.Rules.W8A8Matmul

namespace MerlinVerify.Check.PolicyW8A8Matmul

open MerlinVerify MerlinVerify.Sem.Ops.Linalg
open Rules

def operandTypes (m n k : Nat) : List VTy :=
  [.tensor [m, k] (.int 8), .tensor [k, n] (.int 8), .tensor [m, n] (.int 32)]

def leftAt (m k : Nat) (left : Idx → Option (BitVec 8)) (point : Idx) : Option (BitVec 8) :=
  left (W8A8IntegerTensor.coordinate [m, k] W8A8Matmul.leftMap point)

def rightAt (n k : Nat) (right : Idx → Option (BitVec 8)) (point : Idx) : Option (BitVec 8) :=
  right (W8A8IntegerTensor.coordinate [k, n] W8A8Matmul.rightMap point)

def reference (m n k : Nat) (left right : Idx → Option (BitVec 8))
    (initial : Idx → Option (BitVec 32)) : Tensor :=
  Tensor.canon [m, n] (.int 32) (fun cell => W8A8IntegerBody.value 32
    (W8A8TensorTrace.accumulate (W8A8IntegerTensor.coordinate [m, n] W8A8Matmul.outMap)
      (W8A8IntegerTensor.step (leftAt m k left) (rightAt n k right)) initial
      (allPoints [m, n, k]) cell))

/-- State the semantic component contract before its executable checker. -/
def Correct (m n k : Nat) (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Prop :=
  ∀ ctx call env left right initial,
    genericSem ctx maps iters
      [.tensor (W8A8IntegerTensor.tensor 8 [m, k] left),
        .tensor (W8A8IntegerTensor.tensor 8 [k, n] right)]
      [.tensor (W8A8IntegerTensor.tensor 32 [m, n] initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (reference m n k left right initial)]

def checkLayout (m n k : Nat) (maps : List AffineMap) (iters : List IterKind) : Bool :=
  -- 131072 * 16384 = 2^31. The strict bound covers every full-range signed-byte product.
  decide (k < 131072) &&
  decide (maps = W8A8Matmul.maps) && decide (iters = W8A8Matmul.iters) &&
  (structuredRanges? maps iters.length (operandTypes m n k) == some [m, n, k])

theorem checkLayout_sound {m n k : Nat} {maps : List AffineMap} {iters : List IterKind}
    (h : checkLayout m n k maps iters = true) :
    (k : Int) * 16384 < 2147483648 ∧ maps = W8A8Matmul.maps ∧ iters = W8A8Matmul.iters ∧
      StructuredOk maps iters.length (operandTypes m n k) [m, n, k] := by
  simp only [checkLayout, Bool.and_eq_true] at h
  have hk := of_decide_eq_true h.1.1.1
  exact ⟨by omega, of_decide_eq_true h.1.1.2, of_decide_eq_true h.1.2,
    structuredRanges?_spec (eq_of_beq h.2)⟩

def check (m n k : Nat) (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Bool :=
  checkLayout m n k maps iters && PolicyW8A8Body.check actual

/-- The body, complete mapped loop and all runtime access bounds are established by acceptance.
No facts about floating-point quantization or captured operands follow from this component. -/
theorem check_sound {m n k : Nat} {maps : List AffineMap} {iters : List IterKind} {actual : Region}
    (h : check m n k maps iters actual = true) : Correct m n k maps iters actual := by
  have hh := h
  simp only [check, Bool.and_eq_true] at hh
  obtain ⟨hl, hb⟩ := hh
  obtain ⟨_, hm, hi, geometry⟩ := checkLayout_sound hl
  intro ctx call env left right initial
  subst maps
  subst iters
  exact W8A8IntegerTensor.checked_generic_sound (ib := ctx.indexBits)
    geometry hb call env left right initial

/-- With zero-initialized destination and initialized signed-byte inputs, each actual result cell
is initialized and its signed i32 value is the integer dot product. The checker supplies K. -/
theorem check_zero_sound {m n k : Nat} {maps : List AffineMap} {iters : List IterKind}
    {actual : Region} (h : check m n k maps iters actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (left right : Idx → BitVec 8) :
    ∃ result : Tensor,
      genericSem ctx maps iters
        [.tensor (W8A8IntegerTensor.tensor 8 [m, k] (some ∘ left)),
          .tensor (W8A8IntegerTensor.tensor 8 [k, n] (some ∘ right))]
        [.tensor (W8A8IntegerTensor.tensor 32 [m, n] (fun _ => some 0))]
        (fun c => evalRegion Sem.Ops.opSem c call env actual) = .ok [.tensor result] ∧
      result.shape = [m, n] ∧
      ∀ cell, Idx.InBounds [m, n] cell → ∃ bits : BitVec 32,
        result.get cell = .some (.bits 32 bits) ∧ bits.toInt =
          W8A8Accumulation.dot (W8A8IntegerTensor.pairs
            (W8A8IntegerTensor.coordinate [m, n] W8A8Matmul.outMap)
            (fun p => left (W8A8IntegerTensor.coordinate [m, k] W8A8Matmul.leftMap p))
            (fun p => right (W8A8IntegerTensor.coordinate [k, n] W8A8Matmul.rightMap p))
            (allPoints [m, n, k]) cell) := by
  refine ⟨reference m n k (some ∘ left) (some ∘ right) (fun _ => some 0),
    check_sound h ctx call env _ _ _, rfl, ?_⟩
  intro cell hc
  let a := fun p => left (W8A8IntegerTensor.coordinate [m, k] W8A8Matmul.leftMap p)
  let b := fun p => right (W8A8IntegerTensor.coordinate [k, n] W8A8Matmul.rightMap p)
  let ps := W8A8IntegerTensor.pairs (W8A8IntegerTensor.coordinate [m, n] W8A8Matmul.outMap)
    a b (allPoints [m, n, k]) cell
  have hh := h
  simp only [check, Bool.and_eq_true] at hh
  have bound := (checkLayout_sound hh.1).1
  have count : ps.length = k := W8A8Matmul.mapped_pair_count m n k a b cell hc
  refine ⟨W8A8Accumulation.accumulate ps, ?_, ?_⟩
  · simp only [reference, Tensor.canon, hc, ↓reduceIte]
    change W8A8IntegerBody.value 32
      (W8A8TensorTrace.accumulate (W8A8IntegerTensor.coordinate [m, n] W8A8Matmul.outMap)
        (W8A8IntegerTensor.step (some ∘ a) (some ∘ b)) (some ∘ (fun _ => 0))
        (allPoints [m, n, k]) cell) = _
    rw [W8A8IntegerTensor.initialized_cell]
    rfl
  · apply W8A8Accumulation.exact_of_length
    rwa [count]

#print axioms checkLayout_sound
#print axioms check_sound
#print axioms check_zero_sound

end MerlinVerify.Check.PolicyW8A8Matmul
