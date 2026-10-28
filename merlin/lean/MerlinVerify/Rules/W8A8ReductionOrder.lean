/- Relate actual input-first reduction bodies to the numerical row folds. Byte
maximum commutation is proved only for non-NaNs; NaN payload order stays observable. -/
import MerlinVerify.Rules.W8A8ReductionCells
import MerlinVerify.Rules.W8A8SoftmaxRow

namespace MerlinVerify.Rules.W8A8ReductionOrder

open MerlinVerify W8A8ReductionBody W8A8MaskedMaximum

private theorem bits_eq_of_fields {left right : FBits .f32}
    (signs : W8A8ByteFloat.negative left = W8A8ByteFloat.negative right)
    (magnitudes : W8A8ByteFloat.magnitudeWord left = W8A8ByteFloat.magnitudeWord right) : left = right := by
  apply BitVec.eq_of_toNat_eq
  have leftBound := left.isLt
  have rightBound := right.isLt
  change left.toNat < 4294967296 at leftBound
  change right.toNat < 4294967296 at rightBound
  unfold W8A8ByteFloat.magnitudeWord at magnitudes
  cases leftSign : W8A8ByteFloat.negative left with
  | false =>
    have rightSign := signs.symm.trans leftSign
    unfold W8A8ByteFloat.negative at leftSign rightSign
    simp only [decide_eq_false_iff_not] at leftSign rightSign
    omega
  | true =>
    have rightSign := signs.symm.trans leftSign
    unfold W8A8ByteFloat.negative at leftSign rightSign
    simp only [decide_eq_true_eq] at leftSign rightSign
    omega

private theorem neither_precedes {left right : FBits .f32}
    (forward : W8A8ByteFloat.precedes left right = false)
    (backward : W8A8ByteFloat.precedes right left = false) : left = right := by
  cases leftSign : W8A8ByteFloat.negative left with
  | false =>
    cases rightSign : W8A8ByteFloat.negative right with
    | false =>
      apply bits_eq_of_fields (leftSign.trans rightSign.symm)
      simp only [W8A8ByteFloat.precedes, leftSign, rightSign, Bool.false_eq_true,
        ↓reduceIte, decide_eq_false_iff_not] at forward backward
      omega
    | true => simp [W8A8ByteFloat.precedes, leftSign, rightSign] at backward
  | true =>
    cases rightSign : W8A8ByteFloat.negative right with
    | false => simp [W8A8ByteFloat.precedes, leftSign, rightSign] at forward
    | true =>
      apply bits_eq_of_fields (leftSign.trans rightSign.symm)
      simp only [W8A8ByteFloat.precedes, leftSign, rightSign, ↓reduceIte,
        decide_eq_false_iff_not] at forward backward
      omega

private theorem precedes_asymmetric {left right : FBits .f32}
    (forward : W8A8ByteFloat.precedes left right = true) :
    W8A8ByteFloat.precedes right left = false := by
  cases leftSign : W8A8ByteFloat.negative left <;> cases rightSign : W8A8ByteFloat.negative right
  all_goals simp only [W8A8ByteFloat.precedes, leftSign, rightSign, Bool.false_eq_true,
    ↓reduceIte, decide_eq_true_eq, decide_eq_false_iff_not] at forward ⊢
  all_goals omega

theorem maximum_nonNaN_commute {left right : FBits .f32}
    (leftNotNaN : W8A8IEEE32.NonNaN left) (rightNotNaN : W8A8IEEE32.NonNaN right) :
    W8A8ByteFloat.maximum left right = W8A8ByteFloat.maximum right left := by
  simp only [W8A8ByteFloat.maximum, leftNotNaN, rightNotNaN, Bool.false_eq_true, ↓reduceIte]
  cases forward : W8A8ByteFloat.precedes left right with
  | false =>
    cases backward : W8A8ByteFloat.precedes right left with
    | false => simpa only [forward, backward, Bool.false_eq_true, ↓reduceIte] using neither_precedes forward backward
    | true => simp only [Bool.false_eq_true, ↓reduceIte]
  | true => simp only [precedes_asymmetric forward, Bool.false_eq_true, ↓reduceIte]

theorem admissible_nonNaN {cap : Nat} {bits : FBits .f32} (admissible : Admissible cap bits) :
    W8A8IEEE32.NonNaN bits := by
  rcases admissible with rfl | bounded
  · decide
  · exact W8A8IEEE32.finite_notNaN bounded.1

theorem maximum_fold {cap : Nat} (words : List (FBits .f32)) :
    ∀ initial, Admissible cap initial → (∀ word ∈ words, Admissible cap word) →
      W8A8ReductionCells.reduce .maximum words initial = words.foldl W8A8ByteFloat.maximum initial := by
  induction words with
  | nil => intro _ _ _; rfl
  | cons head tail induction =>
    intro initial initialized members
    have headAdmissible := members head (List.mem_cons_self ..)
    have commute := maximum_nonNaN_commute (admissible_nonNaN headAdmissible) (admissible_nonNaN initialized)
    simp only [W8A8ReductionCells.reduce, List.foldl_cons, combine]
    rw [commute]
    exact induction _ (maximum_admissible initialized headAdmissible)
      (fun word member => members word (List.mem_cons_of_mem _ member))

theorem sum_fold (words : List (FBits .f32)) (initial : FBits .f32) :
    W8A8ReductionCells.reduce .sum words initial = words.foldl W8A8IEEE32.add initial := by
  induction words generalizing initial with
  | nil => rfl
  | cons head tail induction =>
    simp only [W8A8ReductionCells.reduce, List.foldl_cons, combine]
    rw [W8A8OrderedMagnitude.add_comm head initial]
    exact induction _

end MerlinVerify.Rules.W8A8ReductionOrder
