/- Independent integer bounds for W8A8 accumulation.
These do not certify a quantization policy or its MLIR implementation. -/
import Std.Tactic

namespace MerlinVerify.Rules.W8A8Bounds

/-- A product of symmetric signed quantized values fits the stated magnitude. -/
theorem product_bound (a b : Int)
    (ha : -127 ≤ a ∧ a ≤ 127) (hb : -127 ≤ b ∧ b ≤ 127) :
    -16129 ≤ a * b ∧ a * b ≤ 16129 := by
  by_cases ha0 : 0 ≤ a
  · have hlo := Int.mul_le_mul_of_nonneg_left hb.1 ha0
    have hhi := Int.mul_le_mul_of_nonneg_left hb.2 ha0
    constructor <;> omega
  · have ha0' : a ≤ 0 := by omega
    have hlo := Int.mul_le_mul_of_nonpos_left ha0' hb.2
    have hhi := Int.mul_le_mul_of_nonpos_left ha0' hb.1
    constructor <;> omega

/-- Bounds every prefix, including the empty prefix, without modular arithmetic. -/
theorem sum_bound (xs : List Int)
    (h : ∀ x ∈ xs, -16129 ≤ x ∧ x ≤ 16129) :
    -(xs.length : Int) * 16129 ≤ xs.sum ∧ xs.sum ≤ (xs.length : Int) * 16129 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx := h x (by simp)
    have ht := ih (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    constructor <;> omega

/-- A strict length bound suffices for signed i32 representability of every prefix. -/
theorem sum_fits_i32 (xs : List Int)
    (h : ∀ x ∈ xs, -16129 ≤ x ∧ x ≤ 16129)
    (hk : (xs.length : Int) * 16129 < 2147483648) :
    -2147483648 ≤ xs.sum ∧ xs.sum < 2147483648 := by
  have hs := sum_bound xs h
  omega

#print axioms product_bound
#print axioms sum_bound
#print axioms sum_fits_i32

end MerlinVerify.Rules.W8A8Bounds
