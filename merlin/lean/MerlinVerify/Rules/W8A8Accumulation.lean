/-
Integer execution underlying Merlin's W8A8 contraction, including the full signed i8 range.
This connects modular i32 arithmetic to an integer dot product. It does not certify the
quantization policy, tensor indexing, initialization, or the enclosing transformation.
-/
import Std.Tactic

namespace MerlinVerify.Rules.W8A8Accumulation

abbrev Pair := BitVec 8 × BitVec 8

/-- The operation order emitted by Merlin: signed extension, product, then product + accumulator. -/
def step (acc : BitVec 32) (p : Pair) : BitVec 32 :=
  p.1.signExtend 32 * p.2.signExtend 32 + acc

def dot (xs : List Pair) : Int := (xs.map fun p => p.1.toInt * p.2.toInt).sum

def accumulate (xs : List Pair) (initial : BitVec 32 := 0) : BitVec 32 :=
  xs.foldl step initial

/-- The full byte range includes -128; an i8 type alone does not imply symmetric quantization. -/
theorem product_bound (p : Pair) :
    -16384 ≤ p.1.toInt * p.2.toInt ∧ p.1.toInt * p.2.toInt ≤ 16384 := by
  have lo := BitVec.le_toInt_mul_toInt (x := p.1) (y := p.2)
  have hi := BitVec.toInt_mul_toInt_le (x := p.1) (y := p.2)
  exact ⟨lo, hi⟩

theorem dot_bound (xs : List Pair) :
    -(xs.length : Int) * 16384 ≤ dot xs ∧ dot xs ≤ (xs.length : Int) * 16384 := by
  induction xs with
  | nil => simp [dot]
  | cons p ps ih =>
    have hp := product_bound p
    simp only [dot, List.map_cons, List.sum_cons] at *
    simp only [List.length_cons]
    constructor <;> omega

theorem step_ofInt (initial : Int) (p : Pair) :
    step (BitVec.ofInt 32 initial) p = BitVec.ofInt 32 (p.1.toInt * p.2.toInt + initial) := by
  have ext (x : BitVec 8) : x.signExtend 32 = BitVec.ofInt 32 x.toInt := by
    rw [← BitVec.toInt_signExtend_of_le (x := x) (show 8 ≤ 32 by decide)]
    simp
  simp only [step, ext, ← BitVec.ofInt_mul, ← BitVec.ofInt_add]

/-- Exact modular execution, with no overflow or operand-range premise. -/
theorem accumulate_modular (xs : List Pair) (initial : BitVec 32) :
    accumulate xs initial = BitVec.ofInt 32 (dot xs + initial.toInt) := by
  have fold (ys : List Pair) (z : Int) :
      accumulate ys (BitVec.ofInt 32 z) = BitVec.ofInt 32 (dot ys + z) := by
    induction ys generalizing z with
    | nil => simp [accumulate, dot]
    | cons p ps ih =>
      simp only [accumulate, List.foldl_cons, step_ofInt] at *
      rw [ih]
      simp only [dot, List.map_cons, List.sum_cons]
      congr 1
      omega
  simpa only [BitVec.ofInt_toInt] using fold xs initial.toInt

/-- A static reduction-length bound makes a zero-initialized i32 dot product exact over integers. -/
theorem exact_of_length (xs : List Pair)
    (hk : (xs.length : Int) * 16384 < 2147483648) :
    (accumulate xs).toInt = dot xs := by
  have hb := dot_bound xs
  rw [accumulate_modular]
  have hzero : (0 : BitVec 32).toInt = 0 := by decide
  rw [hzero, Int.add_zero]
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

/-- The same bound covers every intermediate accumulator, not just the final result. -/
theorem every_prefix_exact (xs : List Pair)
    (hk : (xs.length : Int) * 16384 < 2147483648) (n : Nat) :
    (accumulate (xs.take n)).toInt = dot (xs.take n) := by
  apply exact_of_length
  have hlen := List.length_take_le' n xs
  omega

#print axioms accumulate_modular
#print axioms exact_of_length
#print axioms every_prefix_exact

end MerlinVerify.Rules.W8A8Accumulation
