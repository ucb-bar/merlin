/- Bounds for the actual ordered floating multiply/add trace. Products are rounded before
addition and every accumulator update retains the original order. -/
import MerlinVerify.Rules.W8A8OrderedSum

namespace MerlinVerify.Rules.W8A8OrderedDot

open MerlinVerify W8A8IEEEBounds W8A8OrderedMagnitude

abbrev Pair := FBits .f32 × FBits .f32

def product (pair : Pair) : FBits .f32 := W8A8IEEE32.mul pair.1 pair.2
def step (accumulator : FBits .f32) (pair : Pair) : FBits .f32 :=
  W8A8IEEE32.add accumulator (product pair)

def evaluate (pairs : List Pair) (initial : FBits .f32) : FBits .f32 :=
  pairs.foldl step initial

def productLimit (left right : Nat) : Nat := mulCap left right + 23
def cap (left right count : Nat) : Nat := sumCap (productLimit left right) count

def admissible (left right count : Nat) : Bool :=
  decide (productLimit left right ≤ 252 ∧ count ≤ 2^22)

theorem evaluate_products (pairs : List Pair) (initial : FBits .f32) :
    evaluate pairs initial = (pairs.map product).foldl W8A8IEEE32.add initial := by
  induction pairs generalizing initial with
  | nil => rfl
  | cons pair rest ih =>
    simpa only [evaluate, List.foldl_cons, List.map_cons, step] using
      ih (W8A8IEEE32.add initial (product pair))

theorem product_bounded {pair : Pair} {left right : Nat}
    (leftBound : ExponentLE left pair.1) (rightBound : ExponentLE right pair.2)
    (safe : productLimit left right ≤ 252) :
    W8A8IEEE32.Finite (product pair) ∧ units (product pair) ≤ 2^(productLimit left right) := by
  have arithmeticSafe : max left 1 + max right 1 ≤ 379 := by
    dsimp only [productLimit, mulCap] at safe
    omega
  have bounded := W8A8IEEEBounds.mul_bounded leftBound rightBound arithmeticSafe
  exact ⟨bounded.1, Nat.le_of_lt (units_lt_of_exponent bounded)⟩

theorem evaluate_bounded {pairs : List Pair} {left right : Nat}
    (inputs : ∀ pair ∈ pairs, ExponentLE left pair.1 ∧ ExponentLE right pair.2)
    (safe : admissible left right pairs.length = true) :
    ExponentLE (cap left right pairs.length) (evaluate pairs 0#32) := by
  have limits : productLimit left right ≤ 252 ∧ pairs.length ≤ 2^22 := of_decide_eq_true safe
  have terms : ∀ value ∈ pairs.map product,
      W8A8IEEE32.Finite value ∧ units value ≤ 2^(productLimit left right) := by
    intro value member
    obtain ⟨pair, found, equal⟩ := List.mem_map.1 member
    subst value
    exact product_bounded (inputs pair found).1 (inputs pair found).2 limits.1
  have result := sum_exponent_bounded limits.1 (pairs.map product) terms
    (by simpa only [List.length_map] using limits.2)
  have folded : ExponentLE (cap left right pairs.length)
      ((pairs.map product).foldl W8A8IEEE32.add 0#32) := by
    simpa only [List.length_map, cap] using result
  exact Eq.mpr (congrArg (fun value => ExponentLE (cap left right pairs.length) value)
    (evaluate_products pairs 0#32)) folded

theorem prefix_bounded {pairs : List Pair} {left right : Nat}
    (inputs : ∀ pair ∈ pairs, ExponentLE left pair.1 ∧ ExponentLE right pair.2)
    (safe : admissible left right pairs.length = true) (count : Nat) :
    ExponentLE (cap left right (pairs.take count).length) (evaluate (pairs.take count) 0#32) := by
  have limits : productLimit left right ≤ 252 ∧ pairs.length ≤ 2^22 := of_decide_eq_true safe
  apply evaluate_bounded
  · intro pair member
    exact inputs pair (List.mem_of_mem_take member)
  · apply decide_eq_true
    exact ⟨limits.1, by rw [List.length_take]; omega⟩

#print axioms product_bounded
#print axioms evaluate_bounded
#print axioms prefix_bounded

end MerlinVerify.Rules.W8A8OrderedDot
