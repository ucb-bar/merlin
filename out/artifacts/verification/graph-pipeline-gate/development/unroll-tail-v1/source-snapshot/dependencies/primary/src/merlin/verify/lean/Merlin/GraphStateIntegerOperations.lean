/- Exact word representation, using existing core BitVec facts. No new axiom
or native numerical agreement is asserted. This explicitly chooses unsigned
canonical Int representatives for signless finite words. -/
import Merlin.GraphStateIntegerRepresentation

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar

@[simp] theorem ofInt_word (width : Nat) (value : Int) :
    BitVec.ofInt width (word width value) = BitVec.ofInt width value := by
  change BitVec.ofNat width (BitVec.ofInt width value).toNat = BitVec.ofInt width value
  simpa only [BitVec.setWidth_eq] using BitVec.ofNat_toNat width (BitVec.ofInt width value)

@[simp] theorem word_word (width : Nat) (value : Int) :
    word width (word width value) = word width value := by
  change ((BitVec.ofInt width (word width value)).toNat : Int) = _
  rw [ofInt_word]
  rfl

theorem word_bounds (width : Nat) (value : Int) :
    0 ≤ word width value ∧ word width value < (2^width : Nat) := by
  have bound := (BitVec.ofInt width value).isLt
  constructor
  · exact Int.natCast_nonneg _
  · exact Int.ofNat_lt.mpr bound

theorem canonical_iff (width : Nat) (value : Int) :
    Canonical (.bits width) value ↔ 0 ≤ value ∧ value < (2^width : Nat) := by
  constructor
  · intro canonical
    have bounds := word_bounds width value
    change word width value = value at canonical
    rw [canonical] at bounds
    exact bounds
  · intro bounds
    simp only [Canonical,word,BitVec.toNat_ofInt]
    rw [Int.emod_eq_of_lt bounds.1 bounds.2,Int.toNat_of_nonneg bounds.1]

theorem word_add (width : Nat) (left right : Int) :
    word width (word width left + word width right) = word width (left+right) := by
  change ((BitVec.ofInt width (word width left + word width right)).toNat : Int) = _
  simp only [BitVec.ofInt_add,ofInt_word]
  rw [← BitVec.ofInt_add]
  rfl

theorem word_sub (width : Nat) (left right : Int) :
    word width (word width left - word width right) = word width (left-right) := by
  change ((BitVec.ofInt width (word width left - word width right)).toNat : Int) = _
  simp only [Int.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg,ofInt_word]
  rw [← BitVec.ofInt_neg,← BitVec.ofInt_add]
  rfl

theorem word_mul (width : Nat) (left right : Int) :
    word width (word width left * word width right) = word width (left*right) := by
  change ((BitVec.ofInt width (word width left * word width right)).toNat : Int) = _
  simp only [BitVec.ofInt_mul,ofInt_word]
  rw [← BitVec.ofInt_mul]
  rfl

def primitive (op : Prim) (arguments : List Int) : Int :=
  let first := arguments[0]?.getD 0
  let second := arguments[1]?.getD 0
  match op with
  | .constant width value => word width value
  | .add width => word width (first+second)
  | .sub width => word width (first-second)
  | .mul width => word width (first*second)
  | .zext _ target | .trunc _ target => word target first
  | .sext source target =>
    let unsigned := word source first
    word target (if unsigned < (2^(source-1) : Int) then unsigned else unsigned-(2^source : Int))

theorem primitive_canonical (op : Prim) (arguments : List Int) :
    Canonical op.output (primitive op arguments) := by
  cases op <;> simp only [Prim.output,Canonical,primitive,word_word]

theorem add_zero (width : Nat) (value : Int) (canonical : Canonical (.bits width) value) :
    primitive (.add width) [value,word width 0] = value := by
  simpa [primitive,word,Canonical] using canonical

theorem zero_add (width : Nat) (value : Int) (canonical : Canonical (.bits width) value) :
    primitive (.add width) [word width 0,value] = value := by
  simpa [primitive,word,Canonical] using canonical

theorem sub_zero (width : Nat) (value : Int) (canonical : Canonical (.bits width) value) :
    primitive (.sub width) [value,word width 0] = value := by
  simpa [primitive,word,Canonical] using canonical

theorem mul_one (width : Nat) (value : Int) (canonical : Canonical (.bits width) value) :
    primitive (.mul width) [value,word width 1] = value := by
  simp only [primitive,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
  change ((BitVec.ofInt width (value * word width 1)).toNat : Int) = value
  rw [BitVec.ofInt_mul,ofInt_word]
  simp only [BitVec.ofInt_ofNat,BitVec.mul_one]
  exact canonical

theorem one_mul (width : Nat) (value : Int) (canonical : Canonical (.bits width) value) :
    primitive (.mul width) [word width 1,value] = value := by
  simp only [primitive,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
  change ((BitVec.ofInt width (word width 1 * value)).toNat : Int) = value
  rw [BitVec.ofInt_mul,ofInt_word]
  simp only [BitVec.ofInt_ofNat,BitVec.one_mul]
  exact canonical

#print axioms canonical_iff
#print axioms word_add
#print axioms word_sub
#print axioms word_mul
#print axioms mul_one
end Merlin.GraphState.Integer
