/- Integer-unit magnitude bounds for the actual ordered binary32 computation.
This additive family preserves the earlier qualified checker and its assumptions. -/
import MerlinVerify.Rules.W8A8OrderedRounding

namespace MerlinVerify.Rules.W8A8OrderedMagnitude

open MerlinVerify Float.Model.UnpackedFloat W8A8IEEE32
open Float.Model (UnpackedFloat)

theorem aligned_units (m : Nat) {e common : Int} (lower : -149 ≤ common) (aligned : common ≤ e) :
    (m <<< (e - common).toNat) * 2^(common + 149).toNat = m * 2^(e + 149).toNat := by
  have exponent : (e - common).toNat + (common + 149).toNat = (e + 149).toNat := by omega
  rw [Nat.shiftLeft_eq, Nat.mul_assoc, ← Nat.pow_add, exponent]

theorem normalize_bounded {value : Int} {e : Int} {limit : Nat} (sign : Sign)
    (lower : -149 ≤ e) (limitSafe : limit ≤ 252)
    (bound : value.natAbs * 2^(e + 149).toNat < 2^(limit + 24)) :
    Finite (W8A8IEEE32.pack (normalize format value e sign)) ∧
      units (W8A8IEEE32.pack (normalize format value e sign)) ≤
        value.natAbs * 2^(e + 149).toNat + 2^limit := by
  cases value with
  | ofNat n =>
    cases n with
    | zero =>
      change Finite (W8A8IEEE32.pack (.zero sign)) ∧
        units (W8A8IEEE32.pack (.zero sign)) ≤ 0 * 2^(e + 149).toNat + 2^limit
      exact ⟨(zero_units sign).1, by rw [(zero_units sign).2]; exact Nat.zero_le _⟩
    | succ n =>
      change Finite (W8A8IEEE32.pack (normalize format ((n+1 : Nat) : Int) e sign)) ∧
        units (W8A8IEEE32.pack (normalize format ((n+1 : Nat) : Int) e sign)) ≤
          (n+1) * 2^(e + 149).toNat + 2^limit
      have order : compare ((n+1 : Nat) : Int) 0 = .gt := Int.compare_eq_gt.2 (by omega)
      simp only [normalize, order, Int.toNat_natCast]
      exact round_bounded (Nat.succ_pos n) lower limitSafe bound
  | negSucc n =>
    have order : compare (Int.negSucc n) 0 = .lt := Int.compare_eq_lt.2 (by omega)
    simp only [normalize, order, Int.neg_negSucc, Int.toNat_natCast]
    exact round_bounded (Nat.succ_pos n) lower limitSafe bound

theorem apply_natAbs (s : Sign) (m : Nat) : (s.apply m).natAbs = m := by
  cases s <;> simp [Sign.apply]

theorem repack_bounded {x : FBits .f32} (finite : Finite x) :
    Finite (W8A8IEEE32.pack (W8A8IEEE32.unpack x)) ∧
      units (W8A8IEEE32.pack (W8A8IEEE32.unpack x)) ≤ units x := by
  have nonNaN := unpack_notNaN (finite_notNaN finite)
  have notInf := unpack_notInf finite
  cases decoded : W8A8IEEE32.unpack x with
  | notANumber => exact False.elim (nonNaN decoded)
  | infinity sign => exact False.elim (notInf sign decoded)
  | zero sign =>
    exact ⟨(zero_units sign).1, by rw [(zero_units sign).2]; exact Nat.zero_le _⟩
  | finite sign m e positive =>
    obtain ⟨lower, exactUnits⟩ := unpack_units decoded
    have upper := (W8A8IEEEBounds.unpack_finite_bounds
      (show W8A8IEEEBounds.ExponentLE 254 x from ⟨finite, by omega⟩) decoded).2
    change e ≤ 104 at upper
    rw [exactUnits]
    exact ⟨W8A8IEEEBounds.pack_finite_exponent upper, pack_units_le lower upper⟩

/-- Swapping the two inputs of this specific pinned operation preserves its bits.
This is not a law for an arbitrary FloatModel or an accumulation reassociation. -/
theorem add_comm (x y : FBits .f32) : W8A8IEEE32.add x y = W8A8IEEE32.add y x := by
  unfold W8A8IEEE32.add
  cases dx : W8A8IEEE32.unpack x <;> cases dy : W8A8IEEE32.unpack y <;> try rfl
  case infinity.infinity sx sy => cases sx <;> cases sy <;> rfl
  case zero.zero sx sy => cases sx <;> cases sy <;> rfl
  case finite.finite sx mx ex px sy my ey py =>
    simp only [UnpackedFloat.add, decreaseExponent, Int.min_comm, Int.add_comm]

/-- Bound the actual signed, exponent-aligned addition and its actual rounding.
No reassociation or abstract rounding-error law is used. -/
theorem add_bounded {x y : FBits .f32} {limit : Nat}
    (left : Finite x) (right : Finite y) (limitSafe : limit ≤ 252)
    (bound : units x + units y < 2^(limit + 24)) :
    Finite (W8A8IEEE32.add x y) ∧ units (W8A8IEEE32.add x y) ≤ units x + units y + 2^limit := by
  have nx := unpack_notNaN (finite_notNaN left)
  have ny := unpack_notNaN (finite_notNaN right)
  have ix := unpack_notInf left
  have iy := unpack_notInf right
  cases dx : W8A8IEEE32.unpack x with
  | notANumber => exact False.elim (nx dx)
  | infinity sign => exact False.elim (ix sign dx)
  | zero sx =>
    cases dy : W8A8IEEE32.unpack y with
    | notANumber => exact False.elim (ny dy)
    | infinity sign => exact False.elim (iy sign dy)
    | zero sy =>
      simp only [W8A8IEEE32.add, dx, dy, UnpackedFloat.add]
      split
      · exact ⟨(zero_units sx).1, by rw [(zero_units sx).2]; exact Nat.zero_le _⟩
      · exact ⟨(zero_units .positive).1, by rw [(zero_units .positive).2]; exact Nat.zero_le _⟩
    | finite sy my ey py =>
      have repack := repack_bounded right
      simp only [dy] at repack
      simp only [W8A8IEEE32.add, dx, dy, UnpackedFloat.add]
      exact ⟨repack.1, Nat.le_trans repack.2
        (Nat.le_trans (Nat.le_add_left _ _) (Nat.le_add_right _ _))⟩
  | finite sx mx ex px =>
    cases dy : W8A8IEEE32.unpack y with
    | notANumber => exact False.elim (ny dy)
    | infinity sign => exact False.elim (iy sign dy)
    | zero sy =>
      have repack := repack_bounded left
      simp only [dx] at repack
      simp only [W8A8IEEE32.add, dx, dy, UnpackedFloat.add]
      exact ⟨repack.1, Nat.le_trans repack.2
        (Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _))⟩
    | finite sy my ey py =>
      obtain ⟨lowerX, unitsX⟩ := unpack_units dx
      obtain ⟨lowerY, unitsY⟩ := unpack_units dy
      let common := min ex ey
      let value := sx.apply (mx <<< (ex - common).toNat) + sy.apply (my <<< (ey - common).toNat)
      have commonLower : -149 ≤ common := by dsimp only [common]; omega
      have leftAligned : common ≤ ex := by dsimp only [common]; omega
      have rightAligned : common ≤ ey := by dsimp only [common]; omega
      have sumBound : value.natAbs * 2^(common + 149).toNat ≤ units x + units y := by
        have triangle := Nat.mul_le_mul_right (2^(common + 149).toNat)
          (Int.natAbs_add_le (sx.apply (mx <<< (ex - common).toNat))
            (sy.apply (my <<< (ey - common).toNat)))
        rw [apply_natAbs, apply_natAbs, Nat.add_mul, aligned_units mx commonLower leftAligned,
          aligned_units my commonLower rightAligned, ← unitsX, ← unitsY] at triangle
        exact triangle
      have normalized := normalize_bounded .positive commonLower limitSafe (Nat.lt_of_le_of_lt sumBound bound)
      simp only [W8A8IEEE32.add, dx, dy, UnpackedFloat.add, decreaseExponent]
      change Finite (W8A8IEEE32.pack (normalize format value common .positive)) ∧
        units (W8A8IEEE32.pack (normalize format value common .positive)) ≤ units x + units y + 2^limit
      exact ⟨normalized.1, Nat.le_trans normalized.2 (Nat.add_le_add_right sumBound _)⟩

/-- Keep one product unit and one conservative rounding unit per actual addition.
The count limit leaves room below the binary32 precision threshold used above. -/
theorem add_counted {x y : FBits .f32} {limit count : Nat}
    (left : Finite x) (right : Finite y) (limitSafe : limit ≤ 252) (countSafe : count < 2^22)
    (accumulator : units x ≤ 2 * count * 2^limit) (term : units y ≤ 2^limit) :
    Finite (W8A8IEEE32.add x y) ∧ units (W8A8IEEE32.add x y) ≤ 2 * (count + 1) * 2^limit := by
  have sumBound : units x + units y ≤ (2 * count + 1) * 2^limit := by
    rw [Nat.add_mul, Nat.one_mul]
    exact Nat.add_le_add accumulator term
  have coefficient : 2 * count + 1 < 2^24 := by omega
  have strict : (2 * count + 1) * 2^limit < 2^(limit + 24) := by
    have fact := Nat.mul_lt_mul_of_pos_right coefficient (Nat.two_pow_pos limit)
    simpa only [Nat.pow_add, Nat.mul_comm] using fact
  obtain ⟨finite, bounded⟩ := add_bounded left right limitSafe (Nat.lt_of_le_of_lt sumBound strict)
  refine ⟨finite, Nat.le_trans bounded ?_⟩
  calc
    units x + units y + 2^limit ≤ 2 * count * 2^limit + 2^limit + 2^limit :=
      Nat.add_le_add_right (Nat.add_le_add accumulator term) _
    _ = 2 * (count + 1) * 2^limit := by
      simp only [Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.two_mul, Nat.add_assoc]

theorem foldl_counted {limit : Nat} (limitSafe : limit ≤ 252) (xs : List (FBits .f32)) :
    ∀ (start : Nat) (initial : FBits .f32),
      (∀ value ∈ xs, Finite value ∧ units value ≤ 2^limit) →
      Finite initial → units initial ≤ 2 * start * 2^limit → start + xs.length ≤ 2^22 →
      Finite (xs.foldl W8A8IEEE32.add initial) ∧
        units (xs.foldl W8A8IEEE32.add initial) ≤ 2 * (start + xs.length) * 2^limit := by
  induction xs with
  | nil =>
    intro start initial _ finite bounded _
    simpa only [List.foldl_nil, List.length_nil, Nat.add_zero] using And.intro finite bounded
  | cons value rest ih =>
    intro start initial terms finite bounded count
    have head := terms value (by simp)
    have next := add_counted finite head.1 limitSafe (show start < 2^22 from by simp only [List.length_cons] at count; omega)
      bounded head.2
    have remaining : ∀ value ∈ rest, Finite value ∧ units value ≤ 2^limit := by
      intro value member
      exact terms value (by simp [member])
    have remainingCount : start + 1 + rest.length ≤ 2^22 := by
      simp only [List.length_cons] at count
      omega
    have result := ih (start + 1) (W8A8IEEE32.add initial value) remaining next.1 next.2 remainingCount
    have countEq : start + 1 + rest.length = start + (value :: rest).length := by
      simp only [List.length_cons]
      omega
    simpa only [List.foldl_cons, countEq] using result

theorem sum_counted {limit : Nat} (limitSafe : limit ≤ 252) (xs : List (FBits .f32))
    (terms : ∀ value ∈ xs, Finite value ∧ units value ≤ 2^limit) (countSafe : xs.length ≤ 2^22) :
    Finite (xs.foldl W8A8IEEE32.add 0#32) ∧
      units (xs.foldl W8A8IEEE32.add 0#32) ≤ 2 * xs.length * 2^limit := by
  have result := foldl_counted limitSafe xs 0 0#32 terms (by decide)
    (by change 0 ≤ 0 * 2^limit; simp) (by simpa using countSafe)
  simpa only [Nat.zero_add] using result

def sumCap (limit count : Nat) : Nat :=
  if count = 0 then 0 else limit + (2 * count).log2 + 1 - 23

theorem sum_exponent_bounded {limit : Nat} (limitSafe : limit ≤ 252) (xs : List (FBits .f32))
    (terms : ∀ value ∈ xs, Finite value ∧ units value ≤ 2^limit) (countSafe : xs.length ≤ 2^22) :
    W8A8IEEEBounds.ExponentLE (sumCap limit xs.length) (xs.foldl W8A8IEEE32.add 0#32) := by
  by_cases empty : xs.length = 0
  · have nil : xs = [] := List.length_eq_zero_iff.1 empty
    subst xs
    change W8A8IEEEBounds.ExponentLE 0 0#32
    exact ⟨by decide, by decide⟩
  · obtain ⟨finite, bounded⟩ := sum_counted limitSafe xs terms countSafe
    refine ⟨finite, ?_⟩
    simp only [sumCap, ite_eq_right empty]
    apply exponent_le_of_units
    apply Nat.lt_of_le_of_lt bounded
    have coefficient : 2 * xs.length < 2^((2 * xs.length).log2 + 1) := Nat.lt_log2_self
    have exponent : limit + (2 * xs.length).log2 + 1 = (2 * xs.length).log2 + 1 + limit := by omega
    rw [exponent, Nat.pow_add]
    exact Nat.mul_lt_mul_of_pos_right coefficient (Nat.two_pow_pos limit)

#print axioms aligned_units
#print axioms normalize_bounded
#print axioms apply_natAbs
#print axioms repack_bounded
#print axioms add_comm
#print axioms add_bounded
#print axioms add_counted
#print axioms foldl_counted
#print axioms sum_counted
#print axioms sum_exponent_bounded

end MerlinVerify.Rules.W8A8OrderedMagnitude
