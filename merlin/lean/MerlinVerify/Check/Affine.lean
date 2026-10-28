/-
Static bounds of indexing maps over an iteration box (design §4.5, progress).

A structured op reads and writes its operands through affine maps evaluated at every point of its
iteration box `0 ≤ d_k < ranges[k]`; an access outside an operand is `stuck`. `affineBounds`
computes an interval containing an expression's value at every point of the box (interval
arithmetic, so it may be wider than the exact range but never narrower), and `mapInBounds` checks
that every result of a map lies inside the operand's shape. `mapInBounds_sound`: on every point of
the box the map evaluates to an in-bounds index.

Only expressions whose evaluation is total on the box get bounds: dimensions of the box, constants,
sums, products with a constant, and `mod`/`floordiv`/`ceildiv` by a positive constant. Symbols and
any other product or divisor are refused.
-/
import MerlinVerify.Sem.Facts

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

/-- Interval of `x * c` for `x ∈ [l, h]`. -/
def scaleBounds (c : Int) (b : Int × Int) : Int × Int :=
  if 0 ≤ c then (b.1 * c, b.2 * c) else (b.2 * c, b.1 * c)

/-- The value of a constant expression. -/
def constOf? : AffineExpr → Option Int
  | .const c => some c
  | _ => none

theorem constOf?_eq {e : AffineExpr} {c : Int} (h : constOf? e = some c) : e = .const c := by
  cases e <;> simp_all [constOf?]

/-- An interval containing the value of `e` at every point of the box `ranges`, if `e` is total on
the box. -/
def affineBounds (ranges : List Nat) : AffineExpr → Option (Int × Int)
  | .dim p =>
    match ranges[p]? with
    | some r => some (0, (r : Int) - 1)
    | none => none
  | .sym _ => none
  | .const c => some (c, c)
  | .add a b => do
    let (la, ha) ← affineBounds ranges a
    let (lb, hb) ← affineBounds ranges b
    pure (la + lb, ha + hb)
  | .mul a b =>
    match constOf? b, constOf? a with
    | some c, _ => (affineBounds ranges a).map (scaleBounds c)
    | none, some c => (affineBounds ranges b).map (scaleBounds c)
    | none, none => none
  | .mod a b =>
    match constOf? b with
    | some c => if 0 < c then (affineBounds ranges a).map fun _ => (0, c - 1) else none
    | none => none
  | .floordiv a b =>
    match constOf? b with
    | some c =>
      if 0 < c then (affineBounds ranges a).map fun (l, h) => (Int.fdiv l c, Int.fdiv h c) else none
    | none => none
  | .ceildiv a b =>
    match constOf? b with
    | some c =>
      if 0 < c then (affineBounds ranges a).map fun (l, h) => (-Int.fdiv (-l) c, -Int.fdiv (-h) c)
      else none
    | none => none

theorem scaleBounds_sound {c v l h : Int} (hl : l ≤ v) (hh : v ≤ h) :
    (scaleBounds c (l, h)).1 ≤ v * c ∧ v * c ≤ (scaleBounds c (l, h)).2 := by
  unfold scaleBounds
  split
  · next hc => exact ⟨Int.mul_le_mul_of_nonneg_right hl hc, Int.mul_le_mul_of_nonneg_right hh hc⟩
  · next hc =>
    have hc' : c ≤ 0 := by omega
    exact ⟨Int.mul_le_mul_of_nonpos_right hh hc', Int.mul_le_mul_of_nonpos_right hl hc'⟩

theorem fdiv_mono {a b c : Int} (hc : 0 < c) (h : a ≤ b) : Int.fdiv a c ≤ Int.fdiv b c := by
  rw [Int.fdiv_eq_ediv_of_nonneg _ (Int.le_of_lt hc), Int.fdiv_eq_ediv_of_nonneg _ (Int.le_of_lt hc)]
  exact Int.ediv_le_ediv hc h

/-- Bounds of a map over an interval, and soundness of the underlying expression, give bounds of
the mapped expression. -/
theorem bounds_map {ranges pt : List Nat} {a : AffineExpr} {f : Int × Int → Int × Int}
    {lo hi : Int} (h : (affineBounds ranges a).map f = some (lo, hi))
    (ih : ∀ {lo hi : Int}, affineBounds ranges a = some (lo, hi) →
      ∃ v, evalAffine pt a = some v ∧ lo ≤ v ∧ v ≤ hi) :
    ∃ l h' v, evalAffine pt a = some v ∧ l ≤ v ∧ v ≤ h' ∧ f (l, h') = (lo, hi) := by
  cases ha : affineBounds ranges a with
  | none => simp [ha] at h
  | some b =>
    simp only [ha, Option.map_some, Option.some.injEq] at h
    obtain ⟨v, hv, hl, hh⟩ := ih (lo := b.1) (hi := b.2) ha
    exact ⟨b.1, b.2, v, hv, hl, hh, h⟩

/-- **Interval soundness.** At a point of the box, an expression with bounds evaluates, to a value
within them. -/
theorem affineBounds_sound {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) :
    ∀ (e : AffineExpr) {lo hi : Int}, affineBounds ranges e = some (lo, hi) →
      ∃ v, evalAffine pt e = some v ∧ lo ≤ v ∧ v ≤ hi := by
  intro e
  induction e with
  | dim p =>
    intro lo hi h
    simp only [affineBounds] at h
    split at h
    · next r hr =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      have hp : p < ranges.length := (List.getElem?_eq_some_iff.1 hr).1
      have hr' : ranges[p] = r := (List.getElem?_eq_some_iff.1 hr).2
      have hpl : p < pt.length := by rw [hpt.1]; exact hp
      have hlt := hpt.2 p hpl hp
      refine ⟨pt[p], by simp [evalAffine, List.getElem?_eq_getElem hpl], by omega, ?_⟩
      rw [hr'] at hlt
      omega
    · cases h
  | sym _ => intro lo hi h; simp [affineBounds] at h
  | const c =>
    intro lo hi h
    simp only [affineBounds, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨c, rfl, Int.le_refl _, Int.le_refl _⟩
  | add a b iha ihb =>
    intro lo hi h
    simp only [affineBounds] at h
    cases ha : affineBounds ranges a with
    | none => simp [ha] at h
    | some pa =>
      cases hb : affineBounds ranges b with
      | none => simp [ha, hb] at h
      | some pb =>
        simp only [ha, hb, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
          Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨va, hva, hla, hha⟩ := iha (lo := pa.1) (hi := pa.2) ha
        obtain ⟨vb, hvb, hlb, hhb⟩ := ihb (lo := pb.1) (hi := pb.2) hb
        refine ⟨va + vb, ?_, Int.add_le_add hla hlb, Int.add_le_add hha hhb⟩
        simp [evalAffine, hva, hvb]
  | mul a b iha ihb =>
    intro lo hi h
    simp only [affineBounds] at h
    split at h
    · next c hc =>
      obtain rfl := constOf?_eq hc
      obtain ⟨l, h', v, hv, hl, hh, hs⟩ := bounds_map h iha
      have := scaleBounds_sound (c := c) hl hh
      rw [hs] at this
      exact ⟨v * c, by simp [evalAffine, hv], this.1, this.2⟩
    · next c _ hc =>
      obtain rfl := constOf?_eq hc
      obtain ⟨l, h', v, hv, hl, hh, hs⟩ := bounds_map h ihb
      have := scaleBounds_sound (c := c) hl hh
      rw [hs, Int.mul_comm v c] at this
      exact ⟨c * v, by simp [evalAffine, hv], this.1, this.2⟩
    · cases h
  | mod a b iha _ =>
    intro lo hi h
    simp only [affineBounds] at h
    split at h
    · next c hc =>
      obtain rfl := constOf?_eq hc
      split at h
      · next hpos =>
        obtain ⟨l, h', v, hv, -, -, hs⟩ := bounds_map h iha
        simp only [Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        refine ⟨Int.fmod v c, ?_, Int.fmod_nonneg_of_pos v hpos, by
          have := Int.fmod_lt_of_pos v hpos; omega⟩
        simp [evalAffine, hv, Int.ne_of_gt hpos]
      · cases h
    · cases h
  | floordiv a b iha _ =>
    intro lo hi h
    simp only [affineBounds] at h
    split at h
    · next c hc =>
      obtain rfl := constOf?_eq hc
      split at h
      · next hpos =>
        obtain ⟨l, h', v, hv, hl, hh, hs⟩ := bounds_map h iha
        simp only [Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        refine ⟨Int.fdiv v c, ?_, fdiv_mono hpos hl, fdiv_mono hpos hh⟩
        simp [evalAffine, hv, Int.ne_of_gt hpos]
      · cases h
    · cases h
  | ceildiv a b iha _ =>
    intro lo hi h
    simp only [affineBounds] at h
    split at h
    · next c hc =>
      obtain rfl := constOf?_eq hc
      split at h
      · next hpos =>
        obtain ⟨l, h', v, hv, hl, hh, hs⟩ := bounds_map h iha
        simp only [Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        refine ⟨-Int.fdiv (-v) c, ?_, ?_, ?_⟩
        · simp [evalAffine, hv, Int.ne_of_gt hpos]
        · exact Int.neg_le_neg (fdiv_mono hpos (Int.neg_le_neg hl))
        · exact Int.neg_le_neg (fdiv_mono hpos (Int.neg_le_neg hh))
      · cases h
    · cases h

/-! ### Maps -/

/-- Every access of `m` over the box `ranges` lies inside an operand of shape `shape`. -/
def mapInBounds (ranges : List Nat) (m : AffineMap) (shape : List Nat) : Bool :=
  m.results.length == shape.length &&
    (m.results.zip shape).all fun (e, s) =>
      match affineBounds ranges e with
      | some (lo, hi) => decide (0 ≤ lo) && decide (hi < s)
      | none => false

theorem toIdx_cons (s : Nat) (ss : List Nat) (x : Int) (xs : List Int) :
    toIdx (s :: ss) (x :: xs) =
      if 0 ≤ x ∧ x < s then (toIdx ss xs).map (x.toNat :: ·) else none := by
  unfold toIdx
  by_cases hl : xs.length = ss.length
  · have hl' : (x :: xs).length = (s :: ss).length := by simp [hl]
    simp only [hl', hl, ne_eq, not_true_eq_false, ↓reduceIte, List.zip_cons_cons, List.mapM_cons]
    split
    · cases (xs.zip ss).mapM (fun (x, s) => if 0 ≤ x ∧ x < (s : Int) then some x.toNat else none) <;>
        rfl
    · rfl
  · have hl' : ¬(x :: xs).length = (s :: ss).length := by simpa using hl
    simp only [hl', hl, ne_eq, not_false_eq_true, ↓reduceIte, Option.map_none, ite_self]

/-- Results bounded inside `shape` evaluate to an in-bounds index. -/
theorem results_inBounds {ranges pt : List Nat} (hpt : Idx.InBounds ranges pt) :
    ∀ (es : List AffineExpr) (shape : List Nat), es.length = shape.length →
      ((es.zip shape).all fun (e, s) =>
        match affineBounds ranges e with
        | some (lo, hi) => decide (0 ≤ lo) && decide (hi < s)
        | none => false) = true →
      ∃ i, (es.mapM (evalAffine pt) >>= toIdx shape) = some i ∧ Idx.InBounds shape i
  | [], [], _, _ => ⟨[], rfl, (Idx.inBounds_nil_iff []).2 rfl⟩
  | e :: es, s :: ss, hl, h => by
    simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at h
    obtain ⟨he, hes⟩ := h
    split at he
    · next lo hi hb =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at he
      obtain ⟨v, hv, hlo, hhi⟩ := affineBounds_sound hpt e hb
      obtain ⟨i, hi, hin⟩ := results_inBounds hpt es ss (by simpa using hl) hes
      cases hm : es.mapM (evalAffine pt) with
      | none => simp [hm] at hi
      | some xs =>
        simp only [hm, Option.bind_eq_bind, Option.bind_some] at hi
        refine ⟨v.toNat :: i, ?_, ?_⟩
        · simp only [List.mapM_cons, hv, hm, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
            toIdx_cons, hi]
          rw [ite_eq_left ⟨by omega, by omega⟩]
          rfl
        · exact (Idx.inBounds_cons_cons _ _ _ _).2 ⟨by omega, hin⟩
    · cases he
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

/-- **Map soundness.** At every point of the box, a map accepted by `mapInBounds` evaluates to an
index inside the operand. -/
theorem mapInBounds_sound {ranges pt : List Nat} {m : AffineMap} {shape : List Nat}
    (h : mapInBounds ranges m shape = true) (hpt : Idx.InBounds ranges pt) :
    ∃ i, (evalMap m pt >>= toIdx shape) = some i ∧ Idx.InBounds shape i := by
  simp only [mapInBounds, Bool.and_eq_true, beq_iff_eq] at h
  exact results_inBounds hpt m.results shape h.1 h.2

end MerlinVerify.Check
