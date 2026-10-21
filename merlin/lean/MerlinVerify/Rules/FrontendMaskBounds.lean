/- Runtime-shape and mask-selection invariants for the inactive candidate. -/
import MerlinVerify.Rules.FrontendMaskRuntime
import MerlinVerify.Sem.Facts
import MerlinVerify.Sem.Defined
import Lean.Elab.Tactic.Omega

namespace MerlinVerify.Rules.FrontendMaskBounds
open MerlinVerify FrontendMaskRuntime

/-- Fixed dimensions agree; dynamic dimensions receive their actual runtime extents. -/
def ShapeMatches : List (Option Nat) → List Nat → Prop :=
  List.Forall₂ fun pattern extent => pattern.getD extent = extent

theorem indexNat_spec {ib : Nat} {value : RVal} {n : Nat} (h : indexNat? ib value = some n) :
    0 < ib ∧ ∃ bits : BitVec ib, value = .scalar (.some (.bits ib bits)) ∧ bits.toInt = (n : Int) := by
  cases value with
  | tensor _ => cases h
  | scalar value =>
    cases value with
    | poison => cases h
    | some value =>
      cases value with
      | flt _ _ => cases h
      | bits w bits =>
        by_cases hc : 0 < ib ∧ w = ib ∧ 0 ≤ bits.toInt
        · simp only [indexNat?, ite_eq_left hc, Option.some.injEq] at h
          obtain ⟨hib, rfl, hn⟩ := hc
          subst n
          exact ⟨hib, bits, rfl, (Int.toNat_of_nonneg hn).symm⟩
        · simp [indexNat?, hc] at h

theorem indexNat_encode {ib n : Nat} (hib : 0 < ib) (hn : n < 2 ^ (ib - 1)) :
    indexNat? ib (.scalar (.some (.bits ib (BitVec.ofNat ib n)))) = some n := by
  have hb : (BitVec.ofNat ib n).toInt = (n : Int) := by
    change (BitVec.ofInt ib (n : Int)).toInt = (n : Int)
    exact BitVec.toInt_ofInt_eq_self hib (by have hpow := Int.natCast_nonneg (2 ^ (ib - 1)); norm_cast at *; omega) (by exact_mod_cast hn)
  simp [indexNat?, hib, hb]

theorem resolveShape_sound (ib : Nat) : ∀ dims args shape,
    resolveShape ib dims args = some shape →
      ShapeMatches dims shape ∧ ∀ value ∈ args, ∃ n, indexNat? ib value = some n
  | [], [], _, h => by cases h; exact ⟨.nil, by simp⟩
  | some n :: dims, args, shape, h => by
    simp only [resolveShape] at h
    cases hr : resolveShape ib dims args with
    | none => simp [hr] at h
    | some rest =>
      simp only [hr, Option.map_some, Option.some.injEq] at h
      subst shape
      obtain ⟨hm, hi⟩ := resolveShape_sound ib dims args rest hr
      exact ⟨.cons rfl hm, hi⟩
  | none :: dims, value :: args, shape, h => by
    simp only [resolveShape] at h
    cases hv : indexNat? ib value with
    | none => simp [hv] at h
    | some n =>
      cases hr : resolveShape ib dims args with
      | none => simp [hv, hr] at h
      | some rest =>
        simp [hv, hr] at h
        subst shape
        obtain ⟨hm, hi⟩ := resolveShape_sound ib dims args rest hr
        refine ⟨.cons rfl hm, ?_⟩
        intro x hx
        rcases List.mem_cons.mp hx with rfl | hx
        · exact ⟨n, hv⟩
        · exact hi x hx
  | [], _ :: _, _, h => by cases h
  | none :: _, [], _, h => by cases h

/-- Number of selected elements in a finite interval, independent of machine arithmetic. -/
def countFrom (mask : Nat → Bool) : Nat → Nat → Nat
  | 0, _ => 0
  | n + 1, start => (if mask start then 1 else 0) + countFrom mask n (start + 1)

theorem countFrom_le (mask : Nat → Bool) : ∀ n start, countFrom mask n start ≤ n
  | 0, _ => by simp [countFrom]
  | n + 1, start => by
    have h := countFrom_le mask n (start + 1)
    cases hm : mask start <;> simp [countFrom, hm] <;> omega

def PrefixInitialized (tensor : Tensor) (count : Nat) : Prop :=
  ∀ i, i < count → tensor.get [i] ≠ .poison

theorem one_inBounds {cap i : Nat} : Idx.InBounds [cap] [i] ↔ i < cap := by
  simp [Idx.inBounds_cons_cons, Idx.inBounds_nil_iff]

theorem prefix_set {tensor : Tensor} {cap count : Nat} {value : Val}
    (hs : tensor.shape = [cap]) (hc : count < cap)
    (hp : PrefixInitialized tensor count) (hv : value ≠ .poison) :
    PrefixInitialized (tensor.set [count] value).canonize (count + 1) := by
  intro i hi
  have hib : Idx.InBounds tensor.shape [i] := by rw [hs]; exact one_inBounds.mpr (by omega)
  by_cases he : i = count
  · subst i; simpa [Tensor.canonize, Tensor.canon, Tensor.set, hib] using hv
  · have hil : i < count := by omega
    simpa [Tensor.canonize, Tensor.canon, Tensor.set, hib, he] using hp i hil

/-- Mathematical gather state. The interpreter connection proves that the actual body performs
these updates; this definition is not substituted for actual operation execution. -/
def gather (mask : Nat → Bool) (values : Nat → Val) : Nat → Nat → Tensor → Nat → Tensor × Nat
  | 0, _, tensor, count => (tensor, count)
  | n + 1, start, tensor, count =>
    if mask start then gather mask values n (start + 1) ((tensor.set [count] (values start)).canonize) (count + 1)
    else gather mask values n (start + 1) tensor count

/-- At each selected iteration the write is inside capacity. The initialized prefix grows by
one; untouched poison storage never becomes an observable returned element. -/
theorem gather_invariant (mask : Nat → Bool) (values : Nat → Val) :
    ∀ n start tensor count cap,
      tensor.shape = [cap] → PrefixInitialized tensor count →
      count + countFrom mask n start ≤ cap →
      (∀ i, start ≤ i → i < start + n → values i ≠ .poison) →
      let final := gather mask values n start tensor count
      final.1.shape = [cap] ∧ final.2 = count + countFrom mask n start ∧
        PrefixInitialized final.1 final.2
  | 0, _, tensor, count, cap, hs, hp, _, _ => by
    simp only [gather, countFrom, Nat.add_zero]
    exact ⟨hs, trivial, hp⟩
  | n + 1, start, tensor, count, cap, hs, hp, hcap, hv => by
    cases hm : mask start
    · simp only [gather, hm, Bool.false_eq_true, ↓reduceIte]
      have hrest : count + countFrom mask n (start + 1) ≤ cap := by simpa [countFrom, hm] using hcap
      have ih := gather_invariant mask values n (start + 1) tensor count cap hs hp hrest
        (fun i hlo hhi => hv i (by omega) (by omega))
      simpa [countFrom, hm] using ih
    · simp only [gather, hm, ↓reduceIte]
      have hwrite : count < cap := by simp [countFrom, hm] at hcap; omega
      have hrest : count + 1 + countFrom mask n (start + 1) ≤ cap := by
        simp [countFrom, hm] at hcap; omega
      have ih := gather_invariant mask values n (start + 1)
        ((tensor.set [count] (values start)).canonize) (count + 1) cap hs
        (prefix_set hs hwrite hp (hv start (by omega) (by omega))) hrest
        (fun i hlo hhi => hv i (by omega) (by omega))
      simpa [countFrom, hm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ih

theorem prefix_full {tensor : Tensor} {cap : Nat} (hs : tensor.shape = [cap])
    (hp : PrefixInitialized tensor cap) : RVal.Defined (.tensor tensor) := by
  intro point hpoint
  rw [hs] at hpoint
  cases point with
  | nil => exact (Idx.inBounds_cons_nil cap [] hpoint).elim
  | cons i rest =>
    obtain ⟨hi, hr⟩ := (Idx.inBounds_cons_cons cap [] i rest).mp hpoint
    have he : rest = [] := (Idx.inBounds_nil_iff rest).mp hr
    subst rest
    exact hp i hi

/-- Exact-count allocation makes the final initialized prefix cover every returned element,
including the empty-selection case, where there are no observable elements. -/
theorem gather_initialized (mask : Nat → Bool) (values : Nat → Val) (n : Nat)
    (hv : ∀ i, i < n → values i ≠ .poison) :
    let capacity := countFrom mask n 0
    let final := gather mask values n 0 (Tensor.full [capacity] (.int 64) .poison) 0
    final.2 = capacity ∧ RVal.Defined (.tensor final.1) := by
  have h := gather_invariant mask values n 0 (Tensor.full [countFrom mask n 0] (.int 64) .poison)
    0 (countFrom mask n 0) rfl (by intro i hi; omega) (by omega)
    (fun i _ hi => hv i (by simpa using hi))
  refine ⟨by simpa using h.2.1, ?_⟩
  apply prefix_full h.1
  simpa only [h.2.1, Nat.zero_add] using h.2.2

/-- Updating an initialized tensor with an initialized value preserves initialized elements. -/
theorem defined_set {t : Tensor} {p : Idx} {v : Val}
    (ht : RVal.Defined (.tensor t)) (hv : v ≠ .poison) :
    RVal.Defined (.tensor (t.set p v).canonize) := by
  intro q hq
  change Idx.InBounds t.shape q at hq
  by_cases he : q = p
  · subst q; simpa [Tensor.canonize, Tensor.canon, Tensor.set, hq] using hv
  · simpa [Tensor.canonize, Tensor.canon, Tensor.set, hq, he] using ht q hq

/-- Scatter compact elements back into the initialized seed, retaining false-mask positions. -/
def scatter (mask : Nat → Bool) (compact : Tensor) : Nat → Nat → Tensor → Nat → Tensor × Nat
  | 0, _, seed, count => (seed, count)
  | n + 1, start, seed, count =>
    if mask start then scatter mask compact n (start + 1)
      ((seed.set [start] (compact.get [count])).canonize) (count + 1)
    else scatter mask compact n (start + 1) seed count

theorem scatter_invariant (mask : Nat → Bool) (compact : Tensor) :
    ∀ n start seed count cap total,
      compact.shape = [cap] → PrefixInitialized compact cap →
      seed.shape = [total] → RVal.Defined (.tensor seed) → start + n ≤ total →
      count + countFrom mask n start ≤ cap →
      let final := scatter mask compact n start seed count
      final.1.shape = [total] ∧ final.2 = count + countFrom mask n start ∧
        RVal.Defined (.tensor final.1)
  | 0, _, seed, count, cap, total, _, _, hs, hd, _, _ => by
    simp only [scatter, countFrom, Nat.add_zero]
    exact ⟨hs, trivial, hd⟩
  | n + 1, start, seed, count, cap, total, hc, hp, hs, hd, hn, hcap => by
    cases hm : mask start
    · simp only [scatter, hm, Bool.false_eq_true, ↓reduceIte]
      have hr : count + countFrom mask n (start + 1) ≤ cap := by simpa [countFrom, hm] using hcap
      simpa [countFrom, hm] using scatter_invariant mask compact n (start + 1) seed count
        cap total hc hp hs hd (by omega) hr
    · simp only [scatter, hm, ↓reduceIte]
      have hread : count < cap := by simp [countFrom, hm] at hcap; omega
      have hr : count + 1 + countFrom mask n (start + 1) ≤ cap := by
        simp [countFrom, hm] at hcap; omega
      have ih := scatter_invariant mask compact n (start + 1)
        ((seed.set [start] (compact.get [count])).canonize) (count + 1) cap total hc hp hs
        (defined_set hd (hp count hread)) (by omega) hr
      simpa [countFrom, hm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ih

theorem scatter_elt (mask : Nat → Bool) (compact : Tensor) :
    ∀ n start seed count, (scatter mask compact n start seed count).1.elt = seed.elt
  | 0, _, _, _ => rfl
  | n+1, start, seed, count => by
    cases hm : mask start <;> simp only [scatter, hm, Bool.false_eq_true, ↓reduceIte]
    · exact scatter_elt mask compact n (start+1) seed count
    · exact scatter_elt mask compact n (start+1) ((seed.set [start] (compact.get [count])).canonize) (count+1)

/-- Every explicit loop counter, including the final advancement, is representable. This is
stronger than bounding only the last executed body index. -/
theorem advancement_bound {ib lb count step k : Nat}
    (hk : k ≤ count) (h : lb + count * step < 2 ^ (ib - 1)) :
    lb + k * step < 2 ^ (ib - 1) := by
  have hm := Nat.mul_le_mul_right step hk
  omega

/-- A successful shape resolution proves initialization, width and nonnegativity of every
size operand. Fixed dimensions and the runtime extents agree position by position. -/
theorem resolved_sizes {ib : Nat} {dims : List (Option Nat)} {args : List RVal} {shape : List Nat}
    (h : resolveShape ib dims args = some shape) :
    ShapeMatches dims shape ∧ ∀ v ∈ args,
      ∃ n, 0 < ib ∧ ∃ bits : BitVec ib,
        v = .scalar (.some (.bits ib bits)) ∧ bits.toInt = (n : Int) := by
  obtain ⟨hs, hv⟩ := resolveShape_sound ib dims args shape h
  exact ⟨hs, fun v hm => let ⟨n, hn⟩ := hv v hm; ⟨n, indexNat_spec hn⟩⟩

#print axioms gather_initialized
#print axioms scatter_invariant
#print axioms resolved_sizes
end MerlinVerify.Rules.FrontendMaskBounds
