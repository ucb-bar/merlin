/- Element predicates through the actual selection algorithm. These lemmas do not
assume a floating-point comparison law or a canonical input representation. -/
import MerlinVerify.Rules.FrontendMaskBounds
import MerlinVerify.Rules.ReshapeIndex

namespace MerlinVerify.Rules.FrontendMaskProperties
open MerlinVerify MerlinVerify.Check FrontendMaskBounds
open Sem.Ops

def Elements (P : Val → Prop) (t : Tensor) : Prop :=
  ∀ p, Idx.InBounds t.shape p → P (t.get p)

def Prefix (P : Val → Prop) (t : Tensor) (count : Nat) : Prop :=
  ∀ i, i < count → P (t.get [i])

theorem prefix_set {P : Val → Prop} {t : Tensor} {cap count : Nat} {v : Val}
    (hs : t.shape = [cap]) (hc : count < cap) (hp : Prefix P t count) (hv : P v) :
    Prefix P (t.set [count] v).canonize (count + 1) := by
  intro i hi
  have hib : Idx.InBounds t.shape [i] := by rw [hs]; exact one_inBounds.mpr (by omega)
  by_cases he : i = count
  · subst i; simpa [Tensor.canonize, Tensor.canon, Tensor.set, hib] using hv
  · simpa [Tensor.canonize, Tensor.canon, Tensor.set, hib, he] using hp i (by omega)

/-- Every written compact element came from the source. In particular no initial
poison outside the written prefix contributes an observed element. -/
theorem gather_prefix (P : Val → Prop) (mask : Nat → Bool) (values : Nat → Val) :
    ∀ n start t count cap, t.shape = [cap] → Prefix P t count →
      count + countFrom mask n start ≤ cap →
      (∀ i, start ≤ i → i < start + n → P (values i)) →
      let result := gather mask values n start t count
      Prefix P result.1 (count + countFrom mask n start)
  | 0, _, _, _, _, _, hp, _, _ => by simpa [gather, countFrom] using hp
  | n + 1, start, t, count, cap, hs, hp, hc, hv => by
    cases hm : mask start
    · simp only [gather, hm, Bool.false_eq_true, ↓reduceIte]
      simpa [countFrom, hm] using gather_prefix P mask values n (start + 1) t count cap hs hp
        (by simpa [countFrom, hm] using hc) (fun i hlo hhi => hv i (by omega) (by omega))
    · have hc' : count + 1 + countFrom mask n (start + 1) ≤ cap := by
        simp [countFrom, hm] at hc; omega
      have hw := prefix_set hs (show count < cap by omega) hp (hv start (by omega) (by omega))
      simp only [gather, hm, ↓reduceIte]
      simpa [countFrom, hm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        gather_prefix P mask values n (start + 1) (t.set [count] (values start)).canonize
          (count + 1) cap hs hw hc' (fun i hlo hhi => hv i (by omega) (by omega))

theorem elements_set {P : Val → Prop} {t : Tensor} {p : Idx} {v : Val}
    (ht : Elements P t) (hv : P v) : Elements P (t.set p v).canonize := by
  intro q hq
  change Idx.InBounds t.shape q at hq
  by_cases he : q = p
  · subst q; simpa [Tensor.canonize, Tensor.canon, Tensor.set, hq] using hv
  · simpa [Tensor.canonize, Tensor.canon, Tensor.set, hq, he] using ht q hq

/-- False-mask positions retain the seed predicate; true-mask positions receive
the compact predicate. The same argument covers an empty selection. -/
theorem scatter_elements (P : Val → Prop) (mask : Nat → Bool) (compact : Tensor) :
    ∀ n start seed count cap, Prefix P compact cap → Elements P seed →
      count + countFrom mask n start ≤ cap →
      Elements P (scatter mask compact n start seed count).1
  | 0, _, _, _, _, _, hs, _ => hs
  | n + 1, start, seed, count, cap, hp, hs, hc => by
    cases hm : mask start
    · simp only [scatter, hm, Bool.false_eq_true, ↓reduceIte]
      exact scatter_elements P mask compact n (start + 1) seed count cap hp hs
        (by simpa [countFrom, hm] using hc)
    · have hc' : count + 1 + countFrom mask n (start + 1) ≤ cap := by
        simp [countFrom, hm] at hc; omega
      simp only [scatter, hm, ↓reduceIte]
      exact scatter_elements P mask compact n (start + 1)
        (seed.set [start] (compact.get [count])).canonize (count + 1) cap hp
        (elements_set hs (hp count (by omega))) hc'

theorem reshape_elements {P : Val → Prop} {t : Tensor} (new : List Nat) (elt : Ty)
    (ht : Elements P t) (hs : shapeSize t.shape = shapeSize new) :
    Elements P (ReshapeIndex.reshape t new elt) := by
  intro p hp
  change Idx.InBounds new p at hp
  have hk : linearIndex new p < shapeSize t.shape := by rw [hs]; exact linearIndex_lt _ _ hp
  have hi := delinearize_inBounds t.shape _ hk
  simpa [ReshapeIndex.reshape, Tensor.canon, hp] using ht _ hi

#print axioms gather_prefix
#print axioms scatter_elements
end MerlinVerify.Rules.FrontendMaskProperties
