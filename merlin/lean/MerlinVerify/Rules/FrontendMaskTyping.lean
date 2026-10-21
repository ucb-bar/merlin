/- Integer word typing through functional gather/scatter updates. -/
import MerlinVerify.Rules.FrontendMaskBounds
import MerlinVerify.Check.Typing

namespace MerlinVerify.Rules.FrontendMaskTyping
open MerlinVerify MerlinVerify.Check FrontendMaskBounds

def Words (t : Tensor) : Prop := ∀ p, Val.HasTy 64 (.int 64) (t.get p)

theorem full_words (shape : List Nat) : Words (Tensor.full shape (.int 64) .poison) := by
  intro p
  simp [Tensor.full, Tensor.canon, Val.HasTy]

theorem set_words {t : Tensor} (ht : Words t) (p : Idx) (v : Val)
    (hv : Val.HasTy 64 (.int 64) v) : Words (t.set p v).canonize := by
  intro q
  simp only [Tensor.canonize, Tensor.canon]
  split
  · simp only [Tensor.set]
    split
    · exact hv
    · exact ht q
  · trivial

theorem gather_words (bits : Nat → Bool) (values : Nat → Val)
    (hv : ∀ i, Val.HasTy 64 (.int 64) (values i)) :
    ∀ n start t count, Words t → Words (gather bits values n start t count).1
  | 0, _, _, _, h => h
  | n+1, start, t, count, h => by
    simp only [gather]
    split
    · exact gather_words bits values hv n (start+1) _ (count+1) (set_words h _ _ (hv start))
    · exact gather_words bits values hv n (start+1) t count h

theorem scatter_words (bits : Nat → Bool) (compact : Tensor) (hc : Words compact) :
    ∀ n start t count, Words t → Words (scatter bits compact n start t count).1
  | 0, _, _, _, h => h
  | n+1, start, t, count, h => by
    simp only [scatter]
    split
    · exact scatter_words bits compact hc n (start+1) _ (count+1) (set_words h _ _ (hc [count]))
    · exact scatter_words bits compact hc n (start+1) t count h

#print axioms gather_words
#print axioms scatter_words
end MerlinVerify.Rules.FrontendMaskTyping
