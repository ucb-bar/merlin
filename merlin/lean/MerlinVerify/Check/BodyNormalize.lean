/- Reusable composition and region lifting for exact typed block normalizers. The recursive
traversal and capture/fact discipline come from SplatNested; local families supply their own base. -/
import MerlinVerify.Check.SplatNested

namespace MerlinVerify.Check.BodyNormalize

open MerlinVerify MerlinVerify.Sem.Ops

variable {ib : Nat}

abbrev Normalizer (ib : Nat) := SplatNormalizer.Normalizer ib

/-- Two local families compose only after independently checking the intermediate program. -/
def sequential (first second : Normalizer ib) : Normalizer ib where
  run d Γ facts ops yields := do
    let middle ← first.run d Γ facts ops yields
    if checkOps (merlinBook ib) d Γ middle yields then second.run d Γ facts middle yields else none
  sound d Γ facts ops normalized yields ctx call env hc hn he hf hi hd := by
    cases hm : first.run d Γ facts ops yields with
    | none => simp [hm] at hn
    | some middle =>
      simp only [hm, bind, Option.bind] at hn
      split at hn
      · next hcheck =>
        exact (second.sound d Γ facts middle normalized yields ctx call env hcheck hn he hf hi hd).trans
          (first.sound d Γ facts ops middle yields ctx call env hc hm he hf hi hd)
      · cases hn

/-- Enter supported generic regions using one normalizer, then run a chosen local normalizer at
this block. No region traversal or structured-loop proof is duplicated by a new rewrite family. -/
def inRegions (current nested : Normalizer ib) : Normalizer ib where
  run d Γ facts ops yields := do
    let middle ← SplatNested.liftOps nested Γ facts ops
    if checkOps (merlinBook ib) d Γ middle yields then current.run d Γ facts middle yields else none
  sound d Γ facts ops normalized yields ctx call env hc hn he hf hi hd := by
    cases hm : SplatNested.liftOps nested Γ facts ops with
    | none => simp [hm] at hn
    | some middle =>
      simp only [hm, bind, Option.bind] at hn
      split at hn
      · next hcheck =>
        exact (current.sound d Γ facts middle normalized yields ctx call env hcheck hn he hf hi hd).trans
          (SplatNested.liftOps_sound nested d Γ facts ops middle yields ctx call env hc hm he hf hi hd)
      · cases hn

/-- The finite budget is a completeness hint. Every returned layer carries its own equality
proof, so an undersized budget can only cause the final comparison to refuse a boundary. -/
def recursive (current : Normalizer ib) : Nat → Normalizer ib
  | 0 => current
  | depth + 1 => inRegions current (recursive current depth)

end MerlinVerify.Check.BodyNormalize
