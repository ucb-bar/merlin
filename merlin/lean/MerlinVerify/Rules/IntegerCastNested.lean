/- Local scalar cast composition and exact dead-producer cleanup, suitable for the existing
BodyNormalize structured-region traversal. Each region starts with its actual typed SSA scope. -/
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Rules.IntegerCastDead

namespace MerlinVerify.Rules.IntegerCastNested

open MerlinVerify MerlinVerify.Check

/-- Recheck the composed intermediate block before typed cleanup. Captured values may supply
actual scalar inputs, but cast-origin facts are learned only within the current block. -/
def localNormalizer : TypedNormalizer :=
  IntegerCast.normalizer.sequential IntegerCastDead.normalizer

def normalizer (ib : Nat) : SplatNormalizer.Normalizer ib where
  run d Γ _ ops yields := localNormalizer.run ib d Γ ops [] yields
  sound d Γ _ ops normalized yields ctx call env hc hn he _ hi hd :=
    localNormalizer.sound ib d Γ ops [] normalized yields ctx call env hc hn he hi hd

end MerlinVerify.Rules.IntegerCastNested
