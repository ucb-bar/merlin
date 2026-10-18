/- An exact source normalizer with a runtime typing premise. Hints are untrusted syntax. -/
import MerlinVerify.Check.TypedRewrite

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

structure TypedNormalizer where
  run : Nat → Nat → TEnv → List Op → List Op → List VTy → Option (List Op)
  sound : ∀ ib d Γ ss hints ns yields ctx call env,
    checkOps (merlinBook ib) d Γ ss yields = true → run ib d Γ ss hints yields = some ns →
    EnvTyped ib Γ env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
    evalOps opSem ctx call env ns = evalOps opSem ctx call env ss

namespace TypedNormalizer

/-- Compose independently proved families after checking the actual intermediate program.
The target syntax remains an untrusted hint to both families. -/
def sequential (first second : TypedNormalizer) : TypedNormalizer where
  run ib d Γ ss hints yields := do
    let middle ← first.run ib d Γ ss hints yields
    if checkOps (merlinBook ib) d Γ middle yields then
      second.run ib d Γ middle hints yields
    else none
  sound ib d Γ ss hints ns yields ctx call env hc hn he hi hd := by
    cases hm : first.run ib d Γ ss hints yields with
    | none => simp [hm] at hn
    | some middle =>
      simp only [hm, bind, Option.bind] at hn
      split at hn
      · next hcheck =>
        exact (second.sound ib d Γ middle hints ns yields ctx call env hcheck hn he hi hd).trans
          (first.sound ib d Γ ss hints middle yields ctx call env hc hm he hi hd)
      · cases hn

end TypedNormalizer

end MerlinVerify.Check
