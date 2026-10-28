/- Correspondence facts shared by rewrite families that remove intermediate bindings. -/
import MerlinVerify.Check.Corr

namespace MerlinVerify.Check

open MerlinVerify

theorem Corr.bindUnmapped {ρ : Rho} {envS envT : Env} {id : ValueId}
    (hc : Corr ρ envS envT) (hf : ρ.find id = none) (v : RVal) :
    Corr ρ ((id, v) :: envS) envT := by
  intro x y hm
  have hne : x ≠ id := by rintro rfl; rw [hf] at hm; cases hm
  simpa [Env.get?, List.lookup, show (x == id) = false by simpa using hne] using hc x y hm

end MerlinVerify.Check
