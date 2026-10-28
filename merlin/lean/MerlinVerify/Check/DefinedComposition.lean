/- Compose independently proved definedness rulebooks without weakening their obligations. -/
import MerlinVerify.Check.Defined

namespace MerlinVerify.Check.Definedness

open MerlinVerify

/-- Try the next family only when the preceding one abstains. Each family carries its own
totality and initialization proof, so adding a family cannot turn a missing proof into acceptance. -/
def Book.orElse {sem : OpSem} {ib : Nat} (first next : Book sem ib) : Book sem ib where
  rule Γ op fs := (first.rule Γ op fs).orElse fun _ => next.rule Γ op fs
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    cases hf : first.rule Γ op fs with
    | some ys =>
      simp only [hf, Option.orElse_some, Option.some.injEq] at h
      subst h
      exact first.sound Γ op fs ys hf ctx call env args hib hΓ ha
    | none =>
      simp only [hf, Option.orElse_none] at h
      exact next.sound Γ op fs rs h ctx call env args hib hΓ ha

end MerlinVerify.Check.Definedness
