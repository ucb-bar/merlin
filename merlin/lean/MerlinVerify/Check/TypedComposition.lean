/- Registration of independently proved typed expansion families. -/
import MerlinVerify.Check.TypedRewrite

namespace MerlinVerify.Check.TypedExpansion

/-- A family that abstains delegates to the next family with the same checked input domain. -/
def orElse (first next : TypedExpansion) : TypedExpansion where
  expand ib s tys := (first.expand ib s tys).orElse fun _ => next.expand ib s tys
  sound ib s tys v h := by
    cases hf : first.expand ib s tys with
    | some w =>
      simp only [hf, Option.orElse_some, Option.some.injEq] at h
      subst v
      exact first.sound ib s tys w hf
    | none =>
      simp only [hf, Option.orElse_none] at h
      exact next.sound ib s tys v h

end MerlinVerify.Check.TypedExpansion
