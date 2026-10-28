/- Lookup-preserving commutation of an enclosing binding with region arguments. -/
import MerlinVerify.Rules.Motion

namespace MerlinVerify.Check.EnvBindingCommute

open MerlinVerify

/-- A fresh enclosing binding commutes with argument binding at every lookup.
The argument identifier/value lists may have different lengths; their actual zip determines
which bindings exist. This is lookup equality, not equality of environment list order. -/
theorem get?_singleton_zip (env : Env) (r : ValueId) (value : RVal)
    (ids : List ValueId) (values : List RVal) (hr : r ∉ ids) (x : ValueId) :
    Env.get? ((r, value) :: (ids.zip values ++ env)) x =
      Env.get? (ids.zip values ++ (r, value) :: env) x := by
  induction ids generalizing values with
  | nil => rfl
  | cons i ids ih =>
    cases values with
    | nil => rfl
    | cons v values =>
      have hne : r ≠ i := fun h => hr (h ▸ List.mem_cons_self ..)
      have htail : r ∉ ids := fun h => hr (List.mem_cons_of_mem _ h)
      simp only [List.zip_cons_cons, List.cons_append]
      rw [Rules.Motion.lookup_swap (ids.zip values ++ env) r i value v hne x]
      unfold Env.get?
      simp only [List.lookup]
      split
      · rfl
      · exact ih values htail

end MerlinVerify.Check.EnvBindingCommute
