/- Shared runtime coordinate bounds for tensor read/write rules. The intervals refer to
actual current scalar bindings and retain their original Definedness.Bounds API. -/
import MerlinVerify.Check.IndexRanges

namespace MerlinVerify.Check.Definedness.Bounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

/-- A whole interval must fit each actual dimension. Unknown or uninitialized indices refuse. -/
def within : List IndexRanges.Fact → List Nat → Bool
  | [], [] => true
  | f :: fs, size :: shape =>
    match f.base, f.info with
    | ⟨.scalar .index, true⟩, some range => range.upper ≤ size && within fs shape
    | _, _ => false
  | _, _ => false

theorem within_sound {ib : Nat} : ∀ {fs : List IndexRanges.Fact} {shape : List Nat} {args : List RVal},
    Refined.FactsHold model ib args fs → within fs shape = true →
    ∃ coords, args.mapM value? = some coords ∧ coords.length = shape.length ∧
      (coords.zip shape).all (fun (x, size) => 0 ≤ x ∧ x < size) = true
  | [], [], [], _, _ => ⟨[], rfl, rfl, rfl⟩
  | f :: fs, size :: shape, v :: args, .cons hv hargs, h => by
    unfold within at h
    split at h
    · next range hb hi =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨upper, htail⟩ := h
      have hr : range.Holds v := by simpa [model, hi] using hv.2
      obtain ⟨z, hz, hlo, hhi⟩ := hr
      obtain ⟨coords, hc, hlen, hin⟩ := within_sound hargs htail
      refine ⟨z :: coords, ?_, by simp [hlen], ?_⟩
      · simp only [List.mapM_cons, hz, hc, Option.pure_def, bind, Option.bind]
      · simp only [List.zip_cons_cons, List.all_cons, hin, Bool.and_true, decide_eq_true_eq]
        constructor <;> omega
    · cases h
  | [], _ :: _, args, hargs, h => by cases hargs; cases h
  | _ :: _, [], args, hargs, h => by cases hargs; cases h
  | [], [], _ :: _, hargs, _ => by cases hargs
  | _ :: _, _ :: _, [], hargs, _ => by cases hargs

end MerlinVerify.Check.Definedness.Bounds
