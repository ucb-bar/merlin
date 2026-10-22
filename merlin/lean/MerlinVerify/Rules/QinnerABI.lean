/- Ordered ABI lemmas. The runtime must pass the same named values in the checked slot order.
These lemmas prove lookup and SSA correspondence; archive authenticity and C descriptor delivery
are explicit external obligations, not numerical laws or axioms. -/
import MerlinVerify.Check.QinnerSSA
namespace MerlinVerify.Rules.QinnerABI
open MerlinVerify MerlinVerify.Check QinnerBinding

def values (slots : List Slot) (binding : String → RVal) : List RVal :=
  slots.map fun s => binding s.key

def environment (slots : List Slot) (binding : String → RVal) : Env :=
  slots.map fun s => (s.arg.id,binding s.key)

theorem environment_zip (slots : List Slot) (binding : String → RVal) :
    (slots.map (·.arg.id)).zip (values slots binding) = environment slots binding := by
  induction slots with
  | nil => rfl
  | cons s slots ih =>
    simp only [values, environment, List.map_cons, List.zip_cons_cons, List.cons.injEq, true_and]
    simpa only [values, environment] using ih

theorem environment_supplies (slots : List Slot) (binding : String → RVal)
    (distinct : (slots.map (·.arg.id)).Nodup) :
    Supplies slots binding (environment slots binding) := by
  induction slots with
  | nil => intro s h; cases h
  | cons s slots ih =>
    intro t ht
    have nd := List.nodup_cons.mp distinct
    rcases List.mem_cons.mp ht with eq | ht
    · subst t; simp [environment, Env.get?, List.lookup]
    · have ne : (t.arg.id == s.arg.id) = false := by
        simp only [beq_eq_false_iff_ne]
        intro eq
        exact nd.1 (List.mem_map.mpr ⟨t,ht,eq⟩)
      simpa [environment, Env.get?, List.lookup, ne] using ih nd.2 t ht

theorem initial (a b : List Value) (slots : List Slot) (binding : String → RVal) (args : List RVal)
    (arity : a.length = args.length) (sameLength : b.length = a.length)
    (distinctOld : (b.map (·.id)).Nodup) (distinctExtra : (slots.map (·.arg.id)).Nodup)
    (separate : protects slots (b.map (·.id)) = true) :
    Corr ((a.map (·.id)).zip (b.map (·.id)))
      ((a.map (·.id)).zip args)
      ((b.map (·.id)).zip args ++ environment slots binding) ∧
    Supplies slots binding ((b.map (·.id)).zip args ++ environment slots binding) := by
  have empty : Corr [] [] (environment slots binding) := by
    intro x y h; simp [Rho.find, List.lookup] at h
  have corr := Corr.extend empty (a.map (·.id)) (b.map (·.id)) args
    (by simpa using arity) (by simpa [sameLength] using arity) distinctOld
    (by intro y hy p hp; cases hp)
  exact ⟨by simpa using corr, supplies_extend (environment_supplies slots binding distinctExtra) args separate⟩

#print axioms initial
end MerlinVerify.Rules.QinnerABI
