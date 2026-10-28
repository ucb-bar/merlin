/- The provisional source interpretation respects checked SSA renaming. This proves the
semantic laws consumed by the generic program theorem; no floating identity is required. -/
import MerlinVerify.Rules.W8A8SourceSemantics
import MerlinVerify.Check.ParametricAlpha

namespace MerlinVerify.Rules.W8A8SourceAlpha

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops W8A8SourceContract

theorem eligible_congr {s t : Op} {types : List Ty} {c : Config}
    (head : opHeadEq s t = true) (arity : s.operands.length = t.operands.length)
    (regions : s.regions.isEmpty = t.regions.isEmpty) : Eligible s types c ↔ Eligible t types c := by
  simp only [opHeadEq, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at head
  obtain ⟨⟨⟨⟨named, attrs⟩, props⟩, successors⟩, results⟩ := head
  have empty : s.regions = [] ↔ t.regions = [] := by
    rw [← List.isEmpty_iff, ← List.isEmpty_iff, regions]
  have metadata : metadataOK c.kind s = metadataOK c.kind t := by
    simp only [metadataOK, Op.attr?, attrs, props]
  simp only [Eligible, named, results, arity, empty, successors, metadata]

theorem config_congr {s t : Op} (head : opHeadEq s t = true)
    (arity : s.operands.length = t.operands.length)
    (regions : s.regions.isEmpty = t.regions.isEmpty) (types : List Ty) :
    config? s types = config? t types := by
  have predicates : ∀ c, Eligible s types c ↔ Eligible t types c :=
    fun _ => eligible_congr head arity regions
  have fields := head
  simp only [opHeadEq, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at fields
  have named := fields.1.1.1.1
  have results := fields.2
  cases sr : s.results with
  | nil =>
    have tr : t.results = [] := by simpa [sr] using results.symm
    simp [config?, sr, tr]
  | cons result rest =>
    cases rest with
    | cons next tail =>
      cases tr : t.results with
      | nil => simp [sr, tr] at results
      | cons first tail' =>
        cases tail' with
        | nil => simp [sr, tr] at results
        | cons second rest' => simp [config?, sr, tr]
    | nil =>
      obtain ⟨other, tr, sameType⟩ : ∃ other, t.results = [other] ∧ result.ty = other.ty := by
        cases tr : t.results with
        | nil => simp [sr, tr] at results
        | cons other tail =>
          cases tail with
          | nil => exact ⟨other, rfl, by simpa [sr, tr] using results⟩
          | cons next rest => simp [sr, tr] at results
      simp only [config?, sr, tr, sameType, named]
      congr 1
      funext pair
      funext rest
      congr 1
      funext kind
      simp only [predicates]

theorem laws : ParametricAlpha.Laws W8A8SourceSemantics.opSem where
  operation := by
    intro ctx call s t args regions head arity empty
    have named : s.name = t.name := by
      simp only [opHeadEq, Bool.and_eq_true, beq_iff_eq] at head
      exact head.1.1.1.1
    simp only [W8A8SourceSemantics.opSem, named]
    split
    · split
      · rw [show config? s = config? t from funext (config_congr head arity empty)]
      · rfl
    · exact Sem.Ops.opSem_blind (blind_of_opHeadEq head)
  call_local := by
    intro ctx call call' op args regions agree
    unfold W8A8SourceSemantics.opSem
    split
    · rfl
    · exact Sem.Ops.opSem_call_local agree

#print axioms config_congr
#print axioms laws

end MerlinVerify.Rules.W8A8SourceAlpha
