/- Initialized static slices, including rank reduction. All geometry is recomputed from IR. -/
import MerlinVerify.Check.DefinedLinalg

namespace MerlinVerify.Rules.FrontendDefinedSlice
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops

theorem place_total : ∀ (sizes cs : List Nat), cs.length = (sizes.filter (· ≠ 1)).length →
    ∃ f, Sem.Ops.Tensor.placeNonUnit sizes cs = some f
  | [], [], _ => ⟨[], rfl⟩
  | [], _ :: _, h => by simp at h
  | s :: ss, cs, h => by
    by_cases hs : s = 1
    · have hl : cs.length = (ss.filter (· ≠ 1)).length := by simpa [hs] using h
      obtain ⟨f, hf⟩ := place_total ss cs hl
      exact ⟨0 :: f, by simp [Sem.Ops.Tensor.placeNonUnit, hs, hf]⟩
    · cases cs with
      | nil => simp [hs] at h
      | cons c cs =>
        have hl : cs.length = (ss.filter (· ≠ 1)).length := by simpa [hs] using h
        obtain ⟨f, hf⟩ := place_total ss cs hl
        exact ⟨c :: f, by simp [Sem.Ops.Tensor.placeNonUnit, hs, hf]⟩

theorem expand_total {sizes reduced : List Nat} {r : Idx}
    (hc : Sem.Ops.Tensor.reducedCompatible sizes reduced = true) (hr : Idx.InBounds reduced r) :
    ∃ f, Sem.Ops.Tensor.expandReduced sizes reduced r = some f := by
  have hcoords := nonUnit_coords_lt (fun (x, s) => if s = 1 then none else some x) (by intro _ _; rfl) r reduced hr.1 hr.2
  have hlength := List.Forall₂.length_eq hcoords
  simp only [Sem.Ops.Tensor.reducedCompatible, Bool.and_eq_true, beq_iff_eq] at hc
  unfold Sem.Ops.Tensor.expandReduced
  exact place_total sizes _ (by rwa [hc.1])

def rule : Definedness.Rule := fun op fs =>
  if dialectOf op.name = "tensor" ∧ op.name = "tensor.extract_slice" then
    match fs, Sem.Ops.Tensor.resultTensorTy op, Sem.Ops.Tensor.staticArray? op "static_offsets",
        Sem.Ops.Tensor.staticArray? op "static_sizes", Sem.Ops.Tensor.staticArray? op "static_strides" with
    | [⟨.tensor sh e, true⟩], some (rshape, elt), some offs, some sizes, some strides =>
      if e = elt ∧
          ¬(offs.length ≠ sh.length ∨ sizes.length ≠ sh.length ∨ strides.length ≠ sh.length) ∧
          Sem.Ops.Tensor.reducedCompatible sizes rshape = true ∧
          ((List.range sh.length).any fun d => sizes.getD d 0 ≠ 0 &&
            offs.getD d 0 + (sizes.getD d 0 - 1) * strides.getD d 0 ≥ sh.getD d 0) = false then
        some [⟨.tensor rshape elt, true⟩]
      else none
    | _, _, _, _, _ => none
  else none

theorem rule_sound (ib : Nat) : Definedness.RuleSound ib rule := by
  intro op fs rs hm h ctx call args _ ha
  unfold rule at h
  split at h
  · next hc =>
    obtain ⟨hd, hn⟩ := hc
    split at h
    · next sh e rshape elt offs sizes strides hr hoffs hsizes hstrides =>
      split at h
      · next hc =>
        obtain ⟨rfl, hlen, hcomp, hfit⟩ := hc
        cases h
        obtain _ | ⟨ht, hnil⟩ := ha
        cases hnil
        rename_i value
        cases value with
        | scalar _ => cases ht.1
        | tensor t =>
          obtain ⟨⟨hts, hte, htg⟩, hinit⟩ := ht
          subst hts hte
          rw [tensor_sem_of hm hd]
          simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr,
            staticArray?_blind, hoffs, hsizes, hstrides, hlen, ↓reduceIte, hcomp, Bool.not_true,
            Bool.false_eq_true, hfit]
          refine ⟨_, rfl, .cons ⟨⟨rfl, rfl, ?_⟩, ?_⟩ .nil⟩
          · intro r hbound
            obtain ⟨f, hf⟩ := expand_total hcomp hbound
            simp only [Tensor.canon, hbound, ↓reduceIte, hf]
            exact htg _ (slice_index_inBounds (by omega) (expandReduced_inBounds hcomp hbound hf) hfit)
          · intro _ r hbound
            change Idx.InBounds rshape r at hbound
            obtain ⟨f, hf⟩ := expand_total hcomp hbound
            simp only [Tensor.canon, hbound, ↓reduceIte, hf]
            exact hinit rfl _ (slice_index_inBounds (by omega) (expandReduced_inBounds hcomp hbound hf) hfit)
      · cases h
    · cases h
  · cases h

def book (ib : Nat) : Book opSem ib where
  rule _ op fs := if unmodelledAttr? op.blind = none ∧ op.regions = [] then rule op fs else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hc =>
      simpa only [hc.2, evalRegions] using rule_sound ib op fs rs hc.1 h ctx call args hib ha
    · cases h

#print axioms rule_sound
end MerlinVerify.Rules.FrontendDefinedSlice
