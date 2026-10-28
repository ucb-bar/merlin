/- Static tensor slices are total, fact-preserving reads when their actual geometry fits.
Rank reduction is checked constructively; an unsuccessful coordinate expansion cannot be
treated as an initialized result merely because the output element type is known. -/
import MerlinVerify.Check.DefinedReadView
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Check.Definedness.Refined.StaticSlice

open MerlinVerify Sem.Ops

theorem placeNonUnit_exists : ∀ (sizes coords : List Nat),
    coords.length = (sizes.filter (· ≠ 1)).length →
    ∃ point, Sem.Ops.Tensor.placeNonUnit sizes coords = some point
  | [], [], _ => ⟨[], rfl⟩
  | [], _ :: _, count => by simp at count
  | size :: sizes, coords, count => by
    by_cases unit : size = 1
    · have smaller : coords.length = (sizes.filter (· ≠ 1)).length := by
        simpa [List.filter_cons, unit] using count
      obtain ⟨point, found⟩ := placeNonUnit_exists sizes coords smaller
      exact ⟨0 :: point, by simp only [Sem.Ops.Tensor.placeNonUnit, unit, ↓reduceIte,
        found, Option.map_eq_map, Option.map_some]⟩
    · cases coords with
      | nil => simp [unit] at count
      | cons coord coords =>
        have smaller : coords.length = (sizes.filter (· ≠ 1)).length := by
          simpa [List.filter_cons, unit] using count
        obtain ⟨point, found⟩ := placeNonUnit_exists sizes coords smaller
        exact ⟨coord :: point, by simp only [Sem.Ops.Tensor.placeNonUnit, unit, ↓reduceIte,
          found, Option.map_eq_map, Option.map_some]⟩

theorem expandReduced_total {sizes reduced : List Nat} {point : Idx}
    (compatible : Sem.Ops.Tensor.reducedCompatible sizes reduced = true)
    (bounds : Idx.InBounds reduced point) :
    ∃ expanded, Sem.Ops.Tensor.expandReduced sizes reduced point = some expanded ∧ Idx.InBounds sizes expanded := by
  have matching := compatible
  simp only [Sem.Ops.Tensor.reducedCompatible, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at matching
  have paired := nonUnit_coords_lt (fun (x, s) => if s = 1 then none else some x)
    (by intro _ _; rfl) point reduced bounds.1 bounds.2
  obtain ⟨expanded, found⟩ := placeNonUnit_exists sizes
    ((point.zip reduced).filterMap fun (x, s) => if s = 1 then none else some x)
    (by rw [matching.1]; exact List.Forall₂.length_eq paired)
  exact ⟨expanded, found, expandReduced_inBounds compatible bounds found⟩

variable {Info : Type} {M : Model Info}

def rule (_ : FEnv Info) (op : Op) (facts : List (Fact Info)) : Option (List (Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧
      op.name = "tensor.extract_slice" ∧ op.regions = [] then
    match facts, Sem.Ops.Tensor.resultTensorTy op, Sem.Ops.Tensor.staticArray? op "static_offsets",
        Sem.Ops.Tensor.staticArray? op "static_sizes", Sem.Ops.Tensor.staticArray? op "static_strides" with
    | [⟨⟨.tensor sourceShape elt, true⟩, info⟩], some (resultShape, resultElt), some offsets, some sizes, some strides =>
      if elt = resultElt ∧
          ¬(offsets.length ≠ sourceShape.length ∨ sizes.length ≠ sourceShape.length ∨ strides.length ≠ sourceShape.length) ∧
          Sem.Ops.Tensor.reducedCompatible sizes resultShape = true ∧
          ((List.range sourceShape.length).any fun dim => sizes.getD dim 0 ≠ 0 &&
            offsets.getD dim 0 + (sizes.getD dim 0 - 1) * strides.getD dim 0 ≥ sourceShape.getD dim 0) = false then
        some [⟨⟨.tensor resultShape resultElt, true⟩, info⟩]
      else none
    | _, _, _, _, _ => none
  else none

theorem rule_sound (elementwise : Elementwise M) (ib : Nat) : ∀ Γ op facts outputs,
    rule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds M ib Γ env → FactsHold M ib args facts → ∃ values,
      opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values ∧ FactsHold M ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _ _ inputs
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next sourceShape elt info resultShape resultElt offsets sizes strides result offsetAttr sizeAttr strideAttr =>
      split at checked
      · next geometry =>
        obtain ⟨rfl, ranks, compatible, fits⟩ := geometry
        cases checked
        obtain _ | ⟨head, tail⟩ := inputs
        cases tail
        rename_i value
        cases value with
        | scalar _ => cases head.1.1
        | tensor tensor =>
          let output := Tensor.canon resultShape elt fun point =>
            match Sem.Ops.Tensor.expandReduced sizes resultShape point with
            | some expanded => tensor.get ((List.range sourceShape.length).map fun dim =>
                offsets.getD dim 0 + expanded.getD dim 0 * strides.getD dim 0)
            | none => .poison
          have view : ReadView tensor output := by
            intro point bounds
            change Idx.InBounds resultShape point at bounds
            obtain ⟨expanded, found, inSlice⟩ := expandReduced_total compatible bounds
            refine ⟨(List.range sourceShape.length).map (fun dim =>
              offsets.getD dim 0 + expanded.getD dim 0 * strides.getD dim 0), ?_, ?_⟩
            · rw [head.1.1.1]
              exact slice_index_inBounds (by omega) inSlice fits
            · simp only [output, Tensor.canon, bounds, ↓reduceIte, found]
          refine ⟨[.tensor output], ?_, .cons (readView_fact elementwise head rfl rfl view) .nil⟩
          rw [regions, evalRegions, tensor_sem_of modelled dialect]
          simp only [Sem.Ops.Tensor.sem, Op.blind_name, named, resultTensorTy_blind, result,
            staticArray?_blind, offsetAttr, sizeAttr, strideAttr, head.1.1.1, ranks, ↓reduceIte,
            compatible, Bool.not_true, Bool.false_eq_true, fits]
          rfl
      · cases checked
    · cases checked
  · cases checked

def book (elementwise : Elementwise M) (ib : Nat) : Book M opSem ib where
  rule := rule
  sound := rule_sound elementwise ib

#print axioms placeNonUnit_exists
#print axioms expandReduced_total
#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.Definedness.Refined.StaticSlice
