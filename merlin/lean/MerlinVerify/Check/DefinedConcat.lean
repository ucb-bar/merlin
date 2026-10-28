/- Initialized concatenation needs a covering extent proof as well as compatible parts.
Every output cell reads one in-bounds source cell; common numerical/range facts are checked
against every input, rather than copied from the first tensor. -/
import MerlinVerify.Check.DefinedReadView
import MerlinVerify.Check.DefinedCommonFacts
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Check.Definedness.Refined.Concat

open MerlinVerify Sem.Ops

theorem pick_read {ib : Nat} {shape : List Nat} {elt : Ty} {axis : Nat} {point : Idx}
    (bounds : Idx.InBounds shape point) : ∀ (parts : List Tensor) (offset : Nat),
    offset ≤ point.getD axis 0 →
    point.getD axis 0 < parts.foldl (fun total part => total + part.shape.getD axis 0) offset →
    (∀ part ∈ parts, ConcatPart ib shape elt axis part) →
    ∃ part ∈ parts, ∃ source, Idx.InBounds part.shape source ∧
      Sem.Ops.Tensor.concatPick axis point parts offset = part.get source
  | [], _, lower, upper, _ => by simp only [List.foldl_nil] at upper; omega
  | part :: parts, offset, lower, upper, compatible => by
    obtain ⟨_, rank, dimensions, _⟩ := compatible part (List.mem_cons_self ..)
    simp only [Sem.Ops.Tensor.concatPick]
    split
    · next inside =>
      refine ⟨part, List.mem_cons_self .., point.set axis (point.getD axis 0 - offset), ?_, rfl⟩
      refine ⟨by rw [List.length_set, bounds.1, rank], fun dim pointDim partDim => ?_⟩
      rw [List.getElem_set]
      split
      · next same =>
        subst same
        have pointAxis : axis < point.length := by simpa using pointDim
        rw [getD_eq_getElem pointAxis] at inside lower ⊢
        rw [getD_eq_getElem partDim] at inside
        omega
      · next different =>
        have pointDim' : dim < point.length := by simpa using pointDim
        have shapeDim : dim < shape.length := by rw [← rank]; exact partDim
        have equal := dimensions dim partDim (Ne.symm different)
        rw [getD_eq_getElem partDim, getD_eq_getElem shapeDim] at equal
        rw [equal]
        exact bounds.2 dim pointDim' shapeDim
    · next outside =>
      obtain ⟨chosen, member, source, inBounds, read⟩ := pick_read bounds parts
        (offset + part.shape.getD axis 0) (by omega) upper
        (fun next member => compatible next (List.mem_cons_of_mem _ member))
      exact ⟨chosen, List.mem_cons_of_mem _ member, source, inBounds, read⟩

variable {Info : Type} {M : Model Info}

theorem tensor_member {ib : Nat} : ∀ {tensors : List Tensor} {facts : List (Fact Info)},
    FactsHold M ib (tensors.map RVal.tensor) facts → ∀ tensor ∈ tensors,
      ∃ fact ∈ facts, fact.Holds M ib (.tensor tensor)
  | [], [], _, _, member => by cases member
  | tensor :: tensors, fact :: facts, .cons head tail, selected, member => by
    rcases List.mem_cons.1 member with rfl | member
    · exact ⟨fact, List.mem_cons_self .., head⟩
    · obtain ⟨other, found, holds⟩ := tensor_member tail selected member
      exact ⟨other, List.mem_cons_of_mem _ found, holds⟩
  | [], _ :: _, holds, _, _ => by cases holds
  | _ :: _, [], holds, _, _ => by cases holds

def rule {ib : Nat} (rules : Contextual.Structured.Invariants M ib)
    (_ : FEnv Info) (op : Op) (facts : List (Fact Info)) : Option (List (Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧
      op.name = "tensor.concat" ∧ op.regions = [] then
    match Sem.Ops.Tensor.resultTensorTy op, attrInt? op "dim" with
    | some (shape, elt), some axis =>
      if 0 ≤ axis ∧ axis.toNat < shape.length ∧ facts ≠ [] ∧
          (∀ fact ∈ facts, fact.base.initialized = true) ∧
          (facts.map (·.base.ty)).all (concatPartOk shape elt axis.toNat) = true ∧
          ((facts.map (·.base.ty)).map VTy.shape).foldl (fun total part => total + part.getD axis.toNat 0) 0 =
            shape.getD axis.toNat 0 then
        some [⟨⟨.tensor shape elt, true⟩, Common.infer rules elt (facts.map (·.info))⟩]
      else none
    | _, _ => none
  else none

theorem rule_sound (elementwise : Elementwise M) (ib : Nat) (rules : Contextual.Structured.Invariants M ib) :
    ∀ Γ op facts outputs, rule rules Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds M ib Γ env → FactsHold M ib args facts → ∃ values,
      opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values ∧ FactsHold M ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _ _ inputs
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next shape elt axis result dimension =>
      split at checked
      · next geometry =>
        obtain ⟨_, axisValid, nonempty, initialized, compatible, extent⟩ := geometry
        cases checked
        have tensorsOnly : ∀ ty ∈ facts.map (·.base.ty), ty.isTensor = true := by
          intro ty member
          have typed := List.all_eq_true.1 compatible ty member
          cases ty <;> simp_all [concatPartOk, VTy.isTensor]
        obtain ⟨tensors, rfl⟩ := tensors_of_typed inputs.typed tensorsOnly
        have shapes : tensors.map Tensor.shape = (facts.map (·.base.ty)).map VTy.shape := by
          rw [← shapes_typed inputs.typed, List.map_map]; rfl
        have parts := concat_parts inputs.typed compatible
        have size : tensors.foldl (fun total part => total + part.shape.getD axis.toNat 0) 0 =
            shape.getD axis.toNat 0 := by rw [← extent, ← shapes, List.foldl_map]
        let output := Tensor.canon shape elt fun point => Sem.Ops.Tensor.concatPick axis.toNat point tensors 0
        have outputFacts : (⟨⟨.tensor shape elt, true⟩, Common.infer rules elt (facts.map (·.info))⟩ : Fact Info).Holds M ib
            (.tensor output) := by
          apply tensorFact_of_cells elementwise rfl rfl
          intro point bounded
          change Idx.InBounds shape point at bounded
          have pointAxis : axis.toNat < point.length := by rw [bounded.1]; exact axisValid
          have covered : point.getD axis.toNat 0 <
              tensors.foldl (fun total part => total + part.shape.getD axis.toNat 0) 0 := by
            rw [size, getD_eq_getElem pointAxis, getD_eq_getElem axisValid]
            exact bounded.2 axis.toNat pointAxis axisValid
          obtain ⟨part, member, source, inBounds, read⟩ := pick_read bounded tensors 0 (Nat.zero_le _) covered parts
          obtain ⟨fact, factMember, holds⟩ := tensor_member inputs part member
          have partInfo : M.holds fact.info (.scalar (part.get source)) := (elementwise _ _).1 holds.2 source inBounds
          have base : Definedness.Fact.Holds ib (.scalar (part.get source)) ⟨.scalar elt, true⟩ := by
            refine ⟨?_, fun _ => holds.1.2 (initialized fact factMember) source inBounds⟩
            change Val.HasTy ib elt (part.get source)
            rw [← (parts part member).1]
            exact (parts part member).2.2.2 source inBounds
          have common := Common.infer_sound rules elt (facts.map (·.info))
            (List.mem_map.2 ⟨fact, factMember, rfl⟩) (show (Common.scalar elt fact.info).Holds M ib
              (.scalar (part.get source)) from ⟨base, partInfo⟩)
          simpa only [output, Tensor.canon, bounded, ↓reduceIte, read] using (show
            (⟨⟨.scalar elt, true⟩, Common.infer rules elt (facts.map (·.info))⟩ : Fact Info).Holds M ib
              (.scalar (part.get source)) from ⟨base, common⟩)
        cases tensors with
        | nil =>
          have count : facts.length = 0 := by simpa using (HasTys.length inputs.typed).symm
          exact (nonempty (List.length_eq_zero_iff.1 count)).elim
        | cons first rest =>
          have ranks : ((first :: rest).any fun part => part.shape.length ≠ shape.length) = false := by
            apply List.any_eq_false.2
            intro part member
            simp [(parts part member).2.1]
          refine ⟨[.tensor output], ?_, .cons outputFacts .nil⟩
          rw [regions, evalRegions, tensor_sem_of modelled dialect]
          simp only [Sem.Ops.Tensor.sem, Op.blind_name, named, resultTensorTy_blind, result, attrInt?_blind, dimension]
          rw [mapM_tensors _ (fun _ => rfl)]
          simp only [ranks, Bool.false_eq_true, ↓reduceIte, size, ne_eq, not_true_eq_false]
          rfl
      · cases checked
    · cases checked
  · cases checked

def book (elementwise : Elementwise M) (ib : Nat) (rules : Contextual.Structured.Invariants M ib) : Book M opSem ib where
  rule := rule rules
  sound := rule_sound elementwise ib rules

#print axioms pick_read
#print axioms tensor_member
#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.Definedness.Refined.Concat
